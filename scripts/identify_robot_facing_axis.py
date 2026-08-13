#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infer robot anatomical forward axes from a left/right body pair.

The script loads each robot's MJCF in MuJoCo at a fixed root yaw and uses the
default-pose hand positions to estimate anatomical forward. It deliberately
does not use ProtoMotions heading helpers, so it can independently check a
robot config's declared ``semantic_forward_axis_xy``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from protomotions.robot_configs.factory import robot_config


Z_UP_EXPLANATION = (
    "The inferred forward axis is 2D because ProtoMotions assumes Z-up robots. "
    "The left-to-right body line is projected into xy, then anatomical forward "
    "is chosen as up x (left -> right). The robot asset/config is responsible "
    "for making the default pose stand upright."
)


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _body_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in {model.nbody} MuJoCo bodies")
    return body_id


def _normalize_xy(vec: np.ndarray) -> np.ndarray:
    xy = np.asarray(vec[:2], dtype=np.float64)
    norm = np.linalg.norm(xy)
    if norm < 1e-8:
        raise ValueError(f"Cannot normalize near-zero xy vector {xy}")
    return xy / norm


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_xy(vec_xy: np.ndarray) -> float:
    return math.atan2(float(vec_xy[1]), float(vec_xy[0]))


def _facing_from_left_right(
    left_pos: np.ndarray, right_pos: np.ndarray
) -> np.ndarray:
    """Return ``up x (left -> right)`` after projecting the pair into xy."""
    left_to_right = _normalize_xy(right_pos - left_pos)
    return np.array([-left_to_right[1], left_to_right[0]], dtype=np.float64)


def _arrow_endpoints(
    root_pos: np.ndarray, facing_xy: np.ndarray, length: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return a horizontal arrow from the root along the inferred facing axis."""
    if length <= 0.0:
        raise ValueError("Arrow length must be positive")
    start = np.asarray(root_pos, dtype=np.float64).copy()
    if start.shape != (3,):
        raise ValueError("Root position must have shape (3,)")
    end = start.copy()
    end[:2] += length * _normalize_xy(facing_xy)
    return start, end


def _show_facing_arrow(result: dict, length: float) -> None:
    """Open MuJoCo's viewer with a green inferred-forward arrow at the root."""
    import time

    import mujoco.viewer

    model = result["_model"]
    data = result["_data"]
    root_pos = np.array(
        data.xpos[_body_id(model, result["root_body_name"])], dtype=np.float64
    )
    start, end = _arrow_endpoints(root_pos, result["hand_facing_xy"], length)

    print(
        f"  Opening viewer for {result['name']}; the green arrow is inferred "
        "forward. Close the window to continue."
    )
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        with viewer.lock():
            scene = viewer.user_scn
            geom = scene.geoms[0]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).reshape(-1),
                np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.035,
                start,
                end,
            )
            scene.ngeom = 1
            viewer.cam.lookat[:] = root_pos
            viewer.cam.distance = max(2.5, 3.0 * length)
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -15.0
        viewer.set_texts((None, None, "Green arrow", "Inferred semantic forward"))
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


def _first_body_name(cfg, abstract_name: str) -> str:
    value = cfg.common_naming_to_robot_body_names[abstract_name]
    if isinstance(value, str):
        return value
    return value[0]


def _optional_pair(cfg, left_key: str, right_key: str):
    try:
        return _first_body_name(cfg, left_key), _first_body_name(cfg, right_key)
    except (KeyError, IndexError, TypeError):
        return None


def _measure_pair(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_name: str,
    right_name: str,
):
    left_pos = np.array(data.xpos[_body_id(model, left_name)], dtype=np.float64)
    right_pos = np.array(data.xpos[_body_id(model, right_name)], dtype=np.float64)
    facing_xy = _facing_from_left_right(left_pos, right_pos)
    return left_pos, right_pos, facing_xy, _yaw_from_xy(facing_xy)


LOCAL_AXIS_YAWS = {
    "+X": 0.0,
    "+Y": math.pi / 2.0,
    "-X": math.pi,
    "-Y": -math.pi / 2.0,
}


def _nearest_local_axis(delta: float) -> str:
    return min(
        LOCAL_AXIS_YAWS,
        key=lambda name: abs(_wrap_angle(delta - LOCAL_AXIS_YAWS[name])),
    )


def _declared_local_axis(cfg) -> str:
    axis = np.asarray(cfg.semantic_forward_axis_xy, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return _nearest_local_axis(_yaw_from_xy(axis))


def _load_model(asset_path: Path) -> mujoco.MjModel:
    try:
        return mujoco.MjModel.from_xml_path(str(asset_path))
    except ValueError as exc:
        if "floor" not in str(exc):
            raise
        xml = asset_path.read_text()
        mesh_dir = (asset_path.parent / "../mesh/G1").resolve()
        xml = xml.replace('meshdir="../mesh/G1/"', f'meshdir="{mesh_dir}/"')
        xml = xml.replace(
            "<worldbody>",
            '<worldbody>\n        <geom name="floor" type="plane" '
            'size="20 20 0.05" pos="0 0 0"/>',
            1,
        )
        return mujoco.MjModel.from_xml_string(xml)


def verify_robot(name: str, yaw: float, verbose_config: bool = False) -> dict:
    if verbose_config:
        cfg = robot_config(name)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = robot_config(name)

    asset_path = Path(cfg.asset.asset_root) / cfg.asset.asset_file_name
    model = _load_model(asset_path)
    data = mujoco.MjData(model)

    model.opt.gravity[:] = 0.0
    if data.ctrl.size:
        data.ctrl[:] = 0.0

    if model.nq < 7:
        raise ValueError(f"{name}: expected a free-root MJCF with nq >= 7, got {model.nq}")
    qpos = np.zeros(model.nq, dtype=np.float64)
    qpos[:3] = np.array([0.0, 0.0, cfg.default_root_height], dtype=np.float64)
    qpos[3:7] = _yaw_quat_wxyz(yaw)

    if isinstance(cfg.default_dof_pos, torch.Tensor):
        default_dof = cfg.default_dof_pos.detach().cpu().numpy()
    else:
        default_dof = np.asarray(cfg.default_dof_pos)
    if model.nq - 7 != default_dof.shape[0]:
        raise ValueError(
            f"{name}: MJCF nq-7 ({model.nq - 7}) does not match robot config "
            f"default_dof_pos ({default_dof.shape[0]})"
        )
    qpos[7:] = default_dof

    data.qpos[:] = qpos
    if data.qvel.size:
        data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    left_hand = _first_body_name(cfg, "all_left_hand_bodies")
    right_hand = _first_body_name(cfg, "all_right_hand_bodies")
    left_pos, right_pos, hand_facing, hand_yaw = _measure_pair(
        model, data, left_hand, right_hand
    )
    hand_delta = _wrap_angle(hand_yaw - yaw)

    foot_result = None
    foot_pair = _optional_pair(cfg, "all_left_foot_bodies", "all_right_foot_bodies")
    if foot_pair is not None:
        try:
            _, _, foot_facing, foot_yaw = _measure_pair(model, data, *foot_pair)
            foot_delta = _wrap_angle(foot_yaw - yaw)
            foot_result = {
                "pair": foot_pair,
                "facing_xy": foot_facing,
                "yaw": foot_yaw,
                "delta": foot_delta,
                "nearest_axis": _nearest_local_axis(foot_delta),
            }
        except (ValueError, KeyError, IndexError) as exc:
            foot_result = {"error": str(exc)}

    return {
        "name": name,
        "asset_path": str(asset_path),
        "hand_pair": (left_hand, right_hand),
        "left_hand_pos": left_pos,
        "right_hand_pos": right_pos,
        "hand_facing_xy": hand_facing,
        "hand_yaw": hand_yaw,
        "hand_delta": hand_delta,
        "hand_nearest_axis": _nearest_local_axis(hand_delta),
        "declared_axis": _declared_local_axis(cfg),
        "foot": foot_result,
        "root_body_name": cfg.kinematic_info.body_names[0],
        "_model": model,
        "_data": data,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer each robot's local anatomical forward axis from its default pose.",
        epilog=Z_UP_EXPLANATION,
    )
    parser.add_argument("--yaw-deg", type=float, default=35.0)
    parser.add_argument("--robots", nargs="+", default=["smpl", "g1", "soma23"])
    parser.add_argument("--verbose-config", action="store_true")
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open a MuJoCo viewer with the inferred forward arrow at the root.",
    )
    parser.add_argument(
        "--arrow-length",
        type=float,
        default=0.75,
        help="Length in meters of the viewer arrow.",
    )
    parser.add_argument(
        "--assert-declared",
        action="store_true",
        help=(
            "Exit nonzero if RobotConfig.semantic_forward_axis_xy disagrees "
            "with the body-pair-derived facing axis."
        ),
    )
    return parser


def _print_method_explanation() -> None:
    print(Z_UP_EXPLANATION)


def main() -> None:
    args = _build_parser().parse_args()
    yaw = math.radians(args.yaw_deg)

    print(f"Fixed root heading: {args.yaw_deg:.1f} deg")
    print("Anatomical facing = up x (left hand -> right hand), after mj_forward")
    _print_method_explanation()
    print("Gravity and controls are disabled; no dynamics step is needed.\n")

    failures = []
    for name in args.robots:
        result = verify_robot(name, yaw, verbose_config=args.verbose_config)
        if result["declared_axis"] != result["hand_nearest_axis"]:
            failures.append(
                f"{name}: declared {result['declared_axis']} != "
                f"FK-derived {result['hand_nearest_axis']}"
            )
        print(f"[{name}]")
        print(f"  asset: {result['asset_path']}")
        print(f"  declared semantic axis: {result['declared_axis']}")
        print(f"  hand pair: {result['hand_pair'][0]} -> {result['hand_pair'][1]}")
        print(
            "  hand-derived facing yaw: "
            f"{math.degrees(result['hand_yaw']):7.2f} deg "
            f"(delta from root heading {math.degrees(result['hand_delta']):7.2f} deg, "
            f"nearest local axis {result['hand_nearest_axis']})"
        )
        print(
            "  hand-derived facing xy:  "
            f"[{result['hand_facing_xy'][0]: .4f}, "
            f"{result['hand_facing_xy'][1]: .4f}]"
        )
        foot = result["foot"]
        if foot and "error" not in foot:
            print(
                "  foot-pair cross-check: "
                f"{foot['pair'][0]} -> {foot['pair'][1]}, "
                f"delta {math.degrees(foot['delta']):7.2f} deg, "
                f"nearest local axis {foot['nearest_axis']}"
            )
        elif foot:
            print(f"  foot-pair cross-check unavailable: {foot['error']}")
        if args.view:
            _show_facing_arrow(result, args.arrow_length)
        print()

    if failures:
        message = "\n".join(failures)
        if args.assert_declared:
            raise SystemExit(message)
        print("Declaration mismatches:")
        print(message)


if __name__ == "__main__":
    main()
