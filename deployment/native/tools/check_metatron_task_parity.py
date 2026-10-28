# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native v5 idle/moving targets to the Python deployment functions."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from tools.metatron_v1_reference import (
    _apply_steering_targets,
    _steering_target_masks,
)
from examples.metatron_goals import MetatronGoals
from examples.metatron_v1_contract import BOOL_INPUTS, INPUT_SHAPES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--rest-dump", type=Path, required=True)
    parser.add_argument("--fk-plan", type=Path, required=True)
    parser.add_argument("--recovery-plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rest = json.loads(
        subprocess.check_output(
            [str(args.rest_dump), str(args.fk_plan)],
            text=True,
        )
    )
    pos = np.asarray(rest["positions"], np.float32).reshape(33, 3)
    rot = np.asarray(rest["rotations"], np.float32).reshape(33, 4)
    style = np.asarray(rest["style"], np.float32)
    tested = 0
    for mode, heading, velocity in (
        ("idle", 0.0, (0.0, 0.0)),
        ("steering", 0.0, (0.0, 0.0)),
        ("steering", 0.5, (0.6, 0.2)),
        ("steering", -0.5, (0.3, -0.8)),
        ("steering", 1.2, (-0.4, 0.2)),
    ):
        native = json.loads(
            subprocess.check_output(
                [
                    str(args.dump),
                    str(args.fk_plan),
                    str(args.recovery_plan),
                    str(args.manifest),
                    mode,
                    str(heading),
                    f"{velocity[0]},{velocity[1]}",
                ],
                text=True,
            )
        )
        if native["recovery_phase"] != 0:
            raise RuntimeError("neutral test entered recovery unexpectedly")
        measured_pos = np.asarray(native["measured_positions"], np.float32).reshape(
            33, 3
        )
        measured_rot = np.asarray(native["measured_rotations"], np.float32).reshape(
            33, 4
        )
        anchor_pos = np.asarray(native["anchor_position"], np.float32)
        raw_pos = np.repeat(measured_pos[None], 5, axis=0)
        raw_rot = np.repeat(measured_rot[None], 5, axis=0)
        arrays = {
            key: np.zeros(shape, np.bool_ if key in BOOL_INPUTS else np.float32)
            for key, shape in INPUT_SHAPES.items()
        }
        p = "masked_mimic."
        arrays[p + "ref_pos"][:] = raw_pos
        arrays[p + "ref_rot"][:] = raw_rot
        arrays[p + "time_offsets"][:] = [0.1, 0.2, 0.3, 0.4, 0.5]
        if mode == "idle":
            arrays[p + "target_bodies_masks"].reshape(5, 5, 2)[-1, 0, 1] = True
            arrays[p + "target_poses_masks"][-1] = True
            arrays[p + "anchor_rotation_mask"][-1, 1] = True
            arrays[p + "ref_rot"][-1, 16] = [0, 0, 0, 1]
            standing = True
        else:
            vel = np.asarray(velocity, np.float32)
            speed = float(np.linalg.norm(vel))
            stationary = speed <= 0.05
            feet = measured_pos[[13, 7]] if stationary else None
            stationary_anchor = None
            if stationary:
                stationary_anchor = np.array(
                    [
                        feet[:, 0].mean(),
                        feet[:, 1].mean(),
                        feet[:, 2].min()
                        + rest["upright_height"]
                        - pos[[13, 7], 2].min(),
                    ],
                    np.float32,
                )
            _apply_steering_targets(
                raw_pos,
                raw_rot,
                anchor_pos,
                0.0,
                heading,
                vel,
                float(measured_pos[:, 2].min() + rest["upright_height"]),
                pos,
                rot,
                0.0,
                stationary_foot_positions=feet,
                stationary_anchor_position=stationary_anchor,
                stationary_heading=0.0 if stationary else None,
            )
            masks, pose_masks, translation, rotation = _steering_target_masks(
                stationary=stationary,
                feet_active=not stationary,
                torso_height=True,
            )
            arrays[p + "target_bodies_masks"][:] = masks.reshape(-1).astype(bool)
            arrays[p + "target_poses_masks"][:] = pose_masks.astype(bool)
            arrays[p + "anchor_translation_mask"][:] = translation.astype(bool)
            arrays[p + "anchor_rotation_mask"][:] = rotation.astype(bool)
            arrays[p + "ref_pos"][:] = raw_pos
            arrays[p + "ref_rot"][:] = raw_rot
            arrays[p + "time_offsets"][:] = [0.1, 0.2, 0.3, 0.4, 0.5]
            arrays[p + "average_velocity"][:] = vel
            arrays[p + "velocity_command_mask"][:] = True
            standing = speed <= 0.03 and abs(heading) <= np.pi / 9
        MetatronGoals(pos, rot, style).apply(arrays, mode, standing)
        checked = (
            "anchor_rotation_mask",
            "anchor_translation_mask",
            "average_velocity",
            "local_ref_pos",
            "local_ref_rot",
            "local_target_bodies_masks",
            "local_target_poses_masks",
            "local_time_offsets",
            "ref_pos",
            "ref_rot",
            "style_condition_masks",
            "style_frame_masks",
            "style_pose_samples",
            "style_velocity_samples",
            "target_bodies_masks",
            "target_poses_masks",
            "time_offsets",
            "velocity_command_mask",
        )
        for name in checked:
            expected = arrays[p + name].reshape(-1)
            actual = np.asarray(native[(p + name).replace(".", "_")], expected.dtype)
            if np.issubdtype(expected.dtype, np.bool_):
                if not np.array_equal(expected, actual):
                    raise AssertionError(
                        f"{mode} {heading} {velocity}: {name} mask mismatch"
                    )
            else:
                error = float(np.max(np.abs(expected - actual)))
                if error > 2e-5:
                    raise AssertionError(
                        f"{mode} {heading} {velocity}: {name} max error {error}"
                    )
        tested += 1
    print(f"Metatron task target parity passed {tested} idle/steering cases")


if __name__ == "__main__":
    main()
