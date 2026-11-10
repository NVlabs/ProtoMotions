# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a native-launcher metrics record (JSONL) as an MP4 with MuJoCo.

The record carries the root pose and the 29 joint angles every cycle, so the
robot can be re-posed kinematically and drawn offscreen without re-running the
policy. Hand goals and the planted support frame are drawn as markers; a label
with the run name, mode and clock is burned into every frame.

    MUJOCO_GL=egl python tools/render_metrics_replay.py --xml g1_holo_compat_sim.xml \
        --metrics run/metrics.jsonl --out run/replay.mp4 --title "teleop reach, IK on"
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 25
WIDTH, HEIGHT = 960, 544
JOINTS = 29
MODES = {0: "idle", 1: "steering", 2: "teleop"}


def load(path: Path):
    frames = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def vector(frame: dict, prefix: str, n: int):
    try:
        return np.array([frame[f"{prefix}/{i}"] for i in range(n)], dtype=float)
    except (KeyError, TypeError):
        return None


def marker_sphere(scene: mujoco.MjvScene, pos, rgba, size=0.03):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]), np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    y = 10
    for text in lines:
        draw.rectangle([8, y - 2, 12 + draw.textlength(text, font=font), y + 24], fill=(0, 0, 0, 160))
        draw.text((10, y), text, fill=(255, 255, 255), font=font)
        y += 28
    return np.asarray(pil)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Metatron native launcher")
    parser.add_argument("--start", type=float, default=0.0, help="seconds")
    parser.add_argument("--end", type=float, default=None, help="seconds")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    data = mujoco.MjData(model)
    joint_qpos = []
    for i in range(JOINTS):
        joint = int(model.actuator_trnid[i][0])
        joint_qpos.append(model.jnt_qposadr[joint])
    free_adr = next(model.jnt_qposadr[j] for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.2
    camera.elevation = -18
    camera.azimuth = 135
    frames = load(Path(args.metrics))
    if not frames:
        raise SystemExit("empty metrics record")
    t0 = frames[0]["t_ns"] * 1e-9
    writer = imageio_ffmpeg.write_frames(args.out, (WIDTH, HEIGHT), fps=FPS, codec="libx264",
                                         output_params=["-pix_fmt", "yuv420p", "-crf", "20"])
    writer.send(None)
    next_frame_time = args.start
    written = 0
    for frame in frames:
        t = frame["t_ns"] * 1e-9 - t0
        if t < next_frame_time:
            continue
        if args.end is not None and t > args.end:
            break
        next_frame_time += 1.0 / FPS
        q = vector(frame, "joint/q", JOINTS)
        root = vector(frame, "pose/root", 3)
        quat = vector(frame, "pose/root_quat", 4)  # xyzw
        if q is None or root is None or quat is None:
            continue
        data.qpos[free_adr:free_adr + 3] = root
        data.qpos[free_adr + 3:free_adr + 7] = [quat[3], quat[0], quat[1], quat[2]]
        for i, adr in enumerate(joint_qpos):
            data.qpos[adr] = q[i]
        mujoco.mj_kinematics(model, data)
        camera.lookat[:] = [root[0], root[1], 0.8]
        renderer.update_scene(data, camera=camera)
        scene = renderer.scene
        for side, color in (("left", (0.2, 0.8, 0.3, 0.9)), ("right", (0.95, 0.6, 0.1, 0.9))):
            goal = vector(frame, f"goal/{side}", 3)
            if goal is not None:
                marker_sphere(scene, goal, color, 0.035)
        # Tracking: the reference pose of the five conditionable bodies, violet.
        for c in range(5):
            ref = vector(frame, f"tracking/ref/{c}", 3)
            if ref is not None:
                marker_sphere(scene, ref, (0.75, 0.55, 1.0, 0.8), 0.03)
        if frame.get("support/valid"):
            center = vector(frame, "support/center", 3)
            if center is not None:
                marker_sphere(scene, center + np.array([0, 0, 0.01]), (0.3, 0.5, 1.0, 0.6), 0.05)
        image = renderer.render()
        mode = frame.get("task/mode_name") or MODES.get(int(frame.get("task/mode") or 0), "?")
        policy = frame.get("policy/name") or "Metatron v1 (epoch 6410)"
        lines = [args.title, f"POLICY={policy}  mode={mode}  t={t:5.2f} s"]
        motion = frame.get("tracking/motion")
        if isinstance(motion, str) and motion:
            lines.append(f"tracking {motion}  {frame.get('tracking/elapsed_s', 0):.1f}/{frame.get('tracking/duration_s', 0):.1f} s"
                         f"  body error {100 * (frame.get('tracking/mean_body_error_m') or 0):.1f} cm"
                         f"  rotation {frame.get('tracking/mean_rotation_error_deg') or 0:.0f} deg"
                         "  violet = reference")
        err_l, err_r = frame.get("task/hand_error_left_m"), frame.get("task/hand_error_right_m")
        if err_l is not None or err_r is not None:
            lines.append(f"hand error L={100 * (err_l or 0):.1f} cm  R={100 * (err_r or 0):.1f} cm"
                         + (f"  ik offset L={frame.get('ik/offset_left_rad', 0):.3f} R={frame.get('ik/offset_right_rad', 0):.3f} rad" if "ik/offset_left_rad" in frame else ""))
        writer.send(np.ascontiguousarray(label(image, lines)))
        written += 1
    writer.close()
    print(f"wrote {args.out}: {written} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
