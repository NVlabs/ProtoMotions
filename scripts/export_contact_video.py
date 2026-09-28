#!/usr/bin/env python3
"""Render one P2 packaged MotionLib motion with synchronized contact timelines.

Run with the remote astro-p2-retarget Python environment. No physics rollout.
Motion IDs are zero based, local to the concrete --motion-file (not global IDs).
"""
import argparse
import csv
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

REPO = Path("/data/chenguanting/ProtoMotions")
DEFAULT_MOTION = Path("/data/chenguanting/datasets/BONES-SEED/p2_motionlib/bones_seed_train_astro_p2_0.pt")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("motion_id", type=int, help="Zero-based ID within --motion-file")
    p.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION)
    p.add_argument("--model", type=Path, default=REPO / "protomotions/data/assets/astro_p2/mjcf/astro_p2_protomotions.xml")
    p.add_argument("--output", type=Path, help="MP4 path; CSV, JSON and preview PNG saved alongside")
    p.add_argument("--bodies", nargs="+", default=["left_ankle_roll_link", "right_ankle_roll_link"])
    p.add_argument("--smooth-window", type=int, default=7)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=640, help="3D viewport height, excluding timeline")
    p.add_argument("--collision", action="store_true", help="Show collision geoms instead of visual meshes")
    a = p.parse_args()
    if a.smooth_window < 1 or a.smooth_window % 2 != 1:
        p.error("--smooth-window must be a positive odd number")
    if a.width < 640 or a.height < 320 or a.width % 2 or a.height % 2:
        p.error("Use even dimensions, width >= 640 and height >= 320")
    if "slurmrank" in str(a.motion_file):
        p.error("Use a concrete shard filename, e.g. _0.pt, not slurmrank")
    return a


def make_model(path, width, height):
    tree = ET.parse(path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir"):
            if compiler.get(key):
                compiler.set(key, str((path.parent / compiler.get(key)).resolve()))
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", str(width))
    glob.set("offheight", str(height))
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="contact_video_floor", type="plane", size="0 0 .1", rgba=".23 .28 .34 1", contype="0", conaffinity="0", group="0")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", directional="true", diffuse=".8 .8 .8")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def main():
    a = arguments()
    torch.set_num_threads(1)
    packed = torch.load(a.motion_file, map_location="cpu", weights_only=False, mmap=True)
    count = len(packed["motion_num_frames"])
    if not 0 <= a.motion_id < count:
        raise ValueError(f"motion_id must be in [0, {count - 1}] for {a.motion_file}")
    start = int(packed["length_starts"][a.motion_id])
    n = int(packed["motion_num_frames"][a.motion_id])
    dt = float(packed["motion_dt"][a.motion_id])
    if n < 1 or dt <= 0:
        raise ValueError("Motion has no frames or invalid dt")
    sl = slice(start, start + n)
    pos = packed["gts"][sl].numpy()
    quat = packed["grs"][sl].numpy()  # XYZW
    dof = packed["dps"][sl].numpy()
    if packed.get("contacts") is None:
        raise ValueError("MotionLib contains no contacts")
    model = make_model(a.model.resolve(), a.width, a.height)
    data = mujoco.MjData(model)
    # P2 packaged order is the MJCF body/joint traversal order. Validate by FK.
    body_ids = [i for i in range(1, model.nbody) if model.body_jntnum[i] > 0]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in body_ids]
    if len(names) != pos.shape[1] or model.nq != 7 + dof.shape[1]:
        raise ValueError("Model does not match packaged body/DOF dimensions")
    columns = [names.index(name) for name in a.bodies]
    raw = packed["contacts"][sl][:, columns].float().numpy()
    if not np.isfinite(raw).all() or not np.isin(raw, [0, 1]).all():
        raise ValueError("Expected unsmoothed binary contacts, as in the training MotionLib")
    pad = a.smooth_window // 2
    soft = np.stack([np.convolve(np.pad(raw[:, j], (pad, pad), mode="edge"), np.ones(a.smooth_window) / a.smooth_window, mode="valid") for j in range(len(columns))], axis=1)

    def set_frame(i):
        data.qpos[:3] = pos[i, 0]
        data.qpos[3:7] = quat[i, 0, [3, 0, 1, 2]]
        data.qpos[7:] = dof[i]
        mujoco.mj_forward(model, data)

    error = 0.
    for i in sorted({0, n // 2, n - 1}):
        set_frame(i)
        error = max(error, float(np.abs(data.xpos[body_ids] - pos[i]).max()))
    if error > 1e-4:
        raise ValueError(f"FK mismatch {error:.6g} m; inspect model and body/DOF ordering")
    out = a.output or Path("output/contact_videos") / f"{a.motion_file.stem}_motion_{a.motion_id:05d}.mp4"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    source = str(packed["motion_files"][a.motion_id])
    times = np.arange(n) * dt
    with out.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s"] + [f"{name}_{kind}" for name in a.bodies for kind in ("raw", "smoothed")])
        for i in range(n):
            writer.writerow([i, times[i]] + [v for j in range(len(columns)) for v in (float(raw[i, j]), float(soft[i, j]))])
    meta = dict(motion_id=a.motion_id, id_scope="zero-based within motion_file", motion_file=str(a.motion_file.resolve()), source=source, model=str(a.model.resolve()), bodies=a.bodies, frames=n, fps=1/dt, last_frame_time_s=float(times[-1]), smooth_window=a.smooth_window, fk_max_abs_error_m=error, rendering="kinematic FK, no policy or physics simulation", quaternion_order="XYZW in MotionLib; WXYZ in MuJoCo")
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 16) if Path(font_path).exists() else ImageFont.load_default()
    small = ImageFont.truetype(font_path, 13) if Path(font_path).exists() else font
    panel_h = 82 + 84 * len(columns)
    panel = Image.new("RGB", (a.width, panel_h), (18, 23, 32))
    draw = ImageDraw.Draw(panel)
    draw.text((18, 8), "REFERENCE CONTACT  |  bars: raw label   line: smoothed training target", fill="white", font=font)
    x0, x1 = 240, a.width - 24
    duration = max(float(times[-1]), dt)
    xx = x0 + times / duration * (x1 - x0)
    colors = [(70, 211, 174), (94, 169, 255), (242, 179, 82)]
    for j, name in enumerate(a.bodies):
        y = 42 + 84*j
        color = colors[j % len(colors)]
        draw.text((18, y), name.replace("_link", ""), fill=color, font=small)
        draw.rectangle((x0, y, x1, y+50), fill=(32, 40, 52))
        # Max over each pixel's source interval preserves short contact pulses.
        for px in range(x1-x0):
            lo = min(n-1, int(px/(x1-x0)*max(n-1, 1)))
            hi = min(n, max(lo+1, int((px+1)/(x1-x0)*max(n-1, 1))+1))
            if raw[lo:hi, j].max() > .5:
                draw.line((x0+px, y+34, x0+px, y+49), fill=color)
        points = [(float(x), float(y+29-v*27)) for x,v in zip(xx, soft[:,j])]
        if len(points)>1:
            draw.line(points, fill=(245, 245, 245), width=2)
    for t in np.linspace(0, float(times[-1]), 6):
        x = x0+t/duration*(x1-x0)
        draw.text((x-15, panel_h-27), f"{t:.1f}s", fill=(175,185,202), font=small)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    option.geomgroup[3 if a.collision else 1] = 1
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.7, 135, -18
    tmp = out.with_name(out.stem + ".partial.mp4")
    print(json.dumps(meta, indent=2), flush=True)
    try:
        with mujoco.Renderer(model, height=a.height, width=a.width) as renderer, imageio.get_writer(str(tmp), fps=1/dt, codec="libx264", quality=8, macro_block_size=2, ffmpeg_log_level="error") as writer:
            for i in range(n):
                set_frame(i)
                camera.lookat[:] = (pos[i,0,0], pos[i,0,1], max(.55, float(pos[i,0,2])))
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = Image.new("RGB", (a.width, a.height+panel_h))
                frame.paste(Image.fromarray(renderer.render()), (0,0))
                frame.paste(panel, (0,a.height))
                d = ImageDraw.Draw(frame)
                d.rectangle((0,0,a.width,67), fill=(18,23,32))
                d.text((18,8), f"P2 RETARGET (kinematic) | motion {a.motion_id} | frame {i+1}/{n} | {times[i]:.3f}s", fill="white", font=font)
                label = Path(source).name
                while d.textbbox((0,0), label, font=small)[2] > a.width-36:
                    label = "..." + label[4:]
                d.text((18,38), label, fill=(180,190,205), font=small)
                cursor = float(xx[i])
                d.line((cursor,a.height+36,cursor,a.height+panel_h-30), fill=(255,105,100), width=2)
                for j in range(len(columns)):
                    y = a.height+42+84*j
                    d.text((18,y+25), f"raw={int(raw[i,j])}   smooth={soft[i,j]:.2f}", fill="white", font=small)
                writer.append_data(np.asarray(frame))
                if i == n//2:
                    frame.save(out.with_suffix(".png"))
                if i % 150 == 0:
                    print(f"Rendered {i+1}/{n}", flush=True)
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
