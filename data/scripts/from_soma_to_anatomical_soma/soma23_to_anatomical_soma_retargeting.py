# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed and diagnostics for source SOMA to ordered-hinge anatomical SOMA."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


CANONICAL_FRAME = "+X-forward,+Y-left,+Z-up"
SOMA_TO_CANONICAL = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# Residual multipliers, not squared-loss weights. As in the G1 retargeter,
# intermediate joints yield to the task goals; their orientations must not
# prevent a wrist or foot from reaching its target.
SOMA_POSITION_WEIGHTS = {
    "Hips": 1.0,
    "Spine1": 0.25,
    "Spine2": 0.25,
    "Chest": 1.0,
    "Neck1": 0.25,
    "Neck2": 0.25,
    "Head": 1.0,
    "RightShoulder": 0.25,
    "RightArm": 0.25,
    "RightForeArm": 0.25,
    "RightHand": 1.0,
    "LeftShoulder": 0.25,
    "LeftArm": 0.25,
    "LeftForeArm": 0.25,
    "LeftHand": 1.0,
    "RightLeg": 0.25,
    "RightShin": 0.25,
    "RightFoot": 1.0,
    "RightToeBase": 1.0,
    "LeftLeg": 0.25,
    "LeftShin": 0.25,
    "LeftFoot": 1.0,
    "LeftToeBase": 1.0,
}
SOMA_ORIENTATION_WEIGHTS = {
    name: (
        1.0
        if name
        in {
            "Hips",
            "Chest",
            "Head",
            "LeftHand",
            "RightHand",
            "LeftFoot",
            "RightFoot",
            "LeftToeBase",
            "RightToeBase",
        }
        else 0.05
    )
    for name in SOMA_POSITION_WEIGHTS
}
SOMA_BONE_PAIRS = (
    ("Hips", "Chest"),
    ("Chest", "Head"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("RightLeg", "RightShin"),
    ("RightShin", "RightFoot"),
    ("RightFoot", "RightToeBase"),
    ("LeftLeg", "LeftShin"),
    ("LeftShin", "LeftFoot"),
    ("LeftFoot", "LeftToeBase"),
)


def canonical_soma_poses(positions, rotations):
    """Change world and body bases once, preserving the source trajectory/timing.

    Source SOMA faces -Y with +X left. Quaternions are scalar-last; both world
    poses and the target asset's local frames use +X forward, +Y left, +Z up.
    """
    basis = SOMA_TO_CANONICAL
    matrices = Rotation.from_quat(rotations.reshape(-1, 4)).as_matrix()
    converted = Rotation.from_matrix(basis @ matrices @ basis.T).as_quat()
    return positions @ basis.T, converted.reshape(rotations.shape)


def canonical_soma_scenes(data):
    """Rotate serialized SceneLib world poses; keep object-local geometry intact.

    Object meshes do not change basis, unlike the robot's local body frames.
    Their rotations therefore use C @ R, not C @ R @ C.T.
    """
    result = copy.deepcopy(data)
    scenes = result["original_scenes"]
    if len(scenes) != result["num_original_scenes"]:
        raise ValueError("Scene count does not match serialized scene metadata")
    basis = SOMA_TO_CANONICAL
    for scene in scenes:
        offset = np.asarray(scene["offset"], dtype=float)
        if offset.shape not in ((2,), (3,)) or not np.isfinite(offset).all():
            raise ValueError("Expected a finite 2D/3D scene offset")
        scene["offset"] = (basis[: len(offset), : len(offset)] @ offset).tolist()
        if len(scene["objects"]) != result["num_objects_per_scene"]:
            raise ValueError("Scene object count does not match metadata")
        for obj in scene["objects"]:
            p, q = (
                np.asarray(obj["translation"], dtype=float),
                np.asarray(obj["rotation"], dtype=float),
            )
            if p.ndim != 2 or p.shape[1] != 3 or q.shape != (len(p), 4):
                raise ValueError(
                    "Expected matching object translation/rotation frame arrays"
                )
            if (
                not np.isfinite(p).all()
                or not np.isfinite(q).all()
                or not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-5)
            ):
                raise ValueError("Invalid scene object pose")
            obj["translation"] = (p @ basis.T).tolist()
            obj["rotation"] = (
                Rotation.from_matrix(basis @ Rotation.from_quat(q).as_matrix())
                .as_quat()
                .tolist()
            )
    return result


def semantic_names(asset):
    return (
        ET.parse(asset)
        .find("custom/text[@name='semantic_body_names']")
        .attrib["data"]
        .split()
    )


def hinge_seed(asset, global_rotations):
    """Bounded seed only: choose nearby equivalent branches across each clip.

    The optimizer fits ORIGINAL poses, not these clipped coordinates. Signed
    orthogonal-axis triples are split across zero-length links in the MJCF.
    """
    tree = ET.parse(asset)
    names = semantic_names(asset)
    bodies = list(tree.find("worldbody").iter("body"))
    parent = {
        c.attrib["name"]: b.attrib["name"] for b in bodies for c in b.findall("body")
    }
    by_name = {b.attrib["name"]: b for b in bodies}
    groups = {name: [] for name in names[1:]}
    for body in bodies:
        for joint in body.findall("joint"):
            if joint.get("type", "hinge") == "hinge":
                groups[joint.attrib["name"].rsplit("_", 1)[0]].append(joint)
    rotations = (
        Rotation.from_quat(global_rotations.reshape(-1, 4))
        .as_matrix()
        .reshape(*global_rotations.shape[:-1], 3, 3)
    )
    result, dof_names = [], []
    for i, name in enumerate(names[1:], 1):
        p = parent[name]
        while p not in names:
            p = parent[p]
        local = rotations[:, names.index(p)].swapaxes(-1, -2) @ rotations[:, i]
        joints = groups[name]
        # Absorb fixed mounting rotations into the joint axes, leaving the
        # semantic body's rest rotation as a right-hand factor of the chain.
        chain = []
        node = name
        while node != p:
            chain.append(by_name[node])
            node = parent[node]
        rest = np.eye(3)
        mounted_axes = {}
        for body in reversed(chain):
            if any(
                key in body.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")
            ):
                raise ValueError("Seed mounting frames must use explicit quaternions")
            quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
            rest = rest @ Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            for joint in body.findall("joint"):
                if joint in joints:
                    mounted_axes[joint.get("name")] = rest @ np.fromstring(
                        joint.get("axis"), sep=" "
                    )
        axes = np.array([mounted_axes[j.get("name")] for j in joints])
        local = local @ rest.T
        limits = np.deg2rad([np.fromstring(j.attrib["range"], sep=" ") for j in joints])
        if len(joints) == 1:
            quat = Rotation.from_matrix(local).as_quat()
            raw = (
                (2 * np.arctan2(quat[:, :3] @ axes[0], quat[:, 3]) + np.pi)
                % (2 * np.pi)
                - np.pi
            )[:, None, None]
        elif len(joints) == 3 and np.allclose(abs(axes) @ abs(axes).T, np.eye(3)):
            axis_ids = np.argmax(abs(axes), axis=1)
            sequence = "".join("XYZ"[index] for index in axis_ids)
            signs = axes[np.arange(3), axis_ids]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                euler = Rotation.from_matrix(local).as_euler(sequence)
            alternate = euler + [np.pi, 0, np.pi]
            alternate[:, 1] = np.pi - euler[:, 1]
            raw = np.stack((euler, alternate), axis=1) * signs
        else:
            raise ValueError(f"Unsupported seed axis group: {name}")
        raw += 2 * np.pi * np.round((limits.mean(axis=1) - raw) / (2 * np.pi))
        bounded = np.clip(raw, limits[:, 0], limits[:, 1])
        if len(joints) == 1:
            candidates = Rotation.from_rotvec(
                bounded.reshape(-1, 1) * axes[0]
            ).as_matrix()
        else:
            candidates = Rotation.from_euler(
                sequence, (bounded * signs).reshape(-1, 3)
            ).as_matrix()
        candidates = candidates.reshape(len(local), -1, 3, 3)
        residual = (
            Rotation.from_matrix(
                (local[:, None].swapaxes(-1, -2) @ candidates).reshape(-1, 3, 3)
            )
            .magnitude()
            .reshape(len(local), -1)
        )
        selected = []
        previous = bounded[0, residual[0].argmin()]
        for frame in range(len(local)):
            score = residual[frame] ** 2 + 0.05 * (
                (bounded[frame] - previous) ** 2
            ).sum(axis=-1)
            previous = bounded[frame, score.argmin()]
            selected.append(previous)
        result.append(np.stack(selected))
        dof_names.extend(j.attrib["name"] for j in joints)
    return np.concatenate(result, axis=1), dof_names


def quality_metrics(
    reference_pos, reference_rot, actual_pos, actual_rot, q, names=None
):
    position_error = np.linalg.norm(actual_pos - reference_pos, axis=-1)
    ref = Rotation.from_quat(reference_rot.reshape(-1, 4))
    out = Rotation.from_quat(actual_rot.reshape(-1, 4))
    angle_error = (ref.inv() * out).magnitude()
    step = np.abs(np.diff(q, axis=0))
    metrics = {
        "position_mean_m": float(position_error.mean()),
        "position_p95_m": float(np.percentile(position_error, 95)),
        "position_max_m": float(position_error.max()),
        "orientation_mean_rad": float(angle_error.mean()),
        "orientation_max_rad": float(angle_error.max()),
        "joint_step_max_rad": float(step.max()) if step.size else 0.0,
        "steps_over_pi": int((step > np.pi).any(axis=1).sum()),
    }
    if names is not None:
        angles = angle_error.reshape(position_error.shape)
        metrics["bodies"] = {
            name: {
                "position_rmse_m": float(np.sqrt(np.mean(position_error[:, i] ** 2))),
                "position_max_m": float(position_error[:, i].max()),
                "orientation_rmse_rad": float(np.sqrt(np.mean(angles[:, i] ** 2))),
            }
            for i, name in enumerate(names)
        }
    return metrics


def save_verified(obj, destination, reload_matches):
    """Save ``obj`` to ``destination`` through a uniquely named file in the same directory.

    The file is reloaded and ``reload_matches(loaded)`` must hold before it is
    moved into place, so a bad write never replaces an existing output. A fixed
    temporary name such as ``<stem>.partial.pt`` would overwrite a source or
    another job's file that happened to share it; the unique name cannot.
    """
    import torch

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".partial",
        delete=False,
    )
    handle.close()
    partial = Path(handle.name)
    try:
        torch.save(obj, partial)
        loaded = torch.load(partial, weights_only=True, map_location="cpu")
        if not reload_matches(loaded):
            raise ValueError(f"Saved-output reload mismatch for {destination}")
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()
