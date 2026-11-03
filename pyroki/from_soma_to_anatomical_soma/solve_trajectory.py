# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-clip PyRoki solve for the anatomic SOMA scalar-hinge asset.

Called by data/scripts/from_soma_to_anatomical_soma/retarget_motions.py
in the separate PyRoki environment.
The MJCF is authoritative; its kinematic URDF is constructed only in memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy
from scipy.ndimage import gaussian_filter1d

from analytic_kinematics import make_forward_kinematics
from banded_trajectory_solver import solve as solve_structured

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.scripts.from_soma_to_anatomical_soma.soma23_to_anatomical_soma_retargeting import (  # noqa: E402
    SOMA_BONE_PAIRS,
    SOMA_ORIENTATION_WEIGHTS,
    SOMA_POSITION_WEIGHTS,
)


def frame_masks(frames, valid_frames):
    """Mask real frames and only temporal factors wholly inside the clip."""
    valid = jnp.arange(frames) < valid_frames
    return valid, valid[:-1] & valid[1:], valid[:-2] & valid[1:-1] & valid[2:]


def pad_trajectory(positions, rotations, contacts, seed, bucket_size):
    """Repeat finite endpoint data without truncation; costs mask the padding."""
    frames = len(seed)
    if frames < 1:
        raise ValueError("Expected at least one trajectory frame")
    if any(len(value) != frames for value in (positions, rotations, contacts)):
        raise ValueError("Trajectory arrays must have matching frame counts")
    padded_frames = (
        ((frames + bucket_size - 1) // bucket_size) * bucket_size
        if bucket_size
        else frames
    )
    return tuple(
        np.concatenate((value, np.repeat(value[-1:], padded_frames - frames, axis=0)))
        for value in (positions, rotations, contacts, seed)
    )


def build_robot(path):
    tree = ET.parse(path)
    bodies = list(tree.find("worldbody").iter("body"))
    names = [b.attrib["name"] for b in bodies]
    parent = {
        child.attrib["name"]: b.attrib["name"]
        for b in bodies
        for child in b.findall("body")
    }
    if tree.find("compiler").get("angle", "degree") != "degree":
        raise ValueError("Expected the anatomic SOMA degree-based MJCF")
    joints = []
    for body in bodies[1:]:
        js = body.findall("joint")
        if len(js) != 1 or js[0].attrib["type"] != "hinge":
            raise ValueError("Expected exactly one hinge per non-root body")
        joint = js[0]
        if "quat" in body.attrib or not np.allclose(
            np.fromstring(joint.attrib["pos"], sep=" "), 0
        ):
            raise ValueError(
                "This adapter requires identity rest rotations and zero joint pivots"
            )
        origin = np.eye(4)
        origin[:3, 3] = np.fromstring(body.attrib["pos"], sep=" ")
        lo, hi = np.deg2rad(np.fromstring(joint.attrib["range"], sep=" "))
        joints.append(
            yourdfpy.Joint(
                name=joint.attrib["name"],
                type="revolute",
                parent=parent[body.attrib["name"]],
                child=body.attrib["name"],
                origin=origin,
                axis=np.fromstring(joint.attrib["axis"], sep=" "),
                limit=yourdfpy.Limit(lower=lo, upper=hi, effort=300, velocity=100),
            )
        )
    urdf = yourdfpy.URDF(
        yourdfpy.Robot(
            "anatomic_soma",
            links=[yourdfpy.Link(name) for name in names],
            joints=joints,
        ),
        load_meshes=False,
    )
    return pk.Robot.from_urdf(urdf)


class TrajectorySolver:
    """Reuse the robot and compiled solve across clips; keep original root/timing.

    Analytic FK accelerates derivatives without changing the objective. The
    solver groups two frames to retain three-frame acceleration factors with an
    exact block-tridiagonal preconditioner.
    """

    def __init__(
        self,
        robot,
        names,
        dof_names,
        iterations=500,
        analytic=True,
        max_joint_velocity=20.0,
        max_joint_acceleration=300.0,
        motion_limit_weight=100.0,
        frame_bucket_size=0,
    ):
        if not isinstance(frame_bucket_size, int) or frame_bucket_size < 0:
            raise ValueError("Frame bucket size must be a nonnegative integer")
        self.frame_bucket_size = frame_bucket_size
        if any(
            not np.isfinite(v) or v <= 0
            for v in (max_joint_velocity, max_joint_acceleration, motion_limit_weight)
        ):
            raise ValueError(
                "Motion budgets and their penalty weight must be positive and finite"
            )
        self.max_joint_velocity = max_joint_velocity
        self.max_joint_acceleration = max_joint_acceleration
        if set(names) != set(SOMA_POSITION_WEIGHTS) or names[0] != "Hips":
            raise ValueError(
                "Expected the explicit SOMA23 semantic goal contract, pelvis first"
            )
        self.order = np.array(
            [dof_names.index(name) for name in robot.joints.actuated_names]
        )
        self.inverse = np.argsort(self.order)
        self.lower = np.asarray(robot.joints.lower_limits)[self.inverse]
        self.upper = np.asarray(robot.joints.upper_limits)[self.inverse]
        links = [robot.links.names.index(name) for name in names]
        fk = (
            make_forward_kinematics(robot, links)
            if analytic
            else lambda q: robot.forward_kinematics(q)[jnp.array(links)]
        )
        midpoint = (robot.joints.upper_limits + robot.joints.lower_limits) / 2
        half_range = (robot.joints.upper_limits - robot.joints.lower_limits) / 2
        position_weights = jnp.array([SOMA_POSITION_WEIGHTS[n] for n in names])
        orientation_weights = jnp.array([SOMA_ORIENTATION_WEIGHTS[n] for n in names])
        bone_start = jnp.array([names.index(a) for a, _ in SOMA_BONE_PAIRS])
        bone_end = jnp.array([names.index(b) for _, b in SOMA_BONE_PAIRS])

        def physical(value):
            return midpoint + half_range * jnp.tanh(value)

        def tracking(vals, var, p, r, contact, valid):
            pose = jaxlie.SE3(fk(physical(vals[var])))
            delta_p = (pose.translation() - p) * (
                50.0 * position_weights * (1 + 4.0 * contact)
            )[:, None]
            delta_r = (jaxlie.SO3(r).inverse() @ pose.rotation()).log() * (
                4.0 * orientation_weights[:, None]
            )
            target_bones = p[bone_end] - p[bone_start]
            actual_bones = pose.translation()[bone_end] - pose.translation()[bone_start]
            target_length = jnp.sqrt(
                jnp.maximum(jnp.sum(target_bones**2, axis=-1, keepdims=True), 1e-12)
            )
            actual_length = jnp.sqrt(
                jnp.maximum(jnp.sum(actual_bones**2, axis=-1, keepdims=True), 1e-12)
            )
            # Match direction without forcing changed anatomical bone lengths.
            direction = (
                actual_bones / actual_length - target_bones / target_length
            ) * 1.0
            return (
                jnp.concatenate((delta_p.ravel(), delta_r.ravel(), direction.ravel()))
                * valid
            )

        def contact_motion(
            vals,
            previous,
            current,
            prev_root_r,
            root_r,
            prev_root_p,
            root_p,
            contact,
            valid,
        ):
            prev_p = (
                jaxlie.SO3(prev_root_r).apply(
                    jaxlie.SE3(fk(physical(vals[previous]))).translation()
                )
                + prev_root_p
            )
            p = (
                jaxlie.SO3(root_r).apply(
                    jaxlie.SE3(fk(physical(vals[current]))).translation()
                )
                + root_p
            )
            # G1's contact-displacement objective, applied only to persistent
            # source contacts. This also preserves hand contacts in crawling.
            return ((p - prev_p) * contact[:, None] * 30.0).ravel() * valid

        def acceleration(vals, previous, current, following, weight, valid):
            return (
                (
                    physical(vals[following])
                    - 2 * physical(vals[current])
                    + physical(vals[previous])
                )
                * weight
                * valid
            )

        def velocity(vals, previous, current, weight, valid):
            return (physical(vals[current]) - physical(vals[previous])) * weight * valid

        def velocity_limit(vals, previous, current, dt, valid):
            speed = (physical(vals[current]) - physical(vals[previous])) / dt
            # A 10% interior penalty margin keeps the finite soft penalty away
            # from the hard output-acceptance boundary. No angle wrapping.
            return (
                jnp.maximum(jnp.abs(speed) / max_joint_velocity - 0.9, 0)
                * motion_limit_weight
                * valid
            )

        def acceleration_limit(vals, previous, current, following, dt, valid):
            accel = (
                physical(vals[following])
                - 2 * physical(vals[current])
                + physical(vals[previous])
            ) / dt**2
            return (
                jnp.maximum(jnp.abs(accel) / max_joint_acceleration - 0.9, 0)
                * motion_limit_weight
                * valid
            )

        @jax.jit
        def optimize(positions, rotations, contact, seed, dt, valid_frames):
            frames = len(seed)
            valid, valid_pairs, valid_triples = frame_masks(frames, valid_frames)
            root_r = jaxlie.SO3(rotations[:, 0][:, [3, 0, 1, 2]])
            target_p = (
                root_r.inverse()
                .apply(positions.swapaxes(0, 1) - positions[:, 0])
                .swapaxes(0, 1)
            )
            target_r = (
                root_r.inverse()
                @ jaxlie.SO3(rotations[:, :, [3, 0, 1, 2]].swapaxes(0, 1))
            ).wxyz.swapaxes(0, 1)
            # Bound physical joints throughout the solve; padding stays inactive.
            initial = jnp.arctanh(
                jnp.clip((seed - midpoint) / half_range, -0.999, 0.999)
            )
            initial = jnp.where(valid[:, None], initial, 0.0)

            def frame(q, p, r, c, v):
                return tracking(q[None], 0, p, r, c, v)

            def pair(a, b, ra, rb, pa, pb, c, v):
                values = jnp.stack((a, b))
                return jnp.concatenate(
                    (
                        contact_motion(values, 0, 1, ra, rb, pa, pb, c, v),
                        velocity(values, 0, 1, 0.01 / dt, v),
                        velocity_limit(values, 0, 1, dt, v),
                    )
                )

            def triple(a, b, c, v):
                values = jnp.stack((a, b, c))
                return jnp.concatenate(
                    (
                        acceleration(values, 0, 1, 2, 0.002 / dt**2, v),
                        acceleration_limit(values, 0, 1, 2, dt, v),
                    )
                )

            def factors(q):
                groups = [(frame, (q, target_p, target_r, contact, valid), 1)]
                if frames > 1:
                    groups.append(
                        (
                            pair,
                            (
                                q[:-1],
                                q[1:],
                                root_r.wxyz[:-1],
                                root_r.wxyz[1:],
                                positions[:-1, 0],
                                positions[1:, 0],
                                contact[:-1] * contact[1:],
                                valid_pairs,
                            ),
                            2,
                        )
                    )
                if frames > 2:
                    groups.append((triple, (q[:-2], q[1:-1], q[2:], valid_triples), 3))
                return groups

            def residuals(q):
                return tuple(jax.vmap(fn)(*args) for fn, args, _ in factors(q))

            def linearize(q):
                return tuple(
                    (
                        jax.vmap(fn)(*args),
                        jax.vmap(jax.jacfwd(fn, argnums=tuple(range(arity))))(*args),
                    )
                    for fn, args, arity in factors(q)
                )

            result, summary = solve_structured(
                initial, residuals, linearize, valid_frames, iterations
            )
            return physical(result), summary

        self.optimize = optimize

    def __call__(
        self,
        positions,
        rotations,
        contacts,
        seed,
        dt,
        allow_motion_budget_exceedance=False,
    ):
        frames = len(seed)
        seed_speed = np.abs(np.diff(seed, axis=0) / dt).max() if frames > 1 else 0.0
        seed_accel = (
            np.abs(np.diff(seed, n=2, axis=0) / dt**2).max() if frames > 2 else 0.0
        )
        smooth_seed = (
            seed_speed > self.max_joint_velocity
            or seed_accel > self.max_joint_acceleration
        )
        if smooth_seed:
            # Only initialization changes: original poses remain the objective.
            # Move sharp bounded Euler seeds away from saturated tanh branches.
            seed = gaussian_filter1d(seed, sigma=2.0, axis=0, mode="nearest")
        positions, rotations, contacts, seed = pad_trajectory(
            positions, rotations, contacts, seed, self.frame_bucket_size
        )
        start = time.perf_counter()
        with jax.default_matmul_precision("float32"):
            result, summary = self.optimize(
                *(
                    jnp.asarray(value, dtype=jnp.float32)
                    for value in (
                        positions,
                        rotations,
                        contacts,
                        seed[:, self.order],
                        dt,
                    )
                ),
                jnp.asarray(frames, dtype=jnp.int32),
            )
            q = np.asarray(result)[:frames, self.inverse]
        self.diagnostics = {
            "initial_seed_smoothed": bool(smooth_seed),
            "solve_seconds": time.perf_counter() - start,
            "frames": frames,
            "solver_frames": len(seed),
            "iterations": int(summary.iterations),
            "termination_criteria": np.asarray(summary.termination_criteria).tolist(),
            "termination_deltas": np.asarray(summary.termination_deltas).tolist(),
            "cost_history": np.asarray(summary.cost_history)[
                : int(summary.iterations) + 1
            ].tolist(),
            "joint_velocity_max_rad_s": float(np.abs(np.diff(q, axis=0) / dt).max())
            if len(q) > 1
            else 0.0,
            "joint_acceleration_max_rad_s2": float(
                np.abs(np.diff(q, n=2, axis=0) / dt**2).max()
            )
            if len(q) > 2
            else 0.0,
            "max_joint_velocity_rad_s": self.max_joint_velocity,
            "max_joint_acceleration_rad_s2": self.max_joint_acceleration,
        }
        if (
            not np.isfinite(q).all()
            or np.any(q < self.lower - 1e-5)
            or np.any(q > self.upper + 1e-5)
        ):
            raise ValueError(
                "Solver returned non-finite/out-of-limit coordinates; no post-solve clipping is performed"
            )
        if not np.any(summary.termination_criteria):
            raise ValueError(f"Solver did not converge: {self.diagnostics}")
        exceeded = self.diagnostics[
            "joint_velocity_max_rad_s"
        ] > self.max_joint_velocity * (1 + 1e-5) or self.diagnostics[
            "joint_acceleration_max_rad_s2"
        ] > self.max_joint_acceleration * (1 + 1e-5)
        self.diagnostics["motion_budget_exceeded"] = exceeded
        self.diagnostics["motion_budget_warning_only"] = allow_motion_budget_exceedance
        if exceeded and not allow_motion_budget_exceedance:
            raise ValueError(
                f"Joint motion budget exceeded; output rejected without clipping: {self.diagnostics}"
            )
        return q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--max-joint-velocity",
        type=float,
        default=20.0,
        help="Retargeting budget in rad/s",
    )
    parser.add_argument(
        "--max-joint-acceleration",
        type=float,
        default=300.0,
        help="Retargeting budget in rad/s^2",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Read input/output NPZ requests as JSON lines",
    )
    parser.add_argument(
        "--allow-motion-budget-exceedance",
        action="store_true",
        help="Export converged recovery fits with rate-budget warnings; retain joint bounds",
    )
    parser.add_argument("--cache-dir", help="Persistent JAX compilation cache")
    parser.add_argument(
        "--frame-bucket-size",
        type=int,
        default=0,
        help="Experimental masked frame padding; 0 keeps exact clip lengths",
    )
    args = parser.parse_args()
    if not args.serve and (not args.input or not args.output):
        parser.error("Pass --input and --output, or --serve")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.frame_bucket_size < 0:
        parser.error("--frame-bucket-size must be nonnegative")
    if args.cache_dir:
        jax.config.update("jax_compilation_cache_dir", args.cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    robot = build_robot(args.asset)
    solver, contract = None, None
    compiled_shapes = set()
    requests = (
        (json.loads(line) for line in sys.stdin)
        if args.serve
        else [{"input": args.input, "output": args.output}]
    )
    for request in requests:
        with np.load(request["input"], allow_pickle=False) as data:
            names, dofs = data["body_names"].tolist(), data["dof_names"].tolist()
            if solver is None:
                solver = TrajectorySolver(
                    robot,
                    names,
                    dofs,
                    args.iterations,
                    max_joint_velocity=args.max_joint_velocity,
                    max_joint_acceleration=args.max_joint_acceleration,
                    frame_bucket_size=args.frame_bucket_size,
                )
                contract = (names, dofs)
            elif (names, dofs) != contract:
                raise ValueError("Body/DOF contract changed within a worker")
            try:
                frames = len(data["seed"])
                shape = (
                    (frames + args.frame_bucket_size - 1) // args.frame_bucket_size
                    if args.frame_bucket_size
                    else frames
                )
                if shape not in compiled_shapes and len(compiled_shapes) >= 16:
                    jax.clear_caches()
                    compiled_shapes.clear()
                compiled_shapes.add(shape)
                q = solver(
                    data["positions"],
                    data["rotations"],
                    data["contacts"],
                    data["seed"],
                    float(data["dt"]),
                    allow_motion_budget_exceedance=args.allow_motion_budget_exceedance,
                )
            except ValueError as exc:
                if not args.serve:
                    raise
                print(json.dumps({"error": str(exc)}), flush=True)
                continue
        np.savez(
            request["output"],
            dof_pos=q,
            diagnostics=json.dumps(solver.diagnostics, allow_nan=False),
        )
        if args.serve:
            print(json.dumps({"output": request["output"]}), flush=True)
