# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Standalone MuJoCo inference test for tracker ONNX policies.

This script is the **deployment contract** — it demonstrates exactly how to
drive a ProtoMotions whole-body tracker policy using only raw MuJoCo state
and an ONNX runtime, with near-zero dependency on the ProtoMotions training
framework.  It is intended as a reference implementation for anyone
integrating the tracker into a different deployment framework.

Pipeline overview
-----------------
Each control step (50 Hz by default):

1. **Read robot state** from MuJoCo (``qpos``, ``qvel``, ``xquat``, ``cvel``).
2. **Derive** ``anchor_rot`` (torso IMU) and ``root_local_ang_vel`` (pelvis)
   via helpers in ``deployment.state_utils``.
3. **Query future motion** frames from ``MotionPlayer`` (25 steps ahead).
4. **Run ONNX inference** → PD position targets.
5. **Apply action post-processing** (acceleration clamp + EMA filter — both
   match the ProtoMotions MuJoCo simulator).
6. **Step MuJoCo** for *decimation* substeps.

Important conventions
---------------------
- **Quaternions**: MuJoCo uses wxyz; ProtoMotions uses xyzw.  Convert at the
  read boundary with ``mujoco_wxyz_to_xyzw()``.
- **Body indexing**: ``data.xquat[body_id + 1]`` — MuJoCo's world body is at
  index 0, so all body indices are offset by 1.
- **Angular velocity**: ``data.cvel[body_id + 1, 0:3]`` — MuJoCo's ``cvel``
  stores ``[ang_vel(3), lin_vel(3)]``.
- **Root vs anchor body**: ``root_local_ang_vel`` uses the **pelvis** (body 0).
  ``anchor_rot`` uses **torso_link** (body 16 on G1).  These are different
  bodies — mixing them up silently produces wrong observations.

ONNX inputs (for this tracker config)
--------------------------------------
======================================  ================  ==========================================
Name                                    Shape             Source
======================================  ================  ==========================================
``current_state_dof_pos``               ``[1, 29]``       ``data.qpos[7:]`` (skip free joint)
``current_state_dof_vel``               ``[1, 29]``       ``data.qvel[6:]`` (skip free joint)
``current_state_anchor_rot``            ``[1, 4]``        anchor body quat (from YAML metadata)
``current_state_root_local_ang_vel``    ``[1, 3]``        ``quat_rot_inv(pelvis_rot, pelvis_ω)``
``mimic_future_rot``                    ``[1, 25, 33, 4]``  ``MotionPlayer.get_future_references()``
``mimic_future_dof_pos``                ``[1, 25, 29]``   ``MotionPlayer.get_future_references()``
``mimic_future_dof_vel``                ``[1, 25, 29]``   ``MotionPlayer.get_future_references()``
======================================  ================  ==========================================

ONNX outputs
-------------
- ``actions``            — raw model output (tanh-bounded, before PD transform)
- ``joint_pos_targets``  — PD position targets (offset + scale already applied)
- ``stiffness_targets``  — per-DOF stiffness (constant, baked into ONNX)
- ``damping_targets``    — per-DOF damping (constant, baked into ONNX)

Action post-processing (NOT baked into ONNX)
---------------------------------------------
- **PD acceleration clamp** (``pd_target_max_accel``): limits second derivative
  of PD targets.  Matches ``base_simulator._apply_accel_clamp()``.
- **EMA action filter** (``action_ema_alpha``): exponential moving average on
  PD targets.  Matches ``MujocoSimulator._action_filter_alpha``.

Motion realignment
------------------
During training, ``realign_motion_with_humanoid_on_each_step`` snaps the
reference motion's XY to the robot's XY each step.  This only affects body
*positions* (``mimic.future_pos``).  The actor obs
(``build_deploy_target_poses``) uses only body *rotations* + DOF
positions/velocities — all position-invariant.  Therefore realignment is
**not needed** for this config.  If a future config uses position-dependent
obs (e.g. enriched target poses with ``xy_offset``), realignment must be
added here.

MuJoCo model setup
-------------------
The ``load_mujoco_model()`` function replicates the ProtoMotions MuJoCo
simulator's physics configuration:

- Patches the MJCF XML (strips sensors, adds ground plane + light).
- Sets ``model.opt.timestep`` to ``physics_dt`` from the YAML metadata.
- Zeros passive ``jnt_stiffness``, ``dof_damping``, and ``dof_frictionloss``.
- Configures actuators as implicit PD controllers (``biastype=affine``).

Dependencies
------------
Always required:
    mujoco, onnxruntime, numpy, pyyaml, torch (``torch.load`` only)

First run only (interpolation mode — if loading a raw ``.motion`` file):
    protomotions (motion interpolation utils: SLERP + lerp)

After ``--cache-motion``, subsequent runs need only the "always required"
packages — zero protomotions imports execute.

Usage
-----
::

    # First run: load raw .motion, cache at 50fps, run policy
    python deployment/test_tracker_mujoco.py \\
        --onnx  path/to/unified_pipeline.onnx \\
        --motion data/motions/walk.motion \\
        --cache-motion --render

    # Subsequent runs: cached motion, no protomotions import
    python deployment/test_tracker_mujoco.py \\
        --onnx  path/to/unified_pipeline.onnx \\
        --motion data/motions/walk.50fps.pt \\
        --render --loops 3

    # Headless benchmark (no viewer, no real-time pacing)
    python deployment/test_tracker_mujoco.py \\
        --onnx  path/to/unified_pipeline.onnx \\
        --motion data/motions/walk.50fps.pt \\
        --no-realtime
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import os
import sys

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

# Ensure the repo root is on sys.path so `deployment.*` imports work
# regardless of where the script is invoked from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deployment.state_utils import (
    mujoco_wxyz_to_xyzw,
    compute_anchor_rot_np,
    compute_yaw_offset_np,
    apply_heading_offset_np,
)
from deployment.motion_utils import MotionPlayer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ---------------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------------


def _resolve_mjcf_path(mjcf_path: str) -> Path:
    """Resolve a (possibly relative) MJCF path to an absolute filesystem path."""
    p = Path(mjcf_path)
    if p.is_absolute() and p.exists():
        return p
    repo_root = Path(__file__).parent.parent
    candidates = [
        repo_root / mjcf_path,
        repo_root / "protomotions" / "data" / "assets" / mjcf_path,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Cannot find MJCF '{mjcf_path}'.  Tried: {[str(c) for c in candidates]}"
    )


def _patch_mjcf_xml(xml_path: Path) -> str:
    """Patch a raw MJCF file for standalone MuJoCo use.

    Mirrors the logic in ``MujocoSimulator._load_mjcf_stripped``:
    1. Strip ``<sensor>`` elements (may reference missing sites).
    2. Add ground plane + light to ``<worldbody>`` if absent.

    Returns the patched XML as a string.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    for sensor_elem in root.findall("sensor"):
        root.remove(sensor_elem)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        has_ground = any(
            "floor" in g.get("name", "").lower()
            or "ground" in g.get("name", "").lower()
            or g.get("type", "").lower() == "plane"
            for g in worldbody.findall("geom")
        )
        if not has_ground:
            ground = ET.SubElement(worldbody, "geom")
            ground.set("name", "floor")
            ground.set("type", "plane")
            ground.set("size", "0 0 0.05")
            ground.set("rgba", "0.7 0.7 0.7 1")

        if not worldbody.findall("light"):
            light = ET.SubElement(worldbody, "light")
            light.set("pos", "2 0 5.0")
            light.set("dir", "0 0 -1")
            light.set("diffuse", "0.4 0.4 0.4")
            light.set("specular", "0.1 0.1 0.1")
            light.set("directional", "true")

    return ET.tostring(root, encoding="unicode")


def load_mujoco_model(
    mjcf_path: str,
    stiffness: list,
    damping: list,
    physics_dt: float,
):
    """Load MuJoCo model and configure physics to match ProtoMotions training.

    Replicates the setup from ``MujocoSimulator._create_simulation``:

    1. Patch MJCF XML (strip sensors, add ground/light).
    2. Set ``model.opt.timestep`` to *physics_dt* (critical -- MuJoCo default
       may differ from the training config).
    3. Zero passive stiffness/damping (``_zero_passive_forces``).
    4. Zero frictionloss (IsaacGym/Newton don't model Coulomb friction,
       policies weren't trained with it).
    5. Configure implicit PD actuators with force limits
       (``_configure_actuators_for_pd``).

    Parameters
    ----------
    mjcf_path:
        Path to the MJCF XML file (from YAML ``robot.mjcf_path``).
    stiffness, damping:
        Per-DOF gains in the joint order defined by the MJCF.
    physics_dt:
        Physics substep duration in seconds (from YAML ``timing.physics_dt``).
        MUST match training config -- MuJoCo default (0.002) usually differs.

    Returns
    -------
    (model, data)
    """
    import tempfile

    mjcf_file = _resolve_mjcf_path(mjcf_path)
    log.info(f"Loading MuJoCo model: {mjcf_file}")

    patched_xml = _patch_mjcf_xml(mjcf_file)

    asset_dir = str(mjcf_file.parent)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=asset_dir, delete=False
    ) as tmp:
        tmp.write(patched_xml)
        tmp_path = tmp.name

    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)

    data = mujoco.MjData(model)

    # ---- Set physics timestep (CRITICAL) ----
    # MuJoCo default is 0.002s; training may use 0.005s or other values.
    # With decimation, control_dt = physics_dt * decimation.
    # Getting this wrong makes the physics run at the wrong rate.
    old_dt = model.opt.timestep
    model.opt.timestep = physics_dt
    log.info(f"  Physics timestep: {old_dt} -> {physics_dt}s ({1.0/physics_dt:.0f}Hz)")

    # ---- Zero passive forces ----
    # Matches MujocoSimulator._zero_passive_forces().
    # We handle PD control via actuators, so passive forces would double-count.
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0

    # Zero frictionloss — the ProtoMotions MuJoCo simulator zeros it
    # per-joint in _override_joint_properties(). Other simulators
    # (IsaacGym, Newton) don't model Coulomb joint friction at all.
    model.dof_frictionloss[:] = 0.0
    log.info("  Zeroed passive stiffness, damping, and frictionloss")

    # ---- Configure implicit PD actuators ----
    # Matches MujocoSimulator._configure_actuators_for_pd().
    num_actuators = model.nu
    assert num_actuators == len(stiffness) == len(damping), (
        f"Actuator count mismatch: model.nu={num_actuators}, "
        f"len(stiffness)={len(stiffness)}, len(damping)={len(damping)}"
    )
    for i in range(num_actuators):
        kp = stiffness[i]
        kd = damping[i]
        # gaintype=0 (fixed), gainprm[0]=kp
        model.actuator_gainprm[i, 0] = kp
        # biastype=1 (affine), biasprm=[0, -kp, -kd]
        model.actuator_biastype[i] = 1
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_biasprm[i, 2] = -kd
        # Don't limit ctrl range (it's a position target, not a force)
        model.actuator_ctrllimited[i] = 0
        # NOTE: force limiting requires per-joint effort_limit values.
        # The ProtoMotions simulator sets actuator_forcerange from robot config.
        # For now we leave forcelimited at MuJoCo default (0 = disabled).
        # TODO: add effort_limit to YAML metadata and set forcerange here.

    log.info(f"  {num_actuators} actuators configured with implicit PD")
    log.info(f"  {model.nbody} bodies, {model.nq} qpos, {model.nv} qvel")
    return model, data


def read_robot_state(data, anchor_body_index: int, root_body_index: int = 0):
    """Read raw robot state from MuJoCo data buffers.

    Returns a dict with NumPy arrays (no batch dimension).

    Convention notes
    ----------------
    - ``data.xquat`` is wxyz; we convert to xyzw via ``mujoco_wxyz_to_xyzw``.
    - ``data.xquat[i + 1]`` for body i (world body occupies index 0).
    - ``data.qpos[3:7]`` is the free-joint quaternion (wxyz) — canonical for
      the root body (pelvis).  For non-root bodies, ``data.xquat`` provides
      FK-computed orientations.
    - ``data.qvel[3:6]`` is the free-joint angular velocity — already in
      **body-local frame** (no rotation needed).
    - ``data.qpos[7:]`` skips the 7-dof free joint (3 pos + 4 quat).
    - ``data.qvel[6:]`` skips the 6-dof free joint (3 vel + 3 ang_vel).
    """
    # Collect body quaternions (wxyz -> xyzw), skip world body at index 0.
    # xquat provides FK-computed orientations for all bodies.
    body_rot_wxyz = data.xquat[1:].copy()          # [num_bodies, 4] wxyz
    body_rot = mujoco_wxyz_to_xyzw(body_rot_wxyz)  # [num_bodies, 4] xyzw

    # For the root body, prefer the canonical free-joint quaternion from qpos
    # (matches robojudo's base_quat path and avoids any FK rounding).
    root_rot_wxyz = data.qpos[3:7].copy()
    body_rot[root_body_index] = mujoco_wxyz_to_xyzw(root_rot_wxyz)

    # Root angular velocity: qvel[3:6] is already in body-local frame.
    # This avoids the world->local rotation that cvel would require.
    root_local_ang_vel = data.qvel[3:6].copy().astype(np.float32)  # [3]

    return {
        "dof_pos":            data.qpos[7:].copy().astype(np.float32),  # [num_dofs]
        "dof_vel":            data.qvel[6:].copy().astype(np.float32),  # [num_dofs]
        "body_rot":           body_rot.astype(np.float32),               # [nb, 4]
        "root_local_ang_vel": root_local_ang_vel,                        # [3]
    }


def set_initial_pose(model, data, motion_player: MotionPlayer) -> None:
    """Initialise the robot at the first frame of the motion."""
    frame0 = motion_player.get_state_at_frame(0)

    # Root position/orientation from first body (pelvis)
    root_pos  = frame0["body_pos"][0]       # [3]
    root_quat = frame0["body_rot"][0]       # [4] xyzw

    # Set free-joint state
    data.qpos[0:3] = root_pos
    # MuJoCo qpos uses wxyz quaternion
    data.qpos[3:7] = root_quat[[3, 0, 1, 2]]  # xyzw -> wxyz
    data.qpos[7:]  = frame0["dof_pos"]

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    log.info(
        f"  Initial pose set: root_pos={root_pos.round(3).tolist()}, "
        f"dof_pos[:5]={frame0['dof_pos'][:5].round(3).tolist()}"
    )


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------


def build_onnx_inputs(
    robot_state: dict,
    future_refs: dict,
    onnx_name_to_key: dict,
    anchor_body_index: int,
    num_dofs: int,
    prev_actions: np.ndarray | None = None,
) -> dict:
    """Assemble ONNX input dict from raw MuJoCo state + motion futures.

    Maps semantic context keys to NumPy arrays with the batch dimension
    (size 1) added.
    """
    dof_pos = robot_state["dof_pos"]                      # [num_dofs]
    dof_vel = robot_state["dof_vel"]                      # [num_dofs]
    body_rot = robot_state["body_rot"]                    # [nb, 4]
    root_local_ang_vel = robot_state["root_local_ang_vel"]  # [3]

    # Anchor rotation: works for any anchor body (pelvis, torso, etc.)
    anchor_rot = compute_anchor_rot_np(body_rot, anchor_body_index)  # [4]

    # Historical actions (previous step's raw actions); zero on first step.
    if prev_actions is None:
        prev_actions = np.zeros(num_dofs, dtype=np.float32)

    # Future anchor rotation: extract anchor body from full future body rots.
    future_anchor_rot = future_refs["body_rot"][:, anchor_body_index, :]  # [nsteps, 4]

    # Build a lookup from semantic key -> array[with batch dim]
    key_to_array = {
        "current.dof_pos":             dof_pos[None],                # [1, ndofs]
        "current.dof_vel":             dof_vel[None],                # [1, ndofs]
        "current.anchor_rot":          anchor_rot[None],             # [1, 4]
        "current.root_local_ang_vel":  root_local_ang_vel[None],     # [1, 3]
        "historical.processed_actions": prev_actions[None, None],    # [1, 1, ndofs]
        # Future references: [n_steps, ndofs/nbodies/4] -> [1, n_steps, ...]
        "mimic.future_anchor_rot": future_anchor_rot[None],          # [1, nsteps, 4]
        "mimic.future_rot":     future_refs["body_rot"][None],       # [1, nsteps, nb, 4]
        "mimic.future_dof_pos": future_refs["dof_pos"][None],        # [1, nsteps, ndofs]
        "mimic.future_dof_vel": future_refs["dof_vel"][None],        # [1, nsteps, ndofs]
    }

    onnx_inputs: dict[str, np.ndarray] = {}
    for onnx_name, sem_key in onnx_name_to_key.items():
        if sem_key in key_to_array:
            onnx_inputs[onnx_name] = key_to_array[sem_key].astype(np.float32)
        else:
            log.warning(f"No value for ONNX input '{onnx_name}' (key='{sem_key}')")

    return onnx_inputs


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(
    onnx_path: str,
    motion_file: str,
    cache_motion: bool = False,
    num_loops: int = 1,
    render: bool = False,
    realtime: bool = True,
    action_ema_alpha: float | None = None,
) -> None:
    """Run the tracker policy in a MuJoCo simulation loop.

    Parameters
    ----------
    onnx_path:
        Path to the exported ``unified_pipeline.onnx``.
    motion_file:
        Path to a ProtoMotions ``.pt`` motion file (raw or cached).
    cache_motion:
        If True and *motion_file* is a raw ProtoMotions file, write a
        pre-resampled cache next to the input file after loading.
    num_loops:
        How many times to loop the motion.
    render:
        If True, open a MuJoCo viewer window.
    realtime:
        If True (default), pace the simulation to wall-clock real time.
        When False, runs as fast as possible.
    action_ema_alpha:
        Exponential moving average filter on PD targets.  Matches
        ``MujocoSimulator._action_filter_alpha``.
        ``a_applied = alpha * a_policy + (1 - alpha) * a_prev``.
        Set to 1.0 to disable filtering.  If None, loads from the YAML
        metadata (``control.action_ema_alpha``).
    """
    onnx_path  = str(onnx_path)
    yaml_path  = onnx_path.replace(".onnx", ".yaml")

    # ------------------------------------------------------------------
    # Load YAML metadata
    # ------------------------------------------------------------------
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    robot_meta  = meta["robot"]
    timing      = meta["timing"]
    motion_meta = meta["motion"]
    control     = meta["control"]
    runtime     = meta["_runtime"]

    anchor_body_index = robot_meta["anchor_body_index"]
    root_body_index   = robot_meta["root_body_index"]
    num_bodies        = robot_meta["num_bodies"]
    num_dofs          = robot_meta["num_dofs"]
    mjcf_path         = robot_meta["mjcf_path"]
    control_dt        = timing["control_dt"]
    decimation        = timing["decimation"]
    future_step_indices = motion_meta["future_step_indices"]
    num_future_steps    = len(future_step_indices)
    stiffness           = control["stiffness"]
    damping             = control["damping"]
    pd_target_max_accel = control.get("pd_target_max_accel")
    if action_ema_alpha is None:
        action_ema_alpha = control.get("action_ema_alpha", 1.0)
    onnx_name_to_key    = runtime["onnx_name_to_in_key"]
    # onnx_out_names are read from the session directly (actual_out_names)

    anchor_body_name = robot_meta.get("anchor_body_name", f"body_{anchor_body_index}")
    root_body_name = robot_meta.get("root_body_name", f"body_{root_body_index}")

    log.info(f"ONNX: {onnx_path}")
    log.info(f"MJCF: {mjcf_path}")
    log.info(
        f"Robot: {num_dofs} DOFs, {num_bodies} bodies, "
        f"anchor={anchor_body_name}[{anchor_body_index}], "
        f"root={root_body_name}[{root_body_index}]"
    )
    log.info(f"control_dt={control_dt}s, decimation={decimation}")
    log.info(f"Future steps: {future_step_indices}")

    # ------------------------------------------------------------------
    # Load ONNX session
    # ------------------------------------------------------------------
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    actual_in_names  = [inp.name for inp in session.get_inputs()]
    actual_out_names = [out.name for out in session.get_outputs()]
    log.info(f"ONNX inputs:  {actual_in_names}")
    log.info(f"ONNX outputs: {actual_out_names}")

    # Warm-up run
    dummy = {n: np.zeros([1] + list(i.shape)[1:], dtype=np.float32)
             for n, i in zip(actual_in_names, session.get_inputs())}
    # Use shape from actual inputs (shape[0] may be None for dynamic batch)
    for inp in session.get_inputs():
        shape = [1 if (d is None or isinstance(d, str)) else d
                 for d in inp.shape]
        dummy[inp.name] = np.zeros(shape, dtype=np.float32)
    try:
        session.run(actual_out_names, dummy)
    except Exception:
        pass  # warmup failure is acceptable (dummy input may be invalid)

    # ------------------------------------------------------------------
    # Load motion
    # ------------------------------------------------------------------
    player = MotionPlayer(motion_file, control_dt=control_dt)

    if cache_motion:
        motion_p = Path(motion_file)
        cache_p  = motion_p.parent / (motion_p.stem + ".50fps.pt")
        # If the parent dir is not writable, write next to the ONNX model instead
        if not os.access(str(cache_p.parent), os.W_OK):
            onnx_dir = Path(onnx_path).parent
            cache_p  = onnx_dir / (motion_p.stem + ".50fps.pt")
        if not cache_p.exists():
            player.cache_to_file(str(cache_p))
        else:
            log.info(f"Cache already exists: {cache_p}")

    log.info(
        f"Motion: {player.total_frames} frames @ "
        f"{1.0 / control_dt:.0f} Hz  (duration={player.total_frames * control_dt:.2f}s)"
    )

    # ------------------------------------------------------------------
    # Load MuJoCo model
    # ------------------------------------------------------------------
    physics_dt = timing["physics_dt"]
    model, data = load_mujoco_model(mjcf_path, stiffness, damping, physics_dt)

    # ------------------------------------------------------------------
    # Optional viewer
    # ------------------------------------------------------------------
    viewer = None
    if render:
        try:
            from mujoco import viewer as mj_viewer

            viewer = mj_viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False,
            )
            # Match ProtoMotions MujocoSimulator camera settings
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -10.0
            viewer.cam.azimuth = 180.0
            viewer.cam.trackbodyid = 1  # pelvis (first non-world body)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            log.info("Viewer launched (tracking pelvis)")
        except Exception as e:
            log.warning(f"Could not launch viewer: {e}")
            viewer = None

    # ------------------------------------------------------------------
    # PD target acceleration clamp
    #
    # During training, ``pd_target_max_accel`` limits the second derivative
    # (acceleration) of the PD position targets.  This prevents oscillatory
    # jitter in the controller.  The policy was *trained* with this clamp
    # active, so we must replicate it here.
    #
    # Formula (from base_simulator.py _apply_accel_clamp):
    #   delta      = actions - prev_actions
    #   prev_delta = prev_actions - prev_prev_actions
    #   accel      = delta - prev_delta
    #   clamped    = prev_actions + prev_delta + clamp(accel, -max, max)
    # ------------------------------------------------------------------
    if pd_target_max_accel is not None:
        log.info(f"PD target accel clamp: {pd_target_max_accel}")

    # ------------------------------------------------------------------
    # Motion realignment note
    #
    # During training, ``realign_motion_with_humanoid_on_each_step=True``
    # snaps the reference motion's XY position to the robot's current XY
    # each step.  This only affects ``mimic.future_pos`` (body positions).
    # The actor's ``build_deploy_target_poses`` obs function uses only
    # body *rotations* + DOF positions/velocities -- all of which are
    # position-invariant.  Therefore realignment does NOT affect the ONNX
    # inputs for this config and is not implemented here.
    #
    # If a future config uses position-dependent obs (e.g., enriched
    # target poses with xy_offset), realignment must be added.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Action EMA filter
    #
    # Matches MujocoSimulator._action_filter_alpha (default 0.8):
    #   targets_applied = alpha * targets_policy + (1 - alpha) * targets_prev
    # Set alpha=1.0 to disable.
    # ------------------------------------------------------------------
    use_ema = action_ema_alpha < 1.0
    if use_ema:
        log.info(f"Action EMA filter: alpha={action_ema_alpha}")
    ema_prev_targets: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    total_steps  = 0
    total_ort_ms = 0.0
    total_sim_ms = 0.0
    max_pd_diff  = 0.0

    # Accel clamp history (3 consecutive PD target frames)
    prev_pd: np.ndarray | None = None
    prev_prev_pd: np.ndarray | None = None

    loop_idx = 0
    while loop_idx < num_loops:
        loop_label = f"{loop_idx + 1}/{num_loops}" if num_loops < 1_000_000 else f"{loop_idx + 1}"
        log.info(f"\n--- Loop {loop_label} ---")
        set_initial_pose(model, data, player)
        prev_pd = None
        prev_prev_pd = None
        ema_prev_targets = None
        prev_actions = None
        heading_offset = None
        loop_wall_start = time.perf_counter()

        for frame_idx in range(player.total_frames):
            step_wall_start = time.perf_counter()

            # ---- read robot state ----
            robot_state = read_robot_state(data, anchor_body_index, root_body_index)

            # ---- compute heading offset on first step ----
            if heading_offset is None:
                robot_anchor_rot = robot_state["body_rot"][anchor_body_index]
                motion_anchor_rot = player.get_state_at_frame(0)["body_rot"][anchor_body_index]
                heading_offset = compute_yaw_offset_np(robot_anchor_rot, motion_anchor_rot)

            # ---- get future motion references ----
            future_refs = player.get_future_references(frame_idx, future_step_indices)

            # ---- apply heading alignment to future body rotations ----
            future_refs["body_rot"] = apply_heading_offset_np(
                heading_offset, future_refs["body_rot"]
            )

            # ---- build ONNX inputs ----
            onnx_inputs = build_onnx_inputs(
                robot_state=robot_state,
                future_refs=future_refs,
                onnx_name_to_key=onnx_name_to_key,
                anchor_body_index=anchor_body_index,
                num_dofs=num_dofs,
                prev_actions=prev_actions,
            )

            # ---- ONNX inference ----
            t0 = time.perf_counter()
            ort_out = session.run(actual_out_names, onnx_inputs)
            total_ort_ms += (time.perf_counter() - t0) * 1000.0

            # Outputs: actions, joint_pos_targets, stiffness, damping
            pd_targets = ort_out[1].squeeze().copy()  # [num_dofs]

            # ---- PD target acceleration clamp ----
            if pd_target_max_accel is not None and prev_pd is not None and prev_prev_pd is not None:
                delta = pd_targets - prev_pd
                prev_delta = prev_pd - prev_prev_pd
                accel = delta - prev_delta
                clamped_accel = np.clip(accel, -pd_target_max_accel, pd_target_max_accel)
                pd_targets = prev_pd + prev_delta + clamped_accel

            # Shift accel clamp history
            prev_prev_pd = prev_pd
            prev_pd = pd_targets.copy()

            # ---- EMA action filter ----
            if use_ema:
                if ema_prev_targets is None:
                    ema_prev_targets = pd_targets.copy()
                pd_targets = action_ema_alpha * pd_targets + (1.0 - action_ema_alpha) * ema_prev_targets
                ema_prev_targets = pd_targets.copy()

            # ---- feedback for next step's historical.processed_actions ----
            prev_actions = pd_targets.copy()

            # ---- write PD targets to MuJoCo ----
            data.ctrl[:] = pd_targets

            # ---- physics substeps ----
            t0 = time.perf_counter()
            for _ in range(decimation):
                mujoco.mj_step(model, data)
            total_sim_ms += (time.perf_counter() - t0) * 1000.0

            # ---- optional: track PD error vs reference ----
            ref_dof_pos = player.get_state_at_frame(frame_idx)["dof_pos"]
            diff = float(np.abs(data.qpos[7:] - ref_dof_pos).max())
            if diff > max_pd_diff:
                max_pd_diff = diff

            # ---- viewer ----
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()

            # ---- real-time pacing ----
            if realtime:
                elapsed = time.perf_counter() - step_wall_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            total_steps += 1

            if frame_idx % 100 == 0:
                root_height = float(data.qpos[2])
                wall_elapsed = time.perf_counter() - loop_wall_start
                sim_elapsed = (frame_idx + 1) * control_dt
                speed_ratio = sim_elapsed / max(wall_elapsed, 1e-6)
                log.info(
                    f"  step={total_steps:5d}  frame={frame_idx:4d}  "
                    f"root_h={root_height:.3f}  max_ref_err={max_pd_diff:.4f}  "
                    f"speed={speed_ratio:.2f}x"
                )

        loop_idx += 1

        # If viewer was closed mid-loop, stop
        if viewer is not None and not viewer.is_running():
            break

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    avg_ort_ms = total_ort_ms / max(total_steps, 1)
    avg_sim_ms = total_sim_ms / max(total_steps, 1)
    log.info(
        f"\n=== Done: {total_steps} steps over {loop_idx} loop(s) ===\n"
        f"  avg ONNX inference : {avg_ort_ms:.2f} ms/step\n"
        f"  avg physics        : {avg_sim_ms:.2f} ms/step\n"
        f"  max joint ref error: {max_pd_diff:.4f} rad"
    )

    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass  # GLX teardown may segfault on some drivers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run tracker ONNX policy in MuJoCo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--onnx",
        required=True,
        help="Path to unified_pipeline.onnx",
    )
    p.add_argument(
        "--motion",
        required=True,
        help="Path to motion .pt file (raw ProtoMotions or pre-cached)",
    )
    p.add_argument(
        "--cache-motion",
        action="store_true",
        default=False,
        help=(
            "After loading a raw motion file, write a 50fps cache next to it. "
            "The cache filename is <motion>.50fps.pt."
        ),
    )
    p.add_argument(
        "--loops",
        type=int,
        default=None,
        help="Number of times to loop the motion (default: infinite with --render, 1 otherwise)",
    )
    p.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Open a MuJoCo viewer window",
    )
    p.add_argument(
        "--no-realtime",
        action="store_true",
        default=False,
        help="Disable real-time pacing (run as fast as possible)",
    )
    p.add_argument(
        "--action-ema-alpha",
        type=float,
        default=None,
        help=(
            "EMA filter on PD targets. Overrides the value from the YAML metadata "
            "(control.action_ema_alpha). 1.0 = no filtering, lower = more smoothing."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Default: loop forever with --render, once without
    num_loops = args.loops if args.loops is not None else (10_000_000 if args.render else 1)
    run(
        onnx_path=args.onnx,
        motion_file=args.motion,
        cache_motion=args.cache_motion,
        num_loops=num_loops,
        render=args.render,
        realtime=not args.no_realtime,
        action_ema_alpha=args.action_ema_alpha,
    )
    # Force clean exit — avoids GLXBadContext segfault from MuJoCo's
    # atexit GL context teardown on some Linux drivers.
    os._exit(0)
