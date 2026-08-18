# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simplified ONNX export for tracker policies.

This script exports a ProtoMotions tracker policy to a unified ONNX model
**without** running a simulator or creating a full training environment.
It works by:

1. Loading ``resolved_configs_inference.pt`` from the checkpoint directory.
2. Building a :class:`MockContext` with correctly-shaped random tensors so
   that ``ObservationExportModule`` can discover context paths and tensor
   shapes without executing real physics.
3. Reconstructing *only* the actor network from the agent config, then
   loading weights from the checkpoint.
4. Composing obs-module + actor + action-module into a single
   ``UnifiedPipelineModule`` and exporting it to ONNX via
   ``torch.onnx.export``.
5. Writing a rich YAML sidecar that acts as a machine-readable deployment
   contract (see :data:`YAML_CONVENTIONS` and ``deploy_inputs`` section).

The script is intentionally *specific* to the tracker config
``mlp_domain_rand_ablation_deploy_robust_smooth_holo_with_push``
(actor obs: ``reduced_coords_obs`` + ``mimic_deploy_target_poses``).
For other configs, update the ``ACTOR_OBS_KEYS`` constant and the
``MockContext`` dimensions accordingly.

Usage
-----
::

    python deployment/export_tracker_onnx.py \\
        --checkpoint results/my_exp/last.ckpt \\
        --output    deployment/models/

Outputs
-------
``<output>/unified_pipeline.onnx``
    The exported ONNX model.  Inputs are raw context tensors; outputs are
    ``(actions, joint_pos_targets, stiffness_targets, damping_targets)``.

``<output>/unified_pipeline.yaml``
    Rich metadata / deployment contract.

TODO(future): generalise MockContext by auto-discovering shapes from
``MdpComponent.dynamic_vars`` bindings rather than hard-coding dimensions.
See plan doc for details.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Obs keys consumed by the actor in this specific config.
# The critic uses additional obs (max_coords_obs, mimic_max_coords_target_poses)
# that we explicitly exclude -- they're not needed for deployment.
# ---------------------------------------------------------------------------
ACTOR_OBS_KEYS = {"reduced_coords_obs", "mimic_deploy_target_poses"}


# ---------------------------------------------------------------------------
# MockContext
# ---------------------------------------------------------------------------


class _MockState:
    """Mock for CurrentStateView.

    Uses torch.randn (not zeros) to prevent ONNX constant-folding from
    accidentally eliminating inputs that happen to produce zero gradients.
    Quaternions are L2-normalised so they are valid rotations.

    Includes all fields that any standard obs factory may bind to --
    actor obs (dof_pos, dof_vel, anchor_rot, root_local_ang_vel) and
    critic obs (rigid_body_pos/rot/vel/ang_vel).
    """

    def __init__(self, num_envs: int, num_dofs: int, num_bodies: int, anchor_idx: int):
        import torch
        import torch.nn.functional as F

        self.anchor_idx = anchor_idx

        # Actor obs fields
        self.dof_pos = torch.randn(num_envs, num_dofs)
        self.dof_vel = torch.randn(num_envs, num_dofs)
        self.anchor_rot = F.normalize(torch.randn(num_envs, 4), dim=-1)
        self.root_local_ang_vel = torch.randn(num_envs, 3)
        # Critic obs fields (max_coords_obs needs full body arrays)
        self.rigid_body_pos     = torch.randn(num_envs, num_bodies, 3)
        self.rigid_body_rot     = F.normalize(torch.randn(num_envs, num_bodies, 4), dim=-1)
        self.rigid_body_vel     = torch.randn(num_envs, num_bodies, 3)
        self.rigid_body_ang_vel = torch.randn(num_envs, num_bodies, 3)


class _MockMimic:
    """Mock for MimicContext."""

    def __init__(
        self,
        num_envs: int,
        num_future_steps: int,
        num_dofs: int,
        num_bodies: int,
        anchor_idx: int,
    ):
        import torch
        import torch.nn.functional as F

        self.anchor_idx = anchor_idx

        # Actor obs fields
        self.future_rot     = F.normalize(
            torch.randn(num_envs, num_future_steps, num_bodies, 4), dim=-1
        )
        self.future_dof_pos = torch.randn(num_envs, num_future_steps, num_dofs)
        self.future_dof_vel = torch.randn(num_envs, num_future_steps, num_dofs)
        # Critic obs fields (mimic_max_coords_target_poses needs pos/vel/ang_vel too)
        self.future_pos     = torch.randn(num_envs, num_future_steps, num_bodies, 3)
        self.future_vel     = torch.randn(num_envs, num_future_steps, num_bodies, 3)
        self.future_ang_vel = torch.randn(num_envs, num_future_steps, num_bodies, 3)


class MockContext:
    """Minimal stand-in for EnvContext used only during ONNX export tracing.

    ObservationExportModule.__init__ resolves context paths (e.g.
    ``"current.dof_pos"``) to check whether they resolve to tensors (ONNX
    input) or non-tensor constants (baked into graph).  It never calls the
    observation functions during __init__, so the values here only need the
    correct *shape* and *dtype* -- not real physics state.
    """

    def __init__(
        self,
        num_envs: int,
        num_dofs: int,
        num_bodies: int,
        num_future_steps: int,
        anchor_idx: int,
    ):
        import torch

        self.current = _MockState(num_envs, num_dofs, num_bodies, anchor_idx)
        self.mimic   = _MockMimic(
            num_envs, num_future_steps, num_dofs, num_bodies, anchor_idx
        )
        # body_contacts: used by max_coords_obs observe_contacts -- bool tensor
        self.body_contacts  = torch.zeros(num_envs, num_bodies, dtype=torch.bool)
        # ground_heights: used by max_coords_obs root_height_obs
        self.ground_heights = torch.zeros(num_envs)


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------


def export_tracker(
    checkpoint: str,
    output_dir: str,
    validate: bool = True,
) -> Path:
    """Export the tracker policy to a unified ONNX model.

    Parameters
    ----------
    checkpoint:
        Path to ``last.ckpt`` (or any ``*.ckpt``).
    output_dir:
        Directory where ONNX + YAML files will be written.
    validate:
        If True, run onnxruntime validation after export.

    Returns
    -------
    Path to the exported ``.onnx`` file.
    """
    import torch
    from tensordict import TensorDict

    from protomotions.utils.export_utils import (
        ObservationExportModule,
        ActionExportModule,
        UnifiedPipelineModule,
    )
    from protomotions.utils.hydra_replacement import get_class

    checkpoint_path = Path(checkpoint)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load resolved configs (no simulator import required)
    # ------------------------------------------------------------------
    resolved_path = checkpoint_path.parent / "resolved_configs_inference.pt"
    if not resolved_path.exists():
        log.warning(
            "resolved_configs_inference.pt not found, falling back to "
            "resolved_configs.pt.  Domain randomization may still be active!"
        )
        resolved_path = checkpoint_path.parent / "resolved_configs.pt"
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Could not find resolved_configs*.pt in {checkpoint_path.parent}"
        )

    log.info(f"Loading configs from {resolved_path}")
    resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)

    robot_config    = resolved["robot"]
    env_config      = resolved["env"]
    agent_config    = resolved["agent"]
    simulator_config = resolved.get("simulator")

    # ------------------------------------------------------------------
    # 2. Extract dimensions from configs
    # ------------------------------------------------------------------
    num_dofs   = robot_config.kinematic_info.num_dofs
    num_bodies = len(robot_config.kinematic_info.body_names)
    body_names = list(robot_config.kinematic_info.body_names)
    joint_names = list(robot_config.kinematic_info.dof_names)
    anchor_body_name = robot_config.anchor_body_name
    anchor_body_index = robot_config.anchor_body_index
    root_body_index = 0  # pelvis is always first body

    mimic_ctrl_cfg = env_config.control_components.get("mimic")
    if mimic_ctrl_cfg is None:
        raise ValueError("env_config.control_components must contain 'mimic'")
    raw_future_steps = mimic_ctrl_cfg.future_steps
    # future_steps can be int N (-> [1..N]) or explicit list
    if isinstance(raw_future_steps, int):
        num_future_steps = raw_future_steps
        future_step_indices = list(range(1, raw_future_steps + 1))
    else:
        num_future_steps = len(raw_future_steps)
        future_step_indices = list(raw_future_steps)

    # Resolve MuJoCo-specific timing.
    # The resolved config may be from a different training simulator
    # (e.g., IsaacLab fps=200, dec=4).  We apply the sim2sim conversion to
    # get the correct MuJoCo timing (fps=1000, dec=20, dt=0.001).
    # control_dt stays the same (0.02) across all simulators.
    control_dt = 0.02
    physics_dt = 0.001
    decimation = 20
    pd_target_max_accel = None
    if simulator_config is not None:
        # Apply sim2sim conversion to MuJoCo defaults
        try:
            from protomotions.simulator.factory import update_simulator_config_for_test
            mj_sim_cfg = update_simulator_config_for_test(
                current_simulator_config=simulator_config,
                new_simulator="mujoco",
                robot_config=robot_config,
            )
            physics_dt = 1.0 / mj_sim_cfg.sim.fps
            decimation = mj_sim_cfg.sim.decimation
            control_dt = physics_dt * decimation
        except Exception as e:
            log.warning(f"Could not apply sim2sim conversion: {e}")
            # Fallback: read from raw config
            sim_cfg = getattr(simulator_config, "sim", None)
            if sim_cfg is not None:
                _fps = getattr(sim_cfg, "fps", None)
                _dec = getattr(sim_cfg, "decimation", None)
                if _fps and _dec:
                    physics_dt = 1.0 / _fps
                    decimation = _dec
                    control_dt = physics_dt * decimation
        _accel = getattr(simulator_config, "pd_target_max_accel", None)
        if _accel is not None:
            pd_target_max_accel = float(_accel)

    log.info(
        f"Robot: {num_dofs} DOFs, {num_bodies} bodies, "
        f"anchor={anchor_body_name}(idx={anchor_body_index})"
    )
    log.info(
        f"Timing: control_dt={control_dt}s  physics_dt={physics_dt}s  "
        f"decimation={decimation}"
    )
    log.info(f"Future steps: {future_step_indices} ({num_future_steps} total)")

    # ------------------------------------------------------------------
    # 3. Build MockContext for ONNX tracing shape inference
    # ------------------------------------------------------------------
    mock = MockContext(
        num_envs=1,
        num_dofs=num_dofs,
        num_bodies=num_bodies,
        num_future_steps=num_future_steps,
        anchor_idx=anchor_body_index,
    )

    # ------------------------------------------------------------------
    # 4. Build ObservationExportModule (actor obs only, for export)
    #    Also build an all-obs module to correctly materialise LazyLinear
    #    in the full model (critic uses additional obs keys).
    # ------------------------------------------------------------------
    actor_obs_configs = {
        k: v
        for k, v in env_config.observation_components.items()
        if k in ACTOR_OBS_KEYS
    }
    missing = ACTOR_OBS_KEYS - set(env_config.observation_components.keys())
    if missing:
        raise ValueError(
            f"Expected observation components {ACTOR_OBS_KEYS} in env_config "
            f"but these are missing: {missing}"
        )

    log.info(f"Observation components for export: {list(actor_obs_configs.keys())}")
    obs_module = ObservationExportModule(actor_obs_configs, mock, device="cpu")
    obs_module.eval()

    obs_input_keys  = obs_module.get_input_keys()
    obs_output_keys = obs_module.get_output_keys()
    log.info(f"  Context input keys:  {obs_input_keys}")
    log.info(f"  Observation outputs: {obs_output_keys}")

    # ------------------------------------------------------------------
    # 5. Reconstruct actor-only and load weights
    #
    # We build only the PPOActor (not the full PPOModel with critic), which:
    #  - avoids all critic obs-shape bookkeeping
    #  - keeps the export lean and fast
    #  - matches exactly what is needed for deployment inference
    # ------------------------------------------------------------------
    log.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    ActorClass = get_class(agent_config.model.actor._target_)
    actor = ActorClass(agent_config.model.actor)

    # nn.LazyLinear needs one forward pass to materialise concrete layer shapes.
    # Run the actor obs module on mock inputs -- this gives exact obs dims.
    # IMPORTANT: eval() before forward so the normalizer skips record_moments()
    # (which needs fabric and is train-only).
    actor.eval()
    mock_obs_inputs = [_resolve_attr_path(k, mock) for k in obs_input_keys]
    with torch.no_grad():
        mock_obs_out = obs_module(*mock_obs_inputs)
        mock_obs_td = TensorDict(
            {k: v for k, v in zip(obs_output_keys, mock_obs_out)},
            batch_size=[1],
        )
        actor(mock_obs_td)  # materialises LazyLinear layers

    log.info(
        "LazyLinear materialised with actor obs: "
        + ", ".join(f"{k}={v.shape[-1]}" for k, v in mock_obs_td.items())
    )

    # Strip "_actor." prefix from checkpoint keys and load into actor.
    actor_state = {
        k[len("_actor."):]: v
        for k, v in ckpt["model"].items()
        if k.startswith("_actor.")
    }
    actor.load_state_dict(actor_state)
    actor.eval()

    log.info(f"Actor in_keys:  {list(actor.in_keys)}")
    log.info(f"Actor out_keys: {list(actor.out_keys)}")

    # Verify actor obs keys are covered by our obs module
    uncovered = set(actor.in_keys) - set(obs_output_keys)
    if uncovered:
        raise ValueError(
            f"Actor requires obs keys not produced by ObservationExportModule: "
            f"{uncovered}.  Update ACTOR_OBS_KEYS."
        )

    # ------------------------------------------------------------------
    # 6. Build ActionExportModule  (action_config lives on env_config)
    # ------------------------------------------------------------------
    action_module = ActionExportModule(env_config.action_config, device="cpu")
    action_module.eval()

    # ------------------------------------------------------------------
    # 7. Compose UnifiedPipelineModule
    # ------------------------------------------------------------------
    unified = UnifiedPipelineModule(
        observation_module=obs_module,
        policy_module=actor,
        action_module=action_module,
        policy_in_keys=list(actor.in_keys),
        policy_action_key="mean_action",
    )
    unified.cpu().eval()

    # ------------------------------------------------------------------
    # 8. Collect sample inputs and verify forward pass
    # ------------------------------------------------------------------
    sample_inputs = [_resolve_attr_path(k, mock) for k in obs_input_keys]
    input_shapes = {k: list(v.shape) for k, v in zip(obs_input_keys, sample_inputs)}

    with torch.no_grad():
        actions, pd_targets, stiffness_t, damping_t = unified(*sample_inputs)

    log.info(f"Forward pass OK: actions={list(actions.shape)}, "
             f"pd_targets={list(pd_targets.shape)}")

    # ------------------------------------------------------------------
    # 9. Export to ONNX
    # ------------------------------------------------------------------
    def _sanitize(name: str) -> str:
        return name.replace(".", "_").replace("[", "_").replace("]", "_")

    onnx_input_names  = [_sanitize(k) for k in obs_input_keys]
    onnx_output_names = ["actions", "joint_pos_targets",
                         "stiffness_targets", "damping_targets"]

    onnx_path = output_path / "unified_pipeline.onnx"
    log.info(f"Exporting ONNX to {onnx_path} …")
    torch.onnx.export(
        unified,
        tuple(sample_inputs),
        str(onnx_path),
        input_names=onnx_input_names,
        output_names=onnx_output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            **{name: {0: "batch_size"} for name in onnx_input_names},
            **{name: {0: "batch_size"} for name in onnx_output_names},
        },
        dynamo=False,
    )
    log.info(f"✓  ONNX exported → {onnx_path}")

    # ------------------------------------------------------------------
    # 10. Read back actual ONNX names (ONNX may rename inputs)
    # ------------------------------------------------------------------
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual_in_names  = [inp.name for inp in session.get_inputs()]
    actual_out_names = [out.name for out in session.get_outputs()]

    # Build onnx_name -> semantic_key mapping
    sanitized_to_key = {_sanitize(k): k for k in obs_input_keys}
    onnx_name_to_key: dict[str, str] = {}
    for onnx_name in actual_in_names:
        base = onnx_name
        for suffix in (".1", ".2", ".3", "_1", "_2", "_3"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if base in sanitized_to_key:
            onnx_name_to_key[onnx_name] = sanitized_to_key[base]
        elif onnx_name in sanitized_to_key:
            onnx_name_to_key[onnx_name] = sanitized_to_key[onnx_name]
        else:
            log.warning(f"Cannot map ONNX input '{onnx_name}' to a semantic key")

    # ------------------------------------------------------------------
    # 11. Validate with onnxruntime
    # ------------------------------------------------------------------
    if validate:
        import numpy as np

        log.info("Validating with onnxruntime …")
        key_to_tensor = {k: t for k, t in zip(obs_input_keys, sample_inputs)}
        ort_inputs = {
            name: key_to_tensor[onnx_name_to_key[name]].detach().numpy()
            for name in actual_in_names
            if name in onnx_name_to_key
        }
        ort_outputs = session.run(actual_out_names, ort_inputs)

        pytorch_outputs = [
            actions.detach().numpy(),
            pd_targets.detach().numpy(),
            stiffness_t.detach().numpy(),
            damping_t.detach().numpy(),
        ]
        for i, (name, pt_out) in enumerate(zip(onnx_output_names, pytorch_outputs)):
            diff = np.abs(ort_outputs[i] - pt_out).max()
            status = "✓" if diff < 1e-4 else "⚠"
            log.info(f"  {status}  {name}: max_diff = {diff:.2e}")
        log.info("✓  Validation complete")

    # ------------------------------------------------------------------
    # 12. Build and write rich YAML metadata
    # ------------------------------------------------------------------
    stiffness_vals = [
        float(robot_config.control.control_info[j].stiffness) for j in joint_names
    ]
    damping_vals = [
        float(robot_config.control.control_info[j].damping) for j in joint_names
    ]
    mjcf_path = robot_config.asset.asset_file_name

    # Determine which obs component selects which future steps
    obs_future_step_selections: dict[str, list] = {}
    for k, cfg in actor_obs_configs.items():
        params = cfg.get_params() if hasattr(cfg, "get_params") else {}
        sel = params.get("future_steps")
        if sel is not None:
            obs_future_step_selections[k] = (
                list(range(1, sel + 1)) if isinstance(sel, int) else list(sel)
            )

    yaml_content = _build_yaml(
        onnx_in_names=actual_in_names,
        onnx_out_names=actual_out_names,
        onnx_name_to_key=onnx_name_to_key,
        input_shapes=input_shapes,
        joint_names=joint_names,
        body_names=body_names,
        stiffness=stiffness_vals,
        damping=damping_vals,
        pd_target_max_accel=pd_target_max_accel,
        anchor_body_name=anchor_body_name,
        anchor_body_index=anchor_body_index,
        root_body_index=root_body_index,
        num_bodies=num_bodies,
        num_dofs=num_dofs,
        mjcf_path=mjcf_path,
        control_dt=control_dt,
        physics_dt=physics_dt,
        decimation=decimation,
        future_step_indices=future_step_indices,
        obs_future_step_selections=obs_future_step_selections,
        checkpoint=str(checkpoint_path),
    )

    yaml_path = output_path / "unified_pipeline.yaml"
    import yaml

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=None, sort_keys=False)
    log.info(f"✓  YAML metadata → {yaml_path}")

    return onnx_path


# ---------------------------------------------------------------------------
# YAML builder
# ---------------------------------------------------------------------------


def _build_yaml(
    *,
    onnx_in_names,
    onnx_out_names,
    onnx_name_to_key,
    input_shapes,
    joint_names,
    body_names,
    stiffness,
    damping,
    pd_target_max_accel,
    anchor_body_name,
    anchor_body_index,
    root_body_index,
    num_bodies,
    num_dofs,
    mjcf_path,
    control_dt,
    physics_dt,
    decimation,
    future_step_indices,
    obs_future_step_selections,
    checkpoint,
) -> dict:
    """Build the rich YAML metadata dict."""

    # Describe each ONNX input in human-readable terms
    deploy_inputs = {}
    for onnx_name in onnx_in_names:
        key = onnx_name_to_key.get(onnx_name, onnx_name)
        shape = input_shapes.get(key, "unknown")
        entry: dict = {"onnx_name": onnx_name, "shape": shape}

        if "dof_pos" in key and "future" not in key:
            entry["description"] = "Joint positions"
            entry["source"] = "data.qpos[7:]"
        elif "dof_vel" in key and "future" not in key:
            entry["description"] = "Joint velocities"
            entry["source"] = "data.qvel[6:]"
        elif "anchor_rot" in key:
            entry["description"] = (
                f"Anchor body ({anchor_body_name}) orientation (xyzw)"
            )
            entry["source"] = (
                f"mujoco_wxyz_to_xyzw(data.xquat[{anchor_body_index + 1}])"
            )
            entry["derived_from"] = (
                f"rigid_body_rot[:, {anchor_body_index}]  "
                f"# anchor body = {anchor_body_name}"
            )
            entry["note"] = (
                "Uses ANCHOR body.  See state_utils.compute_anchor_rot_np()."
            )
        elif "root_local_ang_vel" in key:
            entry["description"] = (
                "Root body (pelvis) angular velocity in root local frame"
            )
            entry["source"] = (
                f"compute_root_local_ang_vel_np(body_rot, body_ang_vel, "
                f"root_body_index={root_body_index})"
            )
            entry["derived_from"] = (
                f"quat_rotate_inverse(rigid_body_rot[:, {root_body_index}], "
                f"rigid_body_ang_vel[:, {root_body_index}])"
            )
            entry["note"] = (
                "Uses ROOT body (pelvis), NOT anchor body!  "
                "See state_utils.compute_root_local_ang_vel_np()."
            )
        elif "mimic" in key and "dof_pos" in key:
            entry["description"] = "Reference motion joint positions (future steps)"
            entry["source"] = "MotionPlayer.get_future_references(frame, step_indices)"
            entry["future_step_indices"] = future_step_indices
            sels = obs_future_step_selections.get("mimic_deploy_target_poses")
            if sels:
                entry["obs_selects_steps"] = sels
                entry["note"] = (
                    f"All {len(future_step_indices)} future steps required; "
                    f"obs function internally selects indices {sels}."
                )
        elif "mimic" in key and "dof_vel" in key:
            entry["description"] = "Reference motion joint velocities (future steps)"
            entry["source"] = "MotionPlayer.get_future_references(frame, step_indices)"
            entry["future_step_indices"] = future_step_indices
        elif "mimic" in key and "rot" in key:
            entry["description"] = "Reference motion body rotations (future steps, xyzw)"
            entry["source"] = "MotionPlayer.get_future_references(frame, step_indices)"
            entry["future_step_indices"] = future_step_indices
            sels = obs_future_step_selections.get("mimic_deploy_target_poses")
            if sels:
                entry["obs_selects_steps"] = sels

        deploy_inputs[key] = entry

    content = {
        # Runtime metadata used by deploy scripts
        "_runtime": {
            "onnx_in_names": onnx_in_names,
            "onnx_out_names": onnx_out_names,
            "onnx_name_to_in_key": onnx_name_to_key,
        },
        # Simulator conventions -- any framework consuming this policy needs these
        "conventions": {
            "quat_format": "xyzw",
            "mujoco_body_index_offset": 1,
            "mujoco_free_joint_qpos_offset": 7,
            "mujoco_free_joint_qvel_offset": 6,
            "mujoco_cvel_layout": "ang_vel_first",
            "note": (
                "data.xquat[body_id + 1] for body quaternion (world body at 0); "
                "data.cvel[:, 0:3] is angular velocity; "
                "data.cvel[:, 3:6] is linear velocity."
            ),
        },
        # Robot topology
        "robot": {
            "mjcf_path": mjcf_path,
            "num_bodies": num_bodies,
            "num_dofs": num_dofs,
            "anchor_body_name": anchor_body_name,
            "anchor_body_index": anchor_body_index,
            "root_body_name": body_names[root_body_index],
            "root_body_index": root_body_index,
            "body_names": body_names,
            "joint_names": joint_names,
        },
        # Control parameters baked into the ONNX model
        "control": {
            "stiffness": stiffness,
            "damping": damping,
            "pd_target_max_accel": pd_target_max_accel,
            "action_ema_alpha": 0.8,
            "note": (
                "stiffness / damping are baked into the ONNX model "
                "(constant-folded); these values are for documentation only.  "
                "pd_target_max_accel and action_ema_alpha are NOT baked in -- "
                "the deploy script must apply them externally.  "
                "action_ema_alpha: a_applied = alpha * a_policy + (1-alpha) * a_prev "
                "(matches MujocoSimulator._action_filter_alpha)."
            ),
        },
        # Timing
        "timing": {
            "control_dt": control_dt,
            "physics_dt": physics_dt,
            "decimation": decimation,
        },
        # Reference motion info
        "motion": {
            "future_step_indices": future_step_indices,
            "future_dt_seconds": [round(s * control_dt, 6) for s in future_step_indices],
            "note": (
                "Provide all future_step_indices frames to the ONNX model. "
                "The obs function internally selects a subset."
            ),
        },
        # Per-input deployment contract
        "deploy_inputs": deploy_inputs,
        # Source checkpoint
        "metadata": {"checkpoint": checkpoint},
    }
    return content


# ---------------------------------------------------------------------------
# Attribute path resolver (mirrors export_utils._resolve_context_path)
# ---------------------------------------------------------------------------


def _resolve_attr_path(path: str, obj):
    """Resolve a dotted attribute path on *obj*, e.g. ``"current.dof_pos"``."""
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Export a tracker policy to ONNX (no simulator required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint file (e.g. results/my_exp/last.ckpt)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <checkpoint_dir>/compiled_models/)",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip onnxruntime validation after export",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    out = args.output
    if out is None:
        out = str(Path(args.checkpoint).parent / "compiled_models")

    onnx_file = export_tracker(
        checkpoint=args.checkpoint,
        output_dir=out,
        validate=not args.no_validate,
    )
    log.info(f"\nDone!  Model exported to: {onnx_file}")
    log.info(f"YAML sidecar: {onnx_file.with_suffix('.yaml')}")
