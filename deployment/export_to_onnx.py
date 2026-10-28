# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ONNX export for BeyondMimic trackers and deterministic Metatron priors.

Exports a ProtoMotions policy to a unified ONNX model
**without** running a simulator.  The script auto-detects actor observation
keys from the checkpoint's agent config and writes the typed
``DeploymentContract`` sidecar.

Typical actor obs for BM configs::

    noisy_reduced_coords_obs
    noisy_mimic_reduced_coords_target_poses
    historical_previous_processed_actions

The obs function ``build_reduced_coords_target_poses`` uses anchor-body
references (``future_anchor_rot/pos/vel/ang_vel``) rather than full-body
``future_rot``, and an action-history input is included.

Usage
-----
::

    python deployment/export_to_onnx.py \\
        --checkpoint exps/my_exp/last.ckpt \\
        --output     exps/my_exp/compiled_models/

Outputs
-------
``<output>/unified_pipeline.onnx``
    The exported ONNX model.

``<output>/unified_pipeline.yaml``
    Rich metadata / deployment contract.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------


def _build_samples(components, context, slot_names):
    """A tracing input per bound context path, read off the sample context.

    Every path an exportable observation binds must be published by the
    simulator-free sample context (``deployment/sample_context.py``); a bound
    path the context does not publish, or a kernel marked non-exportable, is
    refused here so the failure is at export, not on a robot. The published value
    is the tracing input. A published TENSOR is a graph input and must carry a
    ``deploy=`` host kind in ``context_views.py`` (else it is refused); a
    published NON-tensor (a Python int, the anchor body index) is folded in as a
    constant by the observation export module, never a graph input, so it needs
    no declaration. The shape exists only because a publisher produced it, so the
    exporter never fabricates one.

    A history-axis path needs either an action feedback rule or a measurement
    replay rule. Unknown histories are refused before tracing, not fabricated.
    """
    import torch

    from deployment.contract import ELEMENT_LABELS
    from deployment.input_history import history_specs
    from protomotions.envs.context_views import deploy_for
    from protomotions.envs.mdp_component import is_exportable

    samples = {}
    for name, component in components.items():
        if not is_exportable(component.get_compute_func()):
            raise ValueError(
                f"observation {name!r} is marked non-exportable: its kernel needs "
                "the whole field; the deployable graph takes the sampled map as a "
                "passthrough input, not a traced observation"
            )
        for path in component.get_bindings_dict().values():
            if path in samples:
                continue
            if path not in context:
                raise ValueError(
                    f"actor observation {name!r} binds {path!r}, which the "
                    "simulator-free sample context does not publish; a host "
                    "cannot feed a name no publisher produces. Check the "
                    "sensor suite, the mimic control and the history ring in "
                    "deployment/sample_context.py."
                )
            value = context[path]
            if not torch.is_tensor(value):
                # A published constant (the anchor body index, a Python int every
                # max-coords / target-pose / masked-mimic kernel carries): the
                # observation export module folds a non-tensor in as a constant,
                # never a graph input, so it needs no host-kind declaration.
                samples[path] = value
                continue
            kind = deploy_for(path)
            if kind is None:
                raise ValueError(
                    f"actor observation {name!r} binds {path!r}, which has no "
                    "deployment declaration (FieldPath(deploy=...)) in "
                    "context_views.py, so no host can feed it; declare it or "
                    "unbind it"
                )
            if kind not in ELEMENT_LABELS:
                # A misspelled host kind fails here, before torch.onnx.export
                # writes a file, rather than in _build_contract after it is on disk.
                raise ValueError(
                    f"actor observation {name!r} binds {path!r} with host kind "
                    f"{kind!r}, which has no element-label row; known kinds: "
                    f"{sorted(ELEMENT_LABELS)}"
                )
            if path in slot_names and history_specs(path, value.shape) == (None, None):
                raise ValueError(
                    f"actor observation {name!r} binds {path!r}, which the sample "
                    "context publishes with a history axis the host cannot feed: "
                    "no action feedback or measurement replay rule is declared"
                )
            samples[path] = value
    return samples


# The per-kind host-facing descriptions today's sidecar carries, kept so the
# typed contract stays a strict superset of the file the host reads today.
_SOURCE_TEXT = {
    "masked_mimic_ref_pos": "Body targets in metres in the host localization frame; full published body order.",
    "masked_mimic_ref_rot": "Body target orientations as xyzw unit quaternions in the host localization frame.",
    "masked_mimic_time_offsets": "Seconds from the current policy call to each goal; not absolute timestamps.",
    "masked_mimic_target_poses_masks": "Boolean per-target-step visibility; true means conditioned.",
    "masked_mimic_target_bodies_masks": "Flattened [step, conditionable_body, translation/rotation] visibility; true means conditioned.",
    "masked_mimic_anchor_translation_mask": "Per-step anchor visibility for [XY, Z]; true means conditioned.",
    "masked_mimic_anchor_rotation_mask": "Per-step anchor visibility for [full rotation, heading]; true means conditioned.",
    "masked_mimic_local_ref_pos": "Raw body targets in metres; anchor-relative conversion happens inside the graph, not on the host.",
    "masked_mimic_local_ref_rot": "Raw xyzw body targets; the graph removes target-anchor heading, not the host.",
    "masked_mimic_local_time_offsets": "Seconds from the current policy call to each local goal.",
    "masked_mimic_local_target_poses_masks": "Boolean per-local-target-step visibility; true means conditioned.",
    "masked_mimic_local_target_bodies_masks": "Flattened [step, conditionable_body, translation/rotation] local visibility; true means conditioned.",
    "masked_mimic_style_pose_samples": "Flattened [frame, conditionable_body, 9]: anchor-relative position (metres) and tangent/normal rotation features, using each style frame's own heading.",
    "masked_mimic_style_velocity_samples": "Flattened [frame, conditionable_body, 6]: linear velocity (m/s) and angular velocity (rad/s), in each style frame's anchor-heading frame.",
    "masked_mimic_style_condition_masks": "Per-conditionable-body style visibility; true means conditioned.",
    "masked_mimic_style_frame_masks": "Per-style-frame visibility; true means conditioned.",
    "masked_mimic_average_velocity": "World-frame anchor XY velocity command in m/s.",
    "masked_mimic_velocity_command_mask": "Velocity-command visibility; true means conditioned.",
    "odom_start_xy": (
        "Robot/reference anchor XY at motion start in the deployment "
        "world frame. For zero-start deploy frames this is [0, 0]."
    ),
    "odom_disp_start": (
        "Odometer XY displacement from motion start, expressed in "
        "the start-heading frame."
    ),
    "odom_start_heading_inv": "Inverse yaw-only heading captured at motion start.",
    "reference_motion_anchor_pos": (
        "Reference anchor position at the CURRENT step, aligned to the "
        "robot's start frame. Two uses: it is the point the odometer "
        "offset is measured to (supply the raw odometer ingredients "
        "odom_disp_start, odom_start_xy, odom_start_heading_inv alongside "
        "it), and it is the origin for the reference's own travel over "
        "the horizon. Only the anchor is needed, no other reference body."
    ),
    "reference_motion_anchor_pos_future": (
        "Reference anchor position at each future step, same frame as "
        "mimic.ref_anchor_pos. Minus the current-step value, this is how "
        "far the reference itself travels over the horizon, which is what "
        "tells the policy how fast to move."
    ),
    "reference_motion_body_pos": (
        "Current-step reference body positions, aligned to the robot's "
        "start frame. The baked-FK tracker path does NOT request this -- "
        "it needs mimic.ref_anchor_pos alone -- so this only appears for "
        "observations that genuinely read every reference body."
    ),
    "reference_motion_body_pos_future": (
        "Reference body positions at each future step, in the same "
        "frame as mimic.ref_anchor_pos. Only observations that read "
        "every reference body need this; the baked-FK path does not."
    ),
    "reference_motion_body_rot": (
        "Reference body orientations at each future step, rotated by "
        "the start-heading offset."
    ),
    "anchor_pos": (
        "Robot anchor position in the deployment world frame. This "
        "needs a position estimate on the robot; prefer observations "
        "that read only the anchor rotation if you do not have one."
    ),
}

_ACTION_HISTORY_SOURCE = (
    "The applied command in the graph's {output} coordinate system. Feed back "
    "any deploy-side edit, not the rejected network prediction. Slot i holds "
    "the values from steps_back[i] "
    "policy calls ago; until that many calls have run since a reset, "
    "it holds fill_before_available."
)


def _deployable_policy_in_keys(model_config):
    """The obs keys the deployable policy reads, and the actor config if any.

    A plain actor (``model.actor``) exposes its own ``in_keys``. The MaskedMimic
    model has no actor: its deployable path is prior -> latent -> trunk, so the
    inputs are the prior's plus the trunk's, minus the internal latent key the
    prior produces. Returns ``(actor_config_or_None, in_keys)``; ``actor_config``
    is None for a MaskedMimic export, and the caller builds the whole model.
    """
    actor_cfg = getattr(model_config, "actor", None)
    if actor_cfg is not None:
        return actor_cfg, list(actor_cfg.in_keys)
    from protomotions.agents.common.latent import VAE_LATENT_KEY

    # A conditional-VAE prior can emit a decoder condition alongside the latent;
    # the trunk reads the prior's outputs as internal data, never as host inputs,
    # so drop every prior output and the latent from the deployable inputs.
    produced = set(model_config.prior.out_keys) | {VAE_LATENT_KEY}
    trunk_in = [k for k in model_config.trunk.in_keys if k not in produced]
    in_keys = list(dict.fromkeys(list(model_config.prior.in_keys) + trunk_in))
    return None, in_keys


def _build_deployable_policy(actor_cfg, model_config):
    """Construct the module the exporter traces through ``forward_deployable``.

    For a plain actor that is the actor itself. For MaskedMimic there is no
    actor: build the whole model (the deployable path is prior -> latent ->
    trunk; the encoder stays privileged and is never traced). MetatronPriorExport
    supplies batch-shaped zero noise at each call; no rollout state or config
    rewrite is needed.
    """
    from protomotions.utils.hydra_replacement import get_class

    config = actor_cfg if actor_cfg is not None else model_config
    return get_class(config._target_)(config)


def _load_deployable_weights(policy, actor_cfg, model_state):
    """Load the deployable weights from the checkpoint's ``model`` state dict.

    A plain actor keeps the single ``_actor.`` prefix and is materialized by a
    sample forward first. MaskedMimic loads its full saved model strictly,
    including lazy layers and normalization buffers. Its encoder is checked at
    load time but never executed in the exported prior path.
    """
    if actor_cfg is not None:
        actor_state = {
            k[len("_actor.") :]: v
            for k, v in model_state.items()
            if k.startswith("_actor.")
        }
        policy.load_state_dict(actor_state)
        return
    from protomotions.agents.utils.normalization import (
        materialize_lazy_running_stats_from_state_dict,
    )

    policy.materialize_from_state_dict(model_state)
    materialize_lazy_running_stats_from_state_dict(policy, model_state)
    policy.load_state_dict(model_state, strict=True)


def _build_contract(
    *,
    actual_in_names,
    actual_out_names,
    onnx_name_to_key,
    input_shapes,
    input_dtypes,
    future_names,
    obs_source_keys,
    obs_input_keys,
    robot_config,
    env_config,
    future_step_indices,
    pd_target_max_accel,
    checkpoint,
    control_type,
    sensor_metadata,
    onnx_sha256,
):
    """Assemble the typed DeploymentContract this export writes as the sidecar.

    Each graph input is described from the sample context and the one host kind
    the context path's declaration carries: the shape and dtype are the published
    tensor's own; the element labels come from the kind-keyed ``element_labels``
    table (the coordinate width read off the tensor); the future axis and the
    history axis come from the horizon and depth probes; the provider is derived
    from the path's namespace and, for a sensor path, the channel the name
    derives from (``derived_sensor_channel``). For the action histories the
    feedback output and the fill behaviour come from the host protocol. The
    blocks come from the contract's own builders over the resolved robot and
    action configs. The result is a strict superset of today's hand-built YAML
    plus the schema key.
    """
    from deployment.contract import (
        SCHEMA,
        DeploymentContract,
        InputSpec,
        MotionBlock,
        OutputSpec,
        RuntimeBlock,
        build_control_block,
        build_robot_block,
        build_timing_block,
        element_labels,
        provider_for_path,
    )
    from protomotions.envs.context_views import deploy_for
    from deployment.input_history import history_specs
    from deployment.sample_context import reference_control_config
    from protomotions.envs.control.masked_mimic_control import MaskedMimicControlConfig
    from protomotions.envs.control.masked_mimic_prompt import GoalCanvas

    reference_config = reference_control_config(env_config)
    goal_canvas = (
        GoalCanvas.derive(reference_config, robot_config)
        if isinstance(reference_config, MaskedMimicControlConfig)
        else None
    )

    joint_names = tuple(robot_config.kinematic_info.dof_names)
    body_names = tuple(robot_config.kinematic_info.body_names)
    anchor_name = robot_config.resolved_anchor_body_name
    num_dofs = len(joint_names)
    num_future = len(future_step_indices)
    action_config = env_config.action_config
    params = action_config.params if action_config is not None else {}
    pd_action_offset = params.get("pd_action_offset")

    host_to_source = dict(zip(obs_input_keys, obs_source_keys))
    key_values = set(onnx_name_to_key.values())

    inputs = []
    for onnx_name in actual_in_names:
        host_key = onnx_name_to_key.get(onnx_name)
        if host_key is None:
            # Every graph input maps to a context path (the mapping step above
            # fails loud otherwise); a None here would silently drop an input
            # the host must feed, so refuse rather than ship a short sidecar.
            raise ValueError(
                f"ONNX input {onnx_name!r} has no context-path mapping; the "
                "sidecar would omit an input the policy requires"
            )
        source_path = host_to_source[host_key]
        kind = deploy_for(source_path)
        shape = tuple(input_shapes[host_key])
        history, observation_history = history_specs(
            source_path, shape, pd_action_offset
        )
        source = _SOURCE_TEXT.get(kind)
        if history is not None:
            source = _ACTION_HISTORY_SOURCE.format(output=history.output_key)
        inputs.append(
            InputSpec(
                onnx_name=onnx_name,
                key=host_key,
                source_namespace=source_path.split(".", 1)[0],
                kind=kind,
                provider=provider_for_path(source_path),
                shape=shape,
                dtype=input_dtypes[host_key],
                element_names=element_labels(
                    kind, joint_names, body_names, anchor_name, shape[-1]
                ),
                # Body masks flatten the horizon with body/property axes;
                # their width is not a count of future steps.
                future_steps=(
                    reference_config.num_masked_future_steps
                    if source_path.startswith("masked_mimic.")
                    else num_future
                )
                if source_path in future_names
                else None,
                history=history,
                source=source,
                observation_history=observation_history,
            )
        )

    outputs = (
        OutputSpec("actions", "actions", (1, num_dofs), None),
        OutputSpec(
            "joint_pos_targets", "joint_pos_targets", (1, num_dofs), joint_names
        ),
        OutputSpec(
            "stiffness_targets", "stiffness_targets", (1, num_dofs), joint_names
        ),
        OutputSpec("damping_targets", "damping_targets", (1, num_dofs), joint_names),
    )

    runtime = RuntimeBlock(
        onnx_in_names=tuple(actual_in_names),
        onnx_out_names=tuple(actual_out_names),
        onnx_name_to_in_key=dict(onnx_name_to_key),
        passthrough_keys=tuple(k for k in obs_input_keys if k not in key_values),
        obs_context_keys=tuple(obs_input_keys),
    )

    return DeploymentContract(
        schema=SCHEMA,
        inputs=tuple(inputs),
        outputs=outputs,
        robot=build_robot_block(
            robot_config,
            None if goal_canvas is None else goal_canvas.conditionable_bodies,
        ),
        control=build_control_block(action_config, robot_config, pd_target_max_accel),
        timing=build_timing_block(robot_config),
        motion=MotionBlock(future_step_indices=tuple(future_step_indices)),
        runtime=runtime,
        sensors=sensor_metadata,
        checkpoint=checkpoint,
        control_type=control_type,
        onnx_sha256=onnx_sha256,
        masked_mimic=goal_canvas,
    )


def export_tracker(
    checkpoint: str,
    output_dir: str,
    validate: bool = True,
    leapp: bool = False,
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
    leapp:
        Also package the graph and its typed contract for LEAPP inference.

    Returns
    -------
    Path to the exported ``.onnx`` file.
    """
    # Load the one training bundle saved beside the checkpoint, then export from
    # it. The load is the only step that needs a file on disk; everything else
    # runs off the in-memory configs in export_from_configs.
    from protomotions.utils.experiment_configs import (
        load_experiment_configs_from_checkpoint,
    )

    log.info(f"Loading configs from the bundle beside {Path(checkpoint)}")
    resolved = load_experiment_configs_from_checkpoint(Path(checkpoint))
    return export_from_configs(
        resolved, checkpoint, output_dir, validate=validate, leapp=leapp
    )


def export_from_configs(
    resolved,
    checkpoint: str,
    output_dir: str,
    validate: bool = True,
    leapp: bool = False,
) -> Path:
    """Export from resolved configs, without applying inference overrides."""
    import torch
    from tensordict import TensorDict

    from protomotions.utils.export_utils import (
        ObservationExportModule,
        ActionExportModule,
        UnifiedPipelineModule,
    )
    from protomotions.envs.body_selection import resolve_body_selections
    from deployment.sample_context import build_sample_context, reference_control_config
    from deployment.metatron_policy import MetatronPriorExport
    from protomotions.envs.steps import normalize_steps

    checkpoint_path = Path(checkpoint)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    robot_config = resolved.robot
    env_config = resolved.env
    agent_config = resolved.agent
    simulator_config = resolved.simulator
    from deployment.tracker_inputs import (
        tracker_sensor_metadata,
        tracker_input_keys as _tracker_input_keys,
    )

    actor_cfg, deployable_in_keys = _deployable_policy_in_keys(agent_config.model)
    is_metatron = actor_cfg is None
    log.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = _build_deployable_policy(actor_cfg, agent_config.model)
    if is_metatron:
        _load_deployable_weights(actor, actor_cfg, ckpt["model"])
        actor = MetatronPriorExport(actor)
    actor.eval()

    # ------------------------------------------------------------------
    # 2. Auto-detect actor obs keys from agent config
    # ------------------------------------------------------------------
    actor_in_keys = list(deployable_in_keys)
    actor_obs_keys = set(actor_in_keys)
    log.info(f"Auto-detected actor obs keys: {actor_in_keys}")
    sensor_metadata = tracker_sensor_metadata(env_config, robot_config, actor_in_keys)

    # ------------------------------------------------------------------
    # 3. Extract dimensions from configs
    # ------------------------------------------------------------------
    num_dofs = robot_config.kinematic_info.num_dofs
    num_bodies = len(robot_config.kinematic_info.body_names)
    # The anchor body's name, resolved through the derived anchor index. The
    # robot, joint and body details are read off the configs by the contract
    # builders (build_robot_block/build_control_block), not threaded here.
    anchor_body_name = robot_config.resolved_anchor_body_name
    anchor_body_index = robot_config.anchor_body_index

    mimic_ctrl_cfg = reference_control_config(env_config)
    future_step_indices = normalize_steps(mimic_ctrl_cfg.future_steps)
    num_future_steps = len(future_step_indices)

    # Timing comes from the robot's control rate and MuJoCo physics fps; the
    # host decimation is exactly control_dt / physics_dt (an integer, since fps
    # is a multiple of control_hz). No silent fallback.
    control_dt = robot_config.control_dt
    physics_dt = 1.0 / robot_config.simulation_params.mujoco.fps
    decimation = round(control_dt / physics_dt)
    pd_target_max_accel = None
    if simulator_config is not None:
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
    # 4. Build the observation components and the tracing samples
    # ------------------------------------------------------------------
    actor_obs_configs = {
        k: v
        for k, v in env_config.observation_components.items()
        if k in actor_obs_keys
    }
    missing = actor_obs_keys - set(env_config.observation_components.keys())
    if missing:
        raise ValueError(
            f"Actor requires obs keys {actor_obs_keys} but these are missing "
            f"from env_config.observation_components: {missing}"
        )
    # A BodySelection in a deployable recipe becomes concrete indices before the
    # trace, so the exported graph carries the bodies, not the selection object.
    actor_obs_configs = dict(resolve_body_selections(actor_obs_configs, robot_config))

    # Build one simulator-free context the way the environment does: a synthetic
    # robot state through the real sensor runtime, a history ring sized by the
    # actor's own observations, and the mimic control over a reference stand-in.
    # Every bound context path is materialised from the publisher that produces
    # it, so the exporter never fabricates a shape by hand; the two probes find
    # which names carry a future axis (the mimic horizon) and which a history
    # axis (the ring depth).
    sample_context, future_names, slot_names = build_sample_context(
        robot_config, env_config, actor_obs_configs, device="cpu"
    )
    samples = _build_samples(actor_obs_configs, sample_context, slot_names)

    # ------------------------------------------------------------------
    # 5. Build ObservationExportModule (actor obs only, for export)
    # ------------------------------------------------------------------
    log.info(f"Observation components for export: {list(actor_obs_configs.keys())}")
    obs_module = ObservationExportModule(actor_obs_configs, samples, device="cpu")
    obs_module.eval()

    obs_source_keys = obs_module.get_input_keys()
    obs_input_keys = (
        obs_source_keys if is_metatron else _tracker_input_keys(obs_source_keys)
    )
    obs_output_keys = obs_module.get_output_keys()
    log.info(f"  Context input keys:  {obs_input_keys}")
    log.info(f"  Observation outputs: {obs_output_keys}")

    # ------------------------------------------------------------------
    # 6. Materialize observations; load PPO actor weights if not already loaded
    # ------------------------------------------------------------------
    # nn.LazyLinear needs one forward pass to materialise concrete layer shapes.
    actor.eval()
    materialize_inputs = [samples[k] for k in obs_source_keys]
    with torch.no_grad():
        mock_obs_out = obs_module(*materialize_inputs)
        mock_obs_td = TensorDict(
            {k: v for k, v in zip(obs_output_keys, mock_obs_out)},
            batch_size=[1],
        )
        actor.forward_deployable(
            mock_obs_td
        )  # materialises LazyLinear without exploration

    log.info(
        "LazyLinear materialised with actor obs: "
        + ", ".join(f"{k}={v.shape[-1]}" for k, v in mock_obs_td.items())
    )

    if not is_metatron:
        _load_deployable_weights(actor, actor_cfg, ckpt["model"])
    actor.eval()

    log.info(f"Deployable in_keys:  {deployable_in_keys}")
    log.info(f"Deployable out_keys: {list(actor.out_keys)}")

    # Verify the deployable obs keys are covered by our obs module. For
    # MaskedMimic the model's own in_keys include the privileged encoder inputs,
    # so this checks the deployable in_keys, not actor.in_keys.
    uncovered = set(deployable_in_keys) - set(obs_output_keys)
    if uncovered:
        raise ValueError(
            f"The deployable policy requires obs keys not produced by "
            f"ObservationExportModule: {uncovered}.  Check observation_components."
        )

    # ------------------------------------------------------------------
    # 7. Build ActionExportModule
    # ------------------------------------------------------------------
    action_module = ActionExportModule(env_config.action_config, device="cpu")
    action_module.eval()

    # ------------------------------------------------------------------
    # 8. Compose UnifiedPipelineModule
    # ------------------------------------------------------------------
    unified = UnifiedPipelineModule(
        observation_module=obs_module,
        policy_module=actor,
        action_module=action_module,
        policy_in_keys=deployable_in_keys,
    )
    unified.cpu().eval()

    # ------------------------------------------------------------------
    # 9. Collect sample inputs and verify forward pass
    # ------------------------------------------------------------------
    sample_inputs = [samples[k] for k in obs_source_keys]
    input_shapes = {k: list(v.shape) for k, v in zip(obs_input_keys, sample_inputs)}
    # dtype is the published tensor's own, so a non-float channel is described as
    # it is rather than forced to float32 by a hand-written constant.
    input_dtypes = {
        k: str(v.dtype).rsplit(".", 1)[-1]
        for k, v in zip(obs_input_keys, sample_inputs)
    }

    with torch.no_grad():
        actions, pd_targets, stiffness_t, damping_t = unified(*sample_inputs)

    log.info(
        f"Forward pass OK: actions={list(actions.shape)}, "
        f"pd_targets={list(pd_targets.shape)}"
    )

    # ------------------------------------------------------------------
    # 10. Export to ONNX
    # ------------------------------------------------------------------
    def _sanitize(name: str) -> str:
        return name.replace(".", "_").replace("[", "_").replace("]", "_")

    onnx_input_names = [_sanitize(k) for k in obs_input_keys]
    onnx_output_names = [
        "actions",
        "joint_pos_targets",
        "stiffness_targets",
        "damping_targets",
    ]

    onnx_path = output_path / "unified_pipeline.onnx"
    log.info(f"Exporting ONNX to {onnx_path} ...")
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
    log.info(f"ONNX exported -> {onnx_path}")

    # ------------------------------------------------------------------
    # 11. Read back actual ONNX names (ONNX may rename inputs)
    # ------------------------------------------------------------------
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual_in_names = [inp.name for inp in session.get_inputs()]
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
            # Fail at export, not on the robot: an unmapped graph input would be
            # dropped from the contract and the host would never feed it. This
            # happens if two keys collapse to the same sanitized name and ONNX
            # disambiguates with a suffix the strip list above does not cover.
            raise ValueError(
                f"Cannot map ONNX input {onnx_name!r} back to a context path. "
                f"Sanitized candidates: {sorted(sanitized_to_key)}. A name "
                "collision after _sanitize produced an ONNX suffix the exporter "
                "does not strip; disambiguate the context paths or extend the "
                "suffix list."
            )

    # ------------------------------------------------------------------
    # 12. Validate with onnxruntime
    # ------------------------------------------------------------------
    if validate:
        import numpy as np

        log.info("Validating with onnxruntime ...")
        key_to_tensor = {k: t for k, t in zip(obs_input_keys, sample_inputs)}
        # Every graph input maps to a context path (the name mapping refuses an
        # unmappable one), so no name is filtered out here.
        ort_inputs = {
            name: key_to_tensor[onnx_name_to_key[name]].detach().numpy()
            for name in actual_in_names
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
            if diff >= 1e-4:
                # The export boundary is where a stochastic or wrong-key graph
                # must fail: the ONNX output has to equal the traced deployable
                # output (the deterministic mean), not merely warn.
                raise ValueError(
                    f"The exported ONNX graph's {name} output disagrees with the "
                    f"traced policy by max_diff {diff:.2e} (tolerance 1e-4). A "
                    "stochastic or wrong-key graph must fail at export, not on a "
                    "robot."
                )
            log.info(f"  OK  {name}: max_diff = {diff:.2e}")

        # The sidecar claims joint_pos_targets = pd_action_offset + action_scale *
        # action. Check that against what the graph actually produced, so a wrong
        # or stale conversion is caught at export rather than on a robot.
        _acfg_v = env_config.action_config
        _params = _acfg_v.params if _acfg_v is not None else {}
        _scale = _params.get("action_scale")
        _offset = _params.get("pd_action_offset")
        if _scale is not None and _offset is not None:
            _s = np.asarray(_scale, dtype=np.float64).reshape(-1)
            _o = np.asarray(_offset, dtype=np.float64).reshape(-1)
            _a = ort_outputs[0].astype(np.float64).reshape(-1)
            _recon = _o + _s * _a
            _d = float(
                np.abs(_recon - ort_outputs[1].astype(np.float64).reshape(-1)).max()
            )
            if _d < 1e-4:
                log.info(
                    f"  OK  sidecar action conversion reproduces the graph: max_diff = {_d:.2e}"
                )
            else:
                raise ValueError(
                    f"The action_scale/pd_action_offset about to be written to the sidecar do not "
                    f"reproduce the exported graph's joint_pos_targets (max_diff {_d:.2e}). The "
                    f"sidecar would describe a different conversion than the ONNX performs, which "
                    f"is exactly the mismatch these fields exist to make visible."
                )
        log.info("Validation complete")

    # ------------------------------------------------------------------
    # 13. Build and write the typed deployment contract
    # ------------------------------------------------------------------
    import hashlib

    onnx_sha256 = hashlib.sha256(onnx_path.read_bytes()).hexdigest()

    # The action function carries its own name; BUILT_IN_PD is the default for a
    # config that does not name one.
    control_type = "BUILT_IN_PD"
    action_cfg = env_config.action_config
    if hasattr(action_cfg, "_target_"):
        control_type = action_cfg._target_.rsplit(".", 1)[-1]

    contract = _build_contract(
        actual_in_names=actual_in_names,
        actual_out_names=actual_out_names,
        onnx_name_to_key=onnx_name_to_key,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        future_names=future_names,
        obs_source_keys=obs_source_keys,
        obs_input_keys=obs_input_keys,
        robot_config=robot_config,
        env_config=env_config,
        future_step_indices=future_step_indices,
        pd_target_max_accel=pd_target_max_accel,
        checkpoint=str(checkpoint_path),
        control_type=control_type,
        sensor_metadata=sensor_metadata,
        onnx_sha256=onnx_sha256,
    )

    yaml_path = output_path / "unified_pipeline.yaml"
    contract.to_yaml(yaml_path)
    log.info(f"YAML metadata -> {yaml_path}")

    if leapp:
        from deployment.native.tools.leapp_export import export_pipeline_bundle

        bundle = export_pipeline_bundle(onnx_path, contract, output_path / "leapp")
        log.info(f"LEAPP bundle -> {bundle}")

    return onnx_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Export a tracker or deterministic Metatron prior to ONNX/LEAPP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint file (e.g. exps/my_exp/last.ckpt)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <checkpoint_dir>/compiled_models/)",
    )
    p.add_argument(
        "--leapp",
        action="store_true",
        help="Also write a LEAPP bundle with the actual exported input contract",
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
        leapp=args.leapp,
    )
    log.info(f"\nDone!  Model exported to: {onnx_file}")
    log.info(f"YAML sidecar: {onnx_file.with_suffix('.yaml')}")
