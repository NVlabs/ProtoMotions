# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for common MdpComponent configurations.

These factories reduce boilerplate in experiment configs by providing
pre-configured MdpComponent instances for frequently used components.

Usage in experiment configs:
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        previous_actions_factory,
        mimic_tracking_rewards_factory,
        tracking_error_term_factory,
    )

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(),
    }

    reward_components = {
        **mimic_tracking_rewards_factory(gt_weight=0.5, gr_weight=0.3),
        "action_smoothness": action_smoothness_factory(weight=-0.02),
    }

MdpComponent Parameters
------------------------

- **compute_func**: Pure tensor function that performs the computation
- **dynamic_vars**: Runtime-resolved context paths (become ONNX inputs)
- **static_params**: Compile-time constants (baked into ONNX graph)

Example:
    MdpComponent(
        compute_func=compute_fn,
        dynamic_vars={"tensor_input": EnvContext.current.dof_pos},  # ONNX input
        static_params={"local_obs": True, "weight": 0.5},           # ONNX constants
    )
"""

from typing import Any, Dict, List, Optional, Union

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent


# =============================================================================
# Observation Factories
# =============================================================================


def max_coords_obs_factory(
    use_noisy: bool = False,
    local_obs: bool = True,
    root_height_obs: bool = True,
    observe_contacts: bool = False,
) -> MdpComponent:
    """Factory for humanoid max-coords observations.

    Args:
        use_noisy: If True, use noisy state (for actor with domain randomization).
        local_obs: If True, use heading-aligned local coordinates.
        root_height_obs: If True, include root height observation.
        observe_contacts: If True, include contact observations.

    Returns:
        MdpComponent configured for max-coords observations.
    """
    from protomotions.envs.obs import compute_humanoid_max_coords_observations

    state = EnvContext.noisy if use_noisy else EnvContext.current
    ground = EnvContext.noisy_ground_heights if use_noisy else EnvContext.ground_heights

    return MdpComponent(
        compute_func=compute_humanoid_max_coords_observations,
        dynamic_vars={
            "body_pos": state.rigid_body_pos,
            "body_rot": state.rigid_body_rot,
            "body_vel": state.rigid_body_vel,
            "body_ang_vel": state.rigid_body_ang_vel,
            "ground_height": ground,
            "body_contacts": EnvContext.body_contacts,
        },
        static_params={
            "local_obs": local_obs,
            "root_height_obs": root_height_obs,
            "observe_contacts": observe_contacts,
            "w_last": True,
        },
    )


def reduced_coords_obs_factory(
    use_noisy: bool = False,
    root_height_obs: bool = False,
    root_vel_obs: bool = False,
) -> MdpComponent:
    """Factory for humanoid reduced-coords observations.

    Args:
        use_noisy: If True, use noisy state (for actor with domain randomization).
        root_height_obs: If True, include root height.
        root_vel_obs: If True, include root linear velocity.

    Returns:
        MdpComponent configured for reduced-coords observations.
    """
    from protomotions.envs.obs import compute_humanoid_reduced_coords_observations

    state = EnvContext.noisy if use_noisy else EnvContext.current
    ground = EnvContext.noisy_ground_heights if use_noisy else EnvContext.ground_heights

    bindings = {
        "dof_pos": state.dof_pos,
        "dof_vel": state.dof_vel,
        "anchor_rot": state.anchor_rot,
        "root_local_ang_vel": state.root_local_ang_vel,
    }

    if root_height_obs:
        bindings["root_pos"] = state.root_pos
        bindings["ground_height"] = ground

    if root_vel_obs:
        bindings["root_rot"] = state.root_rot
        bindings["root_vel"] = state.root_vel

    return MdpComponent(
        compute_func=compute_humanoid_reduced_coords_observations,
        dynamic_vars=bindings,
        static_params={
            "root_height_obs": root_height_obs,
            "root_vel_obs": root_vel_obs,
            "w_last": True,
        },
    )


def historical_max_coords_obs_factory(
    use_noisy: bool = False,
    local_obs: bool = True,
    root_height_obs: bool = True,
    observe_contacts: bool = False,
    history_steps: Optional[Union[int, list]] = None,
) -> MdpComponent:
    """Factory for historical max-coords observations.

    Args:
        use_noisy: If True, use noisy historical state.
        local_obs: If True, use heading-aligned local coordinates.
        root_height_obs: If True, include root height observation.
        observe_contacts: If True, include contact observations.
        history_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 4, 8, 16]). None = use all.

    Returns:
        MdpComponent configured for historical max-coords observations.
    """
    from protomotions.envs.obs import compute_historical_max_coords_from_state

    hist = EnvContext.noisy_historical if use_noisy else EnvContext.historical

    params = {
        "local_obs": local_obs,
        "root_height_obs": root_height_obs,
        "observe_contacts": observe_contacts,
        "w_last": True,
    }
    if history_steps is not None:
        params["history_steps"] = history_steps

    return MdpComponent(
        compute_func=compute_historical_max_coords_from_state,
        dynamic_vars={
            "historical_rigid_body_pos": hist.rigid_body_pos,
            "historical_rigid_body_rot": hist.rigid_body_rot,
            "historical_rigid_body_vel": hist.rigid_body_vel,
            "historical_rigid_body_ang_vel": hist.rigid_body_ang_vel,
            "historical_ground_heights": hist.ground_heights,
            "historical_body_contacts": hist.body_contacts,
        },
        static_params=params,
    )


def historical_reduced_coords_obs_factory(
    use_noisy: bool = False,
) -> MdpComponent:
    """Factory for historical reduced-coords observations.

    Args:
        use_noisy: If True, use noisy historical state.

    Returns:
        MdpComponent configured for historical reduced-coords observations.
    """
    from protomotions.envs.obs import compute_historical_reduced_coords_from_state

    hist = EnvContext.noisy_historical if use_noisy else EnvContext.historical

    return MdpComponent(
        compute_func=compute_historical_reduced_coords_from_state,
        dynamic_vars={
            "historical_dof_pos": hist.dof_pos,
            "historical_dof_vel": hist.dof_vel,
            "historical_root_rot": hist.root_rot,
            "historical_root_local_ang_vel": hist.root_local_ang_vel,
            "historical_anchor_rot": hist.anchor_rot,
        },
        static_params={"w_last": True},
    )


def previous_actions_factory(
    history_steps: int = 1, processed: bool = False
) -> MdpComponent:
    """Factory for previous actions observation.

    Args:
        history_steps: Number of historical steps to include.
        processed: If True, use processed actions (after tanh/clamp, before PD scaling).
                   If False (default), use raw actions from the policy.

    Returns:
        MdpComponent configured for previous actions.
    """
    from protomotions.envs.obs import compute_historical_actions_from_state

    actions_field = (
        EnvContext.historical.processed_actions
        if processed
        else EnvContext.historical.actions
    )

    return MdpComponent(
        compute_func=compute_historical_actions_from_state,
        dynamic_vars={
            "historical_actions": actions_field,
        },
        static_params={"history_steps": history_steps},
    )


def nearest_surface_obs_factory(
    body_ids: Optional[List[int]] = None,
    terrain_horizontal_scale: float = 0.1,
) -> MdpComponent:
    """Factory for vectors from bodies to nearest terrain or object surface."""
    from protomotions.envs.obs import compute_nearest_surface_vectors

    return MdpComponent(
        compute_func=compute_nearest_surface_vectors,
        dynamic_vars={
            "rigid_body_pos": EnvContext.current.rigid_body_pos,
            "root_pos": EnvContext.current.root_pos,
            "root_rot": EnvContext.current.root_rot,
            "height_points": EnvContext.terrain.height_points,
            "height_samples": EnvContext.terrain.height_samples,
            "object_pos": EnvContext.scene.object_pos,
            "object_rot": EnvContext.scene.object_rot,
            "neutral_pointclouds": EnvContext.scene.neutral_pointclouds,
            "object_valid_mask": EnvContext.scene.object_valid_mask,
        },
        static_params={
            "terrain_horizontal_scale": terrain_horizontal_scale,
            "body_ids": body_ids,
        },
    )


def mimic_target_poses_max_coords_factory(
    use_noisy: bool = False,
    with_velocities: bool = True,
    with_relative: bool = True,
    future_steps: Optional[Union[int, list]] = None,
) -> MdpComponent:
    """Factory for mimic target poses (max-coords format).

    Args:
        use_noisy: If True, use noisy current state for relative computations.
        with_velocities: If True, include velocity information.
        with_relative: If True, include relative pose observations.
        future_steps: Steps to select from MimicControl's future buffer.
            None = use all steps. Int N = first N steps. List = specific step indices.

    Returns:
        MdpComponent configured for max-coords target poses.
    """
    from protomotions.envs.obs import build_max_coords_target_poses

    state = EnvContext.noisy if use_noisy else EnvContext.current

    static_params = {
        "with_velocities": with_velocities,
        "with_relative": with_relative,
        "w_last": True,
    }
    if future_steps is not None:
        static_params["future_steps"] = future_steps

    return MdpComponent(
        compute_func=build_max_coords_target_poses,
        dynamic_vars={
            "current_state_body_pos": state.rigid_body_pos,
            "current_state_body_rot": state.rigid_body_rot,
            "current_state_body_vel": state.rigid_body_vel,
            "current_state_body_ang_vel": state.rigid_body_ang_vel,
            "mimic_ref_pos": EnvContext.mimic.future_pos,
            "mimic_ref_rot": EnvContext.mimic.future_rot,
            "mimic_ref_vel": EnvContext.mimic.future_vel,
            "mimic_ref_ang_vel": EnvContext.mimic.future_ang_vel,
        },
        static_params=static_params,
    )


def mimic_target_poses_future_rel_factory(
    use_noisy: bool = False,
    future_steps: Optional[int] = None,
) -> MdpComponent:
    """Factory for mimic target poses (future-relative format).

    Args:
        use_noisy: If True, use noisy current state for relative computations.
        future_steps: Number of future steps to include. None = use all available.

    Returns:
        MdpComponent configured for future-relative target poses.
    """
    from protomotions.envs.obs import build_max_coords_target_poses_future_rel

    state = EnvContext.noisy if use_noisy else EnvContext.current

    params = {"w_last": True}
    if future_steps is not None:
        params["future_steps"] = future_steps

    return MdpComponent(
        compute_func=build_max_coords_target_poses_future_rel,
        dynamic_vars={
            "current_state_body_pos": state.rigid_body_pos,
            "current_state_body_rot": state.rigid_body_rot,
            "mimic_ref_pos": EnvContext.mimic.future_pos,
            "mimic_ref_rot": EnvContext.mimic.future_rot,
        },
        static_params=params,
    )


def mimic_target_poses_reduced_coords_factory(
    use_noisy: bool = False,
    include_dof_vel: bool = True,
    include_xy_offset: bool = False,
    include_height: bool = False,
    include_anchor_vel: bool = False,
    include_anchor_ang_vel: bool = False,
    zero_xy_offset: bool = False,
) -> MdpComponent:
    """Factory for mimic target poses (reduced-coords format).

    Args:
        use_noisy: If True, use noisy current state.
        include_dof_vel: If True, include DOF velocities.
        include_xy_offset: If True, include XY translation offset in local frame.
        include_height: If True, include absolute height.
        include_anchor_vel: If True, include anchor linear velocity.
        include_anchor_ang_vel: If True, include anchor angular velocity.
        zero_xy_offset: If True, emit zeros for XY offset (for inference).

    Returns:
        MdpComponent configured for reduced-coords target poses.
    """
    from protomotions.envs.obs import build_reduced_coords_target_poses

    state = EnvContext.noisy if use_noisy else EnvContext.current

    return MdpComponent(
        compute_func=build_reduced_coords_target_poses,
        dynamic_vars={
            "current_state_anchor_rot": state.anchor_rot,
            "current_state_anchor_pos": state.anchor_pos,
            "mimic_ref_anchor_rot": EnvContext.mimic.future_anchor_rot,
            "mimic_ref_anchor_pos": EnvContext.mimic.future_anchor_pos,
            "mimic_ref_dof_vel": EnvContext.mimic.future_dof_vel,
            "mimic_ref_dof_pos": EnvContext.mimic.future_dof_pos,
            "mimic_ref_anchor_vel": EnvContext.mimic.future_anchor_vel,
            "mimic_ref_anchor_ang_vel": EnvContext.mimic.future_anchor_ang_vel,
            "current_ref_anchor_pos": EnvContext.mimic.ref_anchor_pos,
        },
        static_params={
            "include_dof_vel": include_dof_vel,
            "include_xy_offset": include_xy_offset,
            "include_height": include_height,
            "include_anchor_vel": include_anchor_vel,
            "include_anchor_ang_vel": include_anchor_ang_vel,
            "zero_xy_offset": zero_xy_offset,
            "w_last": True,
        },
    )


def mimic_deploy_target_poses_factory(
    use_noisy: bool = False,
    include_dof_vel: bool = True,
    future_steps: Optional[Union[int, List[int]]] = None,
) -> MdpComponent:
    """Factory for deployment-ready mimic target poses.

    Produces observations that only require the robot's anchor orientation (IMU)
    and reference motion data.  No position tracking needed for deployment.

    The observation contains:
    - Reference DOF positions (joint targets, frame-invariant)
    - Reference DOF velocities (optional, frame-invariant)
    - Reference body rotations in current anchor frame (6D per body)

    Args:
        use_noisy: If True, use noisy anchor rotation (for actor with DR).
        include_dof_vel: If True, include DOF velocities.
        future_steps: Steps to select from MimicControl's future buffer.
            None = use all steps.  Int N = first N steps.
            List = specific step indices (1-indexed).

    Returns:
        MdpComponent configured for deploy-ready target poses.
    """
    from protomotions.envs.obs import build_deploy_target_poses

    state = EnvContext.noisy if use_noisy else EnvContext.current

    static_params: Dict[str, Any] = {
        "include_dof_vel": include_dof_vel,
        "w_last": True,
    }
    if future_steps is not None:
        static_params["future_steps"] = future_steps

    return MdpComponent(
        compute_func=build_deploy_target_poses,
        dynamic_vars={
            "current_anchor_rot": state.anchor_rot,
            "mimic_ref_rot": EnvContext.mimic.future_rot,
            "mimic_ref_dof_pos": EnvContext.mimic.future_dof_pos,
            "mimic_ref_dof_vel": EnvContext.mimic.future_dof_vel,
        },
        static_params=static_params,
    )


def target_obs_factory() -> MdpComponent:
    """Factory for target-reaching observations."""
    from protomotions.envs.obs import compute_target_obs

    return MdpComponent(
        compute_func=compute_target_obs,
        dynamic_vars={
            "root_pos": EnvContext.current.root_pos,
            "root_rot": EnvContext.current.root_rot,
            "tar_pos": EnvContext.target.tar_pos,
        },
    )


def steering_obs_factory() -> MdpComponent:
    """Factory for steering task observations."""
    from protomotions.envs.obs import compute_steering_obs

    return MdpComponent(
        compute_func=compute_steering_obs,
        dynamic_vars={
            "root_rot": EnvContext.current.root_rot,
            "tar_dir": EnvContext.steering.tar_dir,
            "tar_speed": EnvContext.steering.tar_speed,
            "tar_face_dir": EnvContext.steering.tar_face_dir,
        },
    )


def path_obs_factory() -> MdpComponent:
    """Factory for path-following observations."""
    from protomotions.envs.obs import compute_path_obs

    return MdpComponent(
        compute_func=compute_path_obs,
        dynamic_vars={
            "root_rot": EnvContext.current.root_rot,
            "head_pos": EnvContext.path.head_pos,
            "traj_samples": EnvContext.path.traj_samples,
            "height_conditioned": EnvContext.path.height_conditioned,
        },
    )


# =============================================================================
# Reward Factories
# =============================================================================


def action_smoothness_factory(weight: float = -0.02) -> MdpComponent:
    """Factory for action smoothness reward.

    Args:
        weight: Reward weight (typically negative).

    Returns:
        MdpComponent configured for action smoothness.
    """
    from protomotions.envs.rewards import compute_action_smoothness

    return MdpComponent(
        compute_func=compute_action_smoothness,
        dynamic_vars={
            "current_processed_action": EnvContext.current_processed_action,
            "previous_processed_action": EnvContext.previous_processed_action,
        },
        static_params={"weight": weight},
    )


def gt_rew_factory(weight: float = 0.5, coefficient: float = -100.0) -> MdpComponent:
    """Factory for position tracking reward.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.

    Returns:
        MdpComponent configured for position tracking.
    """
    from protomotions.envs.rewards import compute_gt_rew

    return MdpComponent(
        compute_func=compute_gt_rew,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
        },
        static_params={"weight": weight, "coefficient": coefficient},
    )


def gr_rew_factory(weight: float = 0.3, coefficient: float = -5.0) -> MdpComponent:
    """Factory for rotation tracking reward.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.

    Returns:
        MdpComponent configured for rotation tracking.
    """
    from protomotions.envs.rewards import compute_gr_rew

    return MdpComponent(
        compute_func=compute_gr_rew,
        dynamic_vars={
            "current_rigid_body_rot": EnvContext.current.rigid_body_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
        },
        static_params={"weight": weight, "coefficient": coefficient},
    )


def gv_rew_factory(weight: float = 0.1, coefficient: float = -0.5) -> MdpComponent:
    """Factory for velocity tracking reward.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.

    Returns:
        MdpComponent configured for velocity tracking.
    """
    from protomotions.envs.rewards import compute_gv_rew

    return MdpComponent(
        compute_func=compute_gv_rew,
        dynamic_vars={
            "current_rigid_body_vel": EnvContext.current.rigid_body_vel,
            "ref_rigid_body_vel": EnvContext.mimic.ref_state.rigid_body_vel,
        },
        static_params={"weight": weight, "coefficient": coefficient},
    )


def gav_rew_factory(weight: float = 0.1, coefficient: float = -0.1) -> MdpComponent:
    """Factory for angular velocity tracking reward.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.

    Returns:
        MdpComponent configured for angular velocity tracking.
    """
    from protomotions.envs.rewards import compute_gav_rew

    return MdpComponent(
        compute_func=compute_gav_rew,
        dynamic_vars={
            "current_rigid_body_ang_vel": EnvContext.current.rigid_body_ang_vel,
            "ref_rigid_body_ang_vel": EnvContext.mimic.ref_state.rigid_body_ang_vel,
        },
        static_params={"weight": weight, "coefficient": coefficient},
    )


def rh_rew_factory(weight: float = 0.2, coefficient: float = -100.0) -> MdpComponent:
    """Factory for root height tracking reward.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.

    Returns:
        MdpComponent configured for root height tracking.
    """
    from protomotions.envs.rewards import compute_rh_rew

    return MdpComponent(
        compute_func=compute_rh_rew,
        dynamic_vars={
            "current_root_height": EnvContext.current.root_height,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
        },
        static_params={"weight": weight, "coefficient": coefficient},
    )


def gt_rel_rew_factory(
    weight: float = 0.5,
    coefficient: float = -100.0,
    body_indices=None,
) -> MdpComponent:
    """Factory for heading-local anchor-relative position tracking reward.

    Invariant to global XY translation and yaw heading; remains well-defined when
    ``realign_motion_with_humanoid_on_each_step=False``.  Use in place of
    ``gt_rew_factory`` when the reference motion is not realigned each step.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.
        body_indices: Optional list of body indices to restrict to a subset.

    Returns:
        MdpComponent configured for heading-local relative position tracking.
    """
    from protomotions.envs.rewards import compute_gt_rel_rew

    static_params: Dict[str, Any] = {"weight": weight, "coefficient": coefficient}
    if body_indices is not None:
        static_params["body_indices"] = body_indices
    return MdpComponent(
        compute_func=compute_gt_rel_rew,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params=static_params,
    )


def anchor_xy_rew_factory(
    weight: float = 0.1, coefficient: float = -20.0
) -> MdpComponent:
    """Factory for anchor XY position tracking reward.

    Analogous to ``rh_rew_factory`` but for XY coordinates.  Provides a soft
    global XY position signal when ``realign_motion_with_humanoid_on_each_step``
    is off.  The coefficient should be kept small relative to ``rh_rew_factory``
    since odometer-based XY is noisier than height.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for XY error.

    Returns:
        MdpComponent configured for anchor XY position tracking.
    """
    from protomotions.envs.rewards import compute_anchor_xy_rew

    return MdpComponent(
        compute_func=compute_anchor_xy_rew,
        dynamic_vars={
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"weight": weight, "coefficient": coefficient},
    )


def corrupted_xy_offset_factory(
    log_noise_std: float = 0.12,
    soft_threshold: float = 0.15,
) -> MdpComponent:
    """Factory for odometer-corrupted XY offset observation.

    Produces a heading-local 2D vector from the robot's current position to
    the reference anchor position, with per-episode affine corruption (scale +
    yaw bias, sampled at reset from EnvConfig.odom_scale_range /
    odom_yaw_range_deg) and per-step proportional log-space noise.

    Applied identically in simulation and on the real G1 by passing the real
    odometer reading through the same corruption parameters — eliminating the
    sim-to-real gap on this observation channel.

    See ``build_corrupted_xy_offset`` in target_poses.py for full design rationale,
    and ``data/scripts/visualize_odometer_corruption.py`` for interactive tuning.

    Args:
        log_noise_std: Std of per-step noise in log(1+mag) space (default 0.12).
        soft_threshold: Noise ramp characteristic length in metres (default 0.15).

    Returns:
        MdpComponent producing corrupted XY offset [envs, 2].
    """
    from protomotions.envs.obs import build_corrupted_xy_offset

    return MdpComponent(
        compute_func=build_corrupted_xy_offset,
        dynamic_vars={
            "current_state_anchor_pos": EnvContext.current.anchor_pos,
            "current_state_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
            "odom_scale": EnvContext.odom_scale,
            "odom_yaw_cos_sin": EnvContext.odom_yaw_cos_sin,
        },
        static_params={
            "w_last": True,
            "log_noise_std": log_noise_std,
            "soft_threshold": soft_threshold,
        },
    )


def mimic_tracking_rewards_factory(
    gt_weight: float = 0.5,
    gr_weight: float = 0.3,
    gv_weight: float = 0.1,
    gav_weight: float = 0.1,
    rh_weight: float = 0.2,
    gt_coef: float = -100.0,
    gr_coef: float = -5.0,
    gv_coef: float = -0.5,
    gav_coef: float = -0.1,
    rh_coef: float = -100.0,
) -> Dict[str, MdpComponent]:
    """Factory for standard mimic tracking reward bundle.

    Returns a dict of 5 standard tracking rewards (gt, gr, gv, gav, rh).

    Args:
        gt_weight: Position tracking weight.
        gr_weight: Rotation tracking weight.
        gv_weight: Velocity tracking weight.
        gav_weight: Angular velocity tracking weight.
        rh_weight: Root height tracking weight.
        gt_coef: Position coefficient.
        gr_coef: Rotation coefficient.
        gv_coef: Velocity coefficient.
        gav_coef: Angular velocity coefficient.
        rh_coef: Root height coefficient.

    Returns:
        Dict of MdpComponent instances for tracking rewards.
    """
    return {
        "gt_rew": gt_rew_factory(weight=gt_weight, coefficient=gt_coef),
        "gr_rew": gr_rew_factory(weight=gr_weight, coefficient=gr_coef),
        "gv_rew": gv_rew_factory(weight=gv_weight, coefficient=gv_coef),
        "gav_rew": gav_rew_factory(weight=gav_weight, coefficient=gav_coef),
        "rh_rew": rh_rew_factory(weight=rh_weight, coefficient=rh_coef),
    }


def pow_rew_factory(
    weight: float = -1e-5,
    min_value: Optional[float] = -0.5,
    use_torque_squared: bool = False,
) -> MdpComponent:
    """Factory for power consumption reward.

    Args:
        weight: Reward weight (typically negative).
        min_value: Optional minimum clamp value.
        use_torque_squared: If True, use torque squared instead of absolute.

    Returns:
        MdpComponent configured for power consumption.
    """
    from protomotions.envs.rewards import compute_pow_rew

    static_params = {"weight": weight, "use_torque_squared": use_torque_squared}
    if min_value is not None:
        static_params["min_value"] = min_value

    return MdpComponent(
        compute_func=compute_pow_rew,
        dynamic_vars={
            "dof_forces": EnvContext.current.dof_forces,
            "dof_vel": EnvContext.current.dof_vel,
        },
        static_params=static_params,
    )


def contact_match_rew_factory(
    weight: float = -0.1,
    zero_during_grace_period: bool = True,
) -> MdpComponent:
    """Factory for contact matching reward.

    Args:
        weight: Reward weight (typically negative).
        zero_during_grace_period: If True, zero reward during grace period.

    Returns:
        MdpComponent configured for contact matching.
    """
    from protomotions.envs.rewards import compute_contact_match_rew

    return MdpComponent(
        compute_func=compute_contact_match_rew,
        dynamic_vars={
            "sim_contacts": EnvContext.current.rigid_body_contacts,
            "ref_contacts": EnvContext.mimic.ref_state.rigid_body_contacts,
            "contact_body_ids": EnvContext.contact_body_ids,
        },
        static_params={
            "weight": weight,
            "zero_during_grace_period": zero_during_grace_period,
        },
    )


def contact_force_change_rew_factory(
    weight: float = -1e-5,
    min_value: Optional[float] = -0.5,
    threshold: float = 30.0,
    zero_during_grace_period: bool = True,
) -> MdpComponent:
    """Factory for contact force change reward.

    Args:
        weight: Reward weight (typically negative).
        min_value: Optional minimum clamp value.
        threshold: Force change threshold below which changes are ignored.
        zero_during_grace_period: If True, zero reward during grace period.

    Returns:
        MdpComponent configured for contact force change penalty.
    """
    from protomotions.envs.rewards import compute_contact_force_change_rew

    static_params = {
        "weight": weight,
        "threshold": threshold,
        "zero_during_grace_period": zero_during_grace_period,
    }
    if min_value is not None:
        static_params["min_value"] = min_value

    return MdpComponent(
        compute_func=compute_contact_force_change_rew,
        dynamic_vars={
            "current_contact_force_magnitudes": EnvContext.current_contact_force_magnitudes,
            "prev_contact_force_magnitudes": EnvContext.prev_contact_force_magnitudes,
        },
        static_params=static_params,
    )


def target_reward_factory(
    weight: float = 1.0, pos_err_scale: float = 0.42
) -> MdpComponent:
    """Factory for target-reaching reward."""
    from protomotions.envs.rewards import compute_target_rew

    return MdpComponent(
        compute_func=compute_target_rew,
        dynamic_vars={
            "root_pos": EnvContext.current.root_pos,
            "tar_pos": EnvContext.target.tar_pos,
            "tar_proximity_threshold": EnvContext.target.tar_proximity_threshold,
        },
        static_params={"weight": weight, "pos_err_scale": pos_err_scale},
    )


def steering_reward_factory(weight: float = 1.0) -> MdpComponent:
    """Factory for heading and velocity steering reward."""
    from protomotions.envs.rewards import compute_heading_velocity_rew

    return MdpComponent(
        compute_func=compute_heading_velocity_rew,
        dynamic_vars={
            "root_pos": EnvContext.current.root_pos,
            "prev_root_pos": EnvContext.steering.prev_root_pos,
            "root_rot": EnvContext.current.root_rot,
            "tar_dir": EnvContext.steering.tar_dir,
            "tar_speed": EnvContext.steering.tar_speed,
            "tar_face_dir": EnvContext.steering.tar_face_dir,
            "dt": EnvContext.dt,
        },
        static_params={"weight": weight},
    )


def path_following_reward_factory(
    weight: float = 1.0,
    pos_err_scale: float = 2.0,
    height_err_scale: float = 10.0,
) -> MdpComponent:
    """Factory for path-following reward."""
    from protomotions.envs.rewards import compute_path_following_rew

    return MdpComponent(
        compute_func=compute_path_following_rew,
        dynamic_vars={
            "head_pos": EnvContext.path.head_pos,
            "tar_pos": EnvContext.path.tar_pos,
            "height_conditioned": EnvContext.path.height_conditioned,
        },
        static_params={
            "weight": weight,
            "pos_err_scale": pos_err_scale,
            "height_err_scale": height_err_scale,
        },
    )


# =============================================================================
# Termination Factories
# =============================================================================


def tracking_error_term_factory(threshold: float = 0.5) -> MdpComponent:
    """Factory for tracking error termination.

    Args:
        threshold: Maximum joint error threshold in meters.

    Returns:
        MdpComponent configured for tracking error termination.
    """
    from protomotions.envs.terminations import compute_tracking_error

    return MdpComponent(
        compute_func=compute_tracking_error,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
        },
        static_params={"threshold": threshold},
    )


def fall_termination_factory(termination_height: float = 0.15) -> MdpComponent:
    """Factory for standard fall termination."""
    from protomotions.envs.terminations import fall_termination

    return MdpComponent(
        compute_func=fall_termination,
        dynamic_vars={
            "rigid_body_pos": EnvContext.current.rigid_body_pos,
            "rigid_body_contacts": EnvContext.current.rigid_body_contacts,
            "ground_heights": EnvContext.ground_heights,
            "non_termination_contact_body_ids": (
                EnvContext.non_termination_contact_body_ids
            ),
            "progress_buf": EnvContext.progress_buf,
        },
        static_params={"termination_height": termination_height},
    )


# =============================================================================
# BeyondMimic Reward Factories
# =============================================================================


def global_anchor_pos_rew_factory(
    weight: float = 0.5, sigma: float = 0.3
) -> MdpComponent:
    """Factory for global anchor position reward (BeyondMimic).

    Args:
        weight: Reward weight.
        sigma: Gaussian kernel width.

    Returns:
        MdpComponent configured for global anchor position reward.
    """
    from protomotions.envs.rewards import compute_global_anchor_pos_rew

    return MdpComponent(
        compute_func=compute_global_anchor_pos_rew,
        dynamic_vars={
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"weight": weight, "sigma": sigma},
    )


def global_anchor_ori_rew_factory(
    weight: float = 0.5, sigma: float = 0.4
) -> MdpComponent:
    """Factory for global anchor orientation reward (BeyondMimic).

    Args:
        weight: Reward weight.
        sigma: Gaussian kernel width.

    Returns:
        MdpComponent configured for global anchor orientation reward.
    """
    from protomotions.envs.rewards import compute_global_anchor_ori_rew

    return MdpComponent(
        compute_func=compute_global_anchor_ori_rew,
        dynamic_vars={
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"weight": weight, "sigma": sigma},
    )


def relative_body_pos_rew_factory(
    weight: float = 1.0,
    sigma: float = 0.3,
) -> MdpComponent:
    """Factory for relative body position reward (BeyondMimic).

    Args:
        weight: Reward weight.
        sigma: Gaussian kernel width.

    Returns:
        MdpComponent configured for relative body position reward.
    """
    from protomotions.envs.rewards import compute_relative_body_pos_rew

    return MdpComponent(
        compute_func=compute_relative_body_pos_rew,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"weight": weight, "sigma": sigma},
    )


def relative_body_ori_rew_factory(
    weight: float = 1.0,
    sigma: float = 0.4,
) -> MdpComponent:
    """Factory for relative body orientation reward (BeyondMimic).

    Args:
        weight: Reward weight.
        sigma: Gaussian kernel width.

    Returns:
        MdpComponent configured for relative body orientation reward.
    """
    from protomotions.envs.rewards import compute_relative_body_ori_rew

    return MdpComponent(
        compute_func=compute_relative_body_ori_rew,
        dynamic_vars={
            "current_rigid_body_rot": EnvContext.current.rigid_body_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"weight": weight, "sigma": sigma},
    )


def global_body_lin_vel_rew_factory(
    weight: float = 1.0,
    sigma: float = 1.0,
) -> MdpComponent:
    """Factory for global body linear velocity reward (BeyondMimic).

    Args:
        weight: Reward weight.
        sigma: Gaussian kernel width.

    Returns:
        MdpComponent configured for body linear velocity reward.
    """
    from protomotions.envs.rewards import compute_global_body_lin_vel_rew

    return MdpComponent(
        compute_func=compute_global_body_lin_vel_rew,
        dynamic_vars={
            "current_rigid_body_vel": EnvContext.current.rigid_body_vel,
            "ref_rigid_body_vel": EnvContext.mimic.ref_state.rigid_body_vel,
        },
        static_params={"weight": weight, "sigma": sigma},
    )


def global_body_ang_vel_rew_factory(
    weight: float = 1.0,
    sigma: float = 3.14,
) -> MdpComponent:
    """Factory for global body angular velocity reward (BeyondMimic).

    Args:
        weight: Reward weight.
        sigma: Gaussian kernel width.

    Returns:
        MdpComponent configured for body angular velocity reward.
    """
    from protomotions.envs.rewards import compute_global_body_ang_vel_rew

    return MdpComponent(
        compute_func=compute_global_body_ang_vel_rew,
        dynamic_vars={
            "current_rigid_body_ang_vel": EnvContext.current.rigid_body_ang_vel,
            "ref_rigid_body_ang_vel": EnvContext.mimic.ref_state.rigid_body_ang_vel,
        },
        static_params={"weight": weight, "sigma": sigma},
    )


# =============================================================================
# BeyondMimic Termination Factories
# =============================================================================


def anchor_pos_error_term_factory(threshold: float = 0.5) -> MdpComponent:
    """Factory for anchor position error termination (BeyondMimic).

    Args:
        threshold: Maximum allowed distance in meters.

    Returns:
        MdpComponent configured for anchor position error termination.
    """
    from protomotions.envs.terminations import compute_anchor_pos_error_term

    return MdpComponent(
        compute_func=compute_anchor_pos_error_term,
        dynamic_vars={
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"threshold": threshold},
    )


def anchor_ori_error_term_factory(threshold: float = 0.8) -> MdpComponent:
    """Factory for anchor orientation error termination (BeyondMimic).

    Args:
        threshold: Maximum allowed difference in projected gravity z-component.

    Returns:
        MdpComponent configured for anchor orientation error termination.
    """
    from protomotions.envs.terminations import compute_anchor_ori_error_term

    return MdpComponent(
        compute_func=compute_anchor_ori_error_term,
        dynamic_vars={
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"threshold": threshold},
    )


def relative_body_pos_error_term_factory(threshold: float = 0.25) -> MdpComponent:
    """Factory for relative body position error termination (BeyondMimic).

    Args:
        threshold: Maximum allowed error for any body in meters.

    Returns:
        MdpComponent configured for relative body position error termination.
    """
    from protomotions.envs.terminations import compute_relative_body_pos_error_term

    return MdpComponent(
        compute_func=compute_relative_body_pos_error_term,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"threshold": threshold},
    )


def anchor_height_error_term_factory(threshold: float = 0.25) -> MdpComponent:
    """Factory for anchor height error termination.

    Terminates when root height deviates from reference by more than threshold.

    Args:
        threshold: Maximum allowed height deviation in meters.

    Returns:
        MdpComponent configured for anchor height error termination.
    """
    from protomotions.envs.terminations import compute_anchor_height_error_term

    return MdpComponent(
        compute_func=compute_anchor_height_error_term,
        dynamic_vars={
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"threshold": threshold},
    )


# =============================================================================
# Evaluation Metric Factories
# =============================================================================


def gt_error_factory(threshold: float = None) -> MdpComponent:
    """Factory for mean body position error metric.

    Args:
        threshold: If set, fail when mean error > threshold.

    Returns:
        MdpComponent configured for mean body position error evaluation.
    """
    from protomotions.envs.terminations import mean_body_pos_error

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=mean_body_pos_error,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
        },
        static_params=static_params,
    )


def max_joint_error_factory(threshold: float = None) -> MdpComponent:
    """Factory for max body position error metric.

    Args:
        threshold: If set, fail when max error > threshold.

    Returns:
        MdpComponent configured for max body position error evaluation.
    """
    from protomotions.envs.terminations import max_body_pos_error

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=max_body_pos_error,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
        },
        static_params=static_params,
    )


def gr_error_factory(threshold: float = None) -> MdpComponent:
    """Factory for mean body rotation error metric.

    Args:
        threshold: If set, fail when mean error > threshold (radians).

    Returns:
        MdpComponent configured for mean body rotation error evaluation.
    """
    from protomotions.envs.terminations import mean_body_rot_error

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=mean_body_rot_error,
        dynamic_vars={
            "current_rigid_body_rot": EnvContext.current.rigid_body_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
        },
        static_params=static_params,
    )


def anchor_pos_metric_factory(threshold: float = None) -> MdpComponent:
    """Factory for anchor position error metric.

    Args:
        threshold: If set, fail when error > threshold.

    Returns:
        MdpComponent configured for anchor position error evaluation.
    """
    from protomotions.envs.terminations import anchor_pos_error_value

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=anchor_pos_error_value,
        dynamic_vars={
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params=static_params,
    )


def anchor_ori_metric_factory(threshold: float = None) -> MdpComponent:
    """Factory for anchor orientation error metric.

    Args:
        threshold: If set, fail when error > threshold.

    Returns:
        MdpComponent configured for anchor orientation error evaluation.
    """
    from protomotions.envs.terminations import anchor_ori_error_value

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=anchor_ori_error_value,
        dynamic_vars={
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params=static_params,
    )


def relative_body_pos_metric_factory(threshold: float = None) -> MdpComponent:
    """Factory for max relative body position error metric.

    Args:
        threshold: If set, fail when max error > threshold.

    Returns:
        MdpComponent configured for relative body position error evaluation.
    """
    from protomotions.envs.terminations import relative_body_pos_max_error

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=relative_body_pos_max_error,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params=static_params,
    )


def anchor_height_error_metric_factory(threshold: float = None) -> MdpComponent:
    """Factory for anchor height error metric.

    Args:
        threshold: If set, fail when height error > threshold.

    Returns:
        MdpComponent configured for anchor height error evaluation.
    """
    from protomotions.envs.terminations import anchor_height_error_value

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=anchor_height_error_value,
        dynamic_vars={
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params=static_params,
    )


def _check_path_distance_term_wrapper(**kwargs):
    """Picklable wrapper around jit-scripted check_path_distance_term."""
    from protomotions.envs.terminations import check_path_distance_term

    return check_path_distance_term(**kwargs)


def _check_steering_velocity_error_wrapper(**kwargs):
    """Picklable wrapper around jit-scripted check_steering_velocity_error."""
    from protomotions.envs.terminations import check_steering_velocity_error

    return check_steering_velocity_error(**kwargs)


def path_distance_error_factory(
    threshold: float = 1.0,
    min_progress: int = 10,
) -> MdpComponent:
    """Factory for path distance evaluation metric.

    Returns a boolean-valued component: True when agent is too far from path.
    Use threshold=0.5 with fail_above=True to convert to failure flag.

    Args:
        threshold: Maximum distance from path (meters).
        min_progress: Minimum steps before checking.

    Returns:
        MdpComponent configured for path distance evaluation.
    """
    return MdpComponent(
        compute_func=_check_path_distance_term_wrapper,
        dynamic_vars={
            "head_pos": EnvContext.path.head_pos,
            "target_pos": EnvContext.path.tar_pos,
            "progress_buf": EnvContext.path.progress_buf,
        },
        static_params={
            "fail_dist": threshold,
            "min_progress": min_progress,
            "threshold": 0.5,  # Boolean True (1.0) > 0.5 → fail
        },
    )


def steering_velocity_error_factory(
    speed_tolerance: float = 0.5,
    direction_tolerance: float = 0.7,
) -> MdpComponent:
    """Factory for steering velocity evaluation metric.

    Returns a boolean-valued component: True when velocity deviates too much.
    Use threshold=0.5 with fail_above=True to convert to failure flag.

    Args:
        speed_tolerance: Acceptable speed difference from target (m/s).
        direction_tolerance: Minimum dot product with target direction (0-1).

    Returns:
        MdpComponent configured for steering velocity evaluation.
    """
    return MdpComponent(
        compute_func=_check_steering_velocity_error_wrapper,
        dynamic_vars={
            "root_pos": EnvContext.current.root_pos,
            "prev_root_pos": EnvContext.steering.prev_root_pos,
            "tar_dir": EnvContext.steering.tar_dir,
            "tar_speed": EnvContext.steering.tar_speed,
            "dt": EnvContext.dt,
        },
        static_params={
            "speed_tolerance": speed_tolerance,
            "direction_tolerance": direction_tolerance,
            "threshold": 0.5,  # Boolean True (1.0) > 0.5 → fail
        },
    )


__all__ = [
    # Observation factories
    "max_coords_obs_factory",
    "reduced_coords_obs_factory",
    "historical_max_coords_obs_factory",
    "historical_reduced_coords_obs_factory",
    "previous_actions_factory",
    "nearest_surface_obs_factory",
    "mimic_target_poses_max_coords_factory",
    "mimic_target_poses_future_rel_factory",
    "mimic_target_poses_reduced_coords_factory",
    "mimic_deploy_target_poses_factory",
    "target_obs_factory",
    "steering_obs_factory",
    "path_obs_factory",
    # Reward factories
    "action_smoothness_factory",
    "gt_rew_factory",
    "gr_rew_factory",
    "gv_rew_factory",
    "gav_rew_factory",
    "rh_rew_factory",
    "gt_rel_rew_factory",
    "anchor_xy_rew_factory",
    "mimic_tracking_rewards_factory",
    # Odometer observation factory
    "corrupted_xy_offset_factory",
    "pow_rew_factory",
    "contact_match_rew_factory",
    "contact_force_change_rew_factory",
    "target_reward_factory",
    "steering_reward_factory",
    "path_following_reward_factory",
    # BeyondMimic reward factories
    "global_anchor_pos_rew_factory",
    "global_anchor_ori_rew_factory",
    "relative_body_pos_rew_factory",
    "relative_body_ori_rew_factory",
    "global_body_lin_vel_rew_factory",
    "global_body_ang_vel_rew_factory",
    # Termination factories
    "tracking_error_term_factory",
    "anchor_pos_error_term_factory",
    "anchor_ori_error_term_factory",
    "relative_body_pos_error_term_factory",
    "anchor_height_error_term_factory",
    "fall_termination_factory",
    # Evaluation metric factories
    "anchor_height_error_metric_factory",
    "gt_error_factory",
    "max_joint_error_factory",
    "gr_error_factory",
    "anchor_pos_metric_factory",
    "anchor_ori_metric_factory",
    "relative_body_pos_metric_factory",
    "path_distance_error_factory",
    "steering_velocity_error_factory",
]
