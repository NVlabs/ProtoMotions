# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracker contracts without simulator startup or LFS assets."""

from types import SimpleNamespace

import pytest
import torch
from protomotions.envs.context_requirements import bound_components
from protomotions.envs.obs.state_history_buffer import history_depth

from examples.experiments.metatron import g1_teacher as recipe
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.motion_lib import MotionFileSwitchMode
from protomotions.components.pose_lib import ControlInfo
from protomotions.envs.action import make_bm_pd_action_config
from protomotions.envs.control.mimic_control import MimicControl, MimicControlConfig
from protomotions.envs.obs.target_poses import (
    build_max_coords_target_poses,
    build_target_dof_pos,
)
from protomotions.envs.context_views import EnvContext
from protomotions.envs.rewards.regularization import compute_action_rate_l2
from protomotions.tests.support.fake_reference import fake_reference
from protomotions.tests.support.task_inputs import make_robot, make_task_inputs


def robot_config():
    robot = SimpleNamespace(
        number_of_actions=2,
        anchor_body_index=1,
        control_hz=50,
        default_dof_pos=torch.zeros(2),
        asset=SimpleNamespace(self_collisions=False),
        kinematic_info=SimpleNamespace(
            body_names=[
                "root",
                "torso",
                "left_hand",
                "right_hand",
                "left_foot",
                "right_foot",
            ],
            dof_names=["a", "b"],
            num_dofs=2,
            hinge_axes_map={
                1: torch.tensor([[1.0, 0.0, 0.0]]),
                2: torch.tensor([[0.0, 1.0, 0.0]]),
            },
            dof_limits_lower=torch.tensor([-1.0, -2.0]),
            dof_limits_upper=torch.tensor([1.0, 2.0]),
        ),
        common_naming_to_robot_body_names={
            "torso_body_name": ["torso"],
            "all_left_hand_bodies": ["left_hand"],
            "all_right_hand_bodies": ["right_hand"],
            "all_left_foot_bodies": ["left_foot"],
            "all_right_foot_bodies": ["right_foot"],
            "all_wrist_bodies": [],
        },
        control=SimpleNamespace(
            soft_pos_limit=0.9,
            control_info={
                "a": ControlInfo(stiffness=10.0, damping=1.0, effort_limit=30.0),
                "b": ControlInfo(stiffness=20.0, damping=2.0, effort_limit=40.0),
            },
        ),
        simulation_params=SimpleNamespace(
            isaaclab=SimpleNamespace(fps=200, decimation=4)
        ),
    )
    robot.update_fields = lambda **kwargs: robot.__dict__.update(kwargs)
    from protomotions.robot_configs.base import RobotConfig

    robot.body_indices = lambda *names: RobotConfig.body_indices(robot, *names)
    return robot


def test_recipe_contract_and_inference_overrides():
    robot = robot_config()
    args = SimpleNamespace(
        simulator="isaaclab",
        motion_file="motions.pt",
        scenes_file=None,
        batch_size=32,
        training_max_steps=64,
    )
    simulator = SimpleNamespace()
    recipe.configure_robot_and_simulator(robot, simulator, args)
    env = recipe.env_config(robot, args)
    agent = recipe.agent_config(robot, env, args)
    assert type(agent) is PPOAgentConfig
    assert env.sensors is None
    assert history_depth(bound_components(env, robot)) == 1
    assert type(env.control_components["mimic"]) is MimicControlConfig
    assert env.control_components["mimic"].reference_height_smoothing_seconds == 0.2
    assert env.control_components["mimic"].future_steps == (1, 2, 5, 7, 12, 18, 25)
    assert env.observation_components["mimic_target_poses"].static_params[
        "body_indices"
    ] == [1, 2, 3, 4, 5]
    assert agent.model.actor.mu_model.encoder.in_keys == [
        "mimic_target_poses",
        "target_dof_pos",
    ]
    assert "target_dof_pos" not in agent.model.actor.mu_model.decoder.in_keys
    assert agent.model.actor.mu_model.num_fsq_levels == 33
    assert agent.model.actor.mu_model.num_fsq_scalars == 64
    assert [layer.units for layer in agent.model.critic.layers] == list(
        recipe.DECODER_CRITIC_WIDTHS
    )
    assert agent.model.actor.actor_logstd == -2.9
    assert agent.gradient_clip_val == 1.0
    assert not agent.normalize_rewards
    assert agent.task_reward_w == 0.8
    assert agent.num_mini_epochs == 2
    assert robot.asset.self_collisions
    assert robot.contact_bodies == "all"
    partners = [f"robot/{name}" for name in robot.kinematic_info.body_names]
    assert robot.contact_partners == ["terrain"] + partners
    self_contact = env.reward_components["self_contacts"]
    assert self_contact.static_params["body_names"] == ["root", "torso"]
    assert self_contact.static_params["partner_names"] == partners
    assert simulator.domain_randomization.body_mass.body_names == [
        "left_hand",
        "right_hand",
    ]
    baseline = make_bm_pd_action_config(robot)
    torch.testing.assert_close(
        env.action_config.params["action_scale"], torch.tensor([0.75, 0.5])
    )
    torch.testing.assert_close(
        env.action_config.params["stiffness"], baseline.params["stiffness"]
    )
    torch.testing.assert_close(
        env.action_config.params["damping"], baseline.params["damping"]
    )
    action_rate = env.reward_components["action_smoothness"]
    assert action_rate.compute_func is compute_action_rate_l2
    assert action_rate.dynamic_vars == {
        "raw_action": EnvContext.current_action,
        "previous_raw_action": EnvContext.previous_action,
    }
    assert action_rate.static_params == {}
    assert action_rate.meta.weight == -0.1
    assert (
        "threshold" not in agent.evaluator.metric_components["gt_error"].static_params
    )
    assert agent.evaluator.failure_components[
        "tracking_failure_ratio"
    ].component.static_params["upright_mean"]
    assert agent.evaluator.failure_components["tracking_failure_ratio"].threshold == 1.0
    assert set(agent.evaluator.failure_components) == {
        "tracking_failure_ratio",
        "foot_position",
        "foot_height",
    }
    assert agent.evaluator.score_key == "eval/success_rate"
    agent.evaluator.validate()
    assert (
        "upright_mean"
        not in env.termination_components["tracking_error"].component.static_params
    )
    terrain = recipe.terrain_config(args)
    motion = recipe.motion_lib_config(args)
    assert motion.motion_file_switch_mode is MotionFileSwitchMode.FIXED
    args.motion_file = "motions_slurmrank.pt"
    assert (
        recipe.motion_lib_config(args).motion_file_switch_mode
        is MotionFileSwitchMode.LIVE
    )
    from protomotions.utils.experiment_configs import ExperimentConfigs

    recipe.apply_inference_overrides(
        ExperimentConfigs(
            robot=robot,
            simulator=simulator,
            terrain=terrain,
            scene_lib=None,
            motion_lib=motion,
            env=env,
            agent=agent,
        ),
        args,
    )
    assert not env.termination_components
    assert simulator.domain_randomization is None


@pytest.mark.parametrize(
    "efforts,stiffnesses", [([30.0, 40.0], [10.0, 20.0]), ([80.0, 12.0], [20.0, 6.0])]
)
def test_action_scale_is_quarter_effort_over_stiffness(efforts, stiffnesses):
    robot = robot_config()
    for name, effort, stiffness in zip(
        robot.kinematic_info.dof_names, efforts, stiffnesses
    ):
        robot.control.control_info[name].effort_limit = effort
        robot.control.control_info[name].stiffness = stiffness
    args = SimpleNamespace(simulator="isaaclab")
    recipe.configure_robot_and_simulator(robot, SimpleNamespace(), args)
    env = recipe.env_config(robot, args)
    # Derive the expected physical scale without calling the production helper.
    expected = 0.25 * torch.tensor(efforts) / torch.tensor(stiffnesses)
    torch.testing.assert_close(env.action_config.params["action_scale"], expected)


@pytest.mark.parametrize("control_hz", [50, 30, 15])
def test_future_target_horizon(control_hz):
    robot = robot_config()
    robot.control_hz = control_hz
    steps = recipe._future_steps(robot, SimpleNamespace(simulator="isaaclab"))
    assert len(steps) == len(set(steps)) == 7
    assert steps == sorted(steps)
    # The last horizon step is ~0.5 s of control steps at the policy rate.
    assert abs(steps[-1] / control_hz - 0.5) <= 1.0 / control_hz


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("velocities", [False, True])
def test_body_subset_keeps_original_anchor_and_all_target_blocks(relative, velocities):
    torch.manual_seed(5)
    n, t, b = 2, 3, 6
    current = torch.randn(n, b, 3)
    reference = torch.randn(n, t, b, 3)
    q = torch.nn.functional.normalize(torch.randn(n, b, 4), dim=-1)
    rq = torch.nn.functional.normalize(torch.randn(n, t, b, 4), dim=-1)
    kwargs = dict(
        current_state_body_pos=current,
        current_state_body_rot=q,
        current_state_body_vel=current * 0.3,
        current_state_body_ang_vel=current * 0.7,
        mimic_ref_pos=reference,
        mimic_ref_rot=rq,
        mimic_ref_vel=reference * 0.2,
        mimic_ref_ang_vel=reference * 0.5,
        anchor_body_index=4,
        with_relative=relative,
        with_velocities=velocities,
        w_last=True,
    )
    full = build_max_coords_target_poses(**kwargs).reshape(n, t, -1)
    selected = [5, 0, 2]  # Anchor excluded, order intentionally permuted.
    widths = ([3, 3, 6, 6] if relative else [3, 6]) + ([3, 3] if velocities else [])
    blocks = full.split([b * width for width in widths], dim=-1)
    expected = torch.cat(
        [
            block.reshape(n, t, b, width)[:, :, selected].flatten(2)
            for block, width in zip(blocks, widths)
        ],
        dim=-1,
    ).flatten(1)
    actual = build_max_coords_target_poses(**kwargs, body_indices=selected)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        build_max_coords_target_poses(**kwargs, body_indices=list(range(b))),
        full.flatten(1),
    )
    for indices in ([], [-1], [b]):
        with pytest.raises(ValueError, match="body_indices"):
            build_max_coords_target_poses(**kwargs, body_indices=indices)


def test_joint_target_time_order_and_raw_action_rate():
    angles = torch.arange(24.0).reshape(2, 4, 3)
    torch.testing.assert_close(
        build_target_dof_pos(angles, [1, 3]), angles[:, [0, 2]].flatten(1)
    )
    torch.testing.assert_close(
        compute_action_rate_l2(
            torch.tensor([[1.0, -2.0], [3.0, 4.0]]),
            torch.tensor([[0.0, 1.0], [3.0, 4.0]]),
        ),
        torch.tensor([10.0, 0.0]),
    )


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.1])
def test_reference_height_is_causal_stateless_and_clip_local(dt):
    queries = []

    def positions(ids, times):
        queries.append((ids.clone(), times.clone()))
        pos = torch.zeros(len(ids), 2, 3)
        pos[:, 1, 2] = ids + times  # Non-root anchor; each clip has a unique height.
        return pos

    def terrain_offset(pos, env_ids):
        offset = torch.zeros_like(pos)
        # Pose-dependent correction proves each historical sample is corrected.
        offset[:, :, 2] = env_ids[:, None] * 10 + pos[:, 1, 2:3] * 0.1
        return offset

    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        dt=dt,
        robot_config=SimpleNamespace(anchor_body_index=1),
        motion_lib=SimpleNamespace(
            get_motion_positions=positions,
            get_motion_length=lambda ids: torch.full_like(ids, 2.0, dtype=torch.float),
        ),
        motion_manager=SimpleNamespace(
            motion_ids=torch.zeros(3, dtype=torch.long), motion_times=torch.zeros(3)
        ),
    )
    env.reference = fake_reference(env, placement_offset=terrain_offset)
    control = MimicControl(
        MimicControlConfig(reference_height_smoothing_seconds=0.2),
        make_task_inputs(
            3,
            robot=make_robot(("root", "anchor"), anchor_body_index=1),
            reference=env.reference,
            dt=dt,
        ),
    )

    def smoothed(ids, times, env_ids=None):
        # The smoother reads the clip assignment through the query.
        env.motion_manager.motion_ids = ids.clone()
        env.motion_manager.motion_times = times.clone()
        return control._smoothed_reference_height(env_ids)

    ids = torch.tensor([2, 0, 1])
    times = torch.tensor([0.0, 0.1, 2.5])
    actual = smoothed(ids, times)
    sample_ids, sample_times = queries[-1]
    sample_times = sample_times.reshape(3, 51)
    assert torch.equal(sample_ids.reshape(3, 51), ids[:, None].expand(3, 51))
    assert (sample_times >= 0).all() and (sample_times <= 2).all()
    assert (sample_times <= times[:, None]).all()
    expected_times = (times[:, None] - torch.arange(50, -1, -1) * 0.02).clamp(0, 2)
    torch.testing.assert_close(sample_times, expected_times)
    heights = (ids[:, None] + sample_times) * 1.1 + torch.arange(3)[:, None] * 10
    expected = heights[:, 0]
    alpha = 1 - torch.exp(torch.tensor(-0.1))
    for height in heights[:, 1:].unbind(1):
        expected = expected.lerp(height, alpha)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[0], torch.tensor(2.2))

    control.advance(None)
    control.reset_rows(torch.tensor([0, 2]), None)
    smoothed(ids.flip(0), times.flip(0))
    selection = torch.tensor([2, 0])
    torch.testing.assert_close(smoothed(ids, times, selection), actual[selection])
    torch.testing.assert_close(smoothed(ids, times), actual)


@pytest.mark.parametrize("tau", [-0.1, float("nan"), float("inf")])
def test_height_filter_rejects_invalid_time_constant(tau):
    with pytest.raises(ValueError, match="reference_height_smoothing_seconds"):
        MimicControl(
            MimicControlConfig(reference_height_smoothing_seconds=tau),
            make_task_inputs(1),
        )
