# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public student recipe and modality-isolation tests; no simulator or payloads."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from protomotions.envs.context_requirements import bound_components
from protomotions.envs.obs.state_history_buffer import history_depth
from tensordict import TensorDict

from examples.experiments.metatron import g1_student as recipe
from examples.experiments.metatron import g1_teacher as tracker
from protomotions.agents.common.config import MLPWithConcatConfig, TransformerConfig
from protomotions.agents.common.common import ModuleContainer
from protomotions.agents.supervised.agent import SupervisedAgent
from protomotions.agents.supervised.config import RolloutActor
from protomotions.agents.supervised.expert_utils import validate_expert_action_contract
from protomotions.agents.supervised.masked_mimic_model import MaskedMimicModel
from protomotions.agents.supervised.masked_mimic_config import LatentGeometry
from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.envs.control.masked_mimic_tasks.goals import BodyGoal, pack_goals
from protomotions.utils import rotations
from protomotions.robot_configs.g1 import G1RobotConfig
from protomotions.simulator.factory import simulator_config


@pytest.fixture
def configs(monkeypatch):
    args = SimpleNamespace(
        simulator="isaaclab",
        motion_file="motions.pt",
        scenes_file=None,
        batch_size=4,
        training_max_steps=64,
        expert_model_path="teacher/last.ckpt",
    )
    robot = G1RobotConfig()
    sim = simulator_config("isaaclab", robot, True, 4, "test")
    tracker.configure_robot_and_simulator(robot, sim, args)
    teacher_env = tracker.env_config(robot, args)
    teacher_agent = tracker.agent_config(robot, teacher_env, args)
    teacher = SimpleNamespace(
        robot=deepcopy(robot),
        simulator=deepcopy(sim),
        env=teacher_env,
        agent=teacher_agent,
        terrain=tracker.terrain_config(args),
    )
    monkeypatch.setattr(
        recipe, "load_experiment_configs_from_checkpoint", lambda _: teacher
    )
    recipe.configure_robot_and_simulator(robot, sim, args)
    env = recipe.env_config(robot, args)
    agent = recipe.agent_config(robot, env, args)
    return robot, sim, env, agent, args, teacher


def test_teacher_action_physics_and_future_steps_are_preserved(configs):
    robot, sim, env, agent, args, teacher = configs
    validate_expert_action_contract(teacher.robot, teacher.env, robot, env)
    assert sim.sim == teacher.simulator.sim
    assert sim.domain_randomization == teacher.simulator.domain_randomization
    assert sim.domain_randomization is not teacher.simulator.domain_randomization
    assert not hasattr(sim, "track_self_contacts")
    assert robot.contact_bodies == teacher.robot.contact_bodies
    assert robot.contact_partners == ["terrain"]
    assert robot.asset.self_collisions == teacher.robot.asset.self_collisions
    for component in env.observation_components.values():
        if component.compute_func is recipe.to_float:
            assert getattr(component.compute_func, "run_eagerly", False) is True
    assert "self_contacts" not in env.reward_components
    assert env.control_components["masked_mimic"].future_steps == (
        1,
        2,
        5,
        7,
        12,
        18,
        25,
    )
    assert history_depth(bound_components(env, robot)) == 8
    assert {
        key.removeprefix("expert_")
        for key in env.observation_components
        if key.startswith("expert_")
    } == set(teacher.agent.model.actor.in_keys)
    assert agent.rollout_actor is RolloutActor.STUDENT
    assert agent.expert_model_path == "teacher/last.ckpt"
    assert agent.expert_rollout_probability == 0.1
    assert agent.model.vae.latent_geometry is LatentGeometry.SPHERE
    assert agent.model.vae.vae_latent_dim == 64
    assert agent.model.prior.out_keys == ["prior_mu", "prior_logvar"]
    assert agent.model.encoder.in_keys == ["max_coords_obs", "mimic_target_poses"]
    assert agent.model.trunk.in_keys == [*recipe.STATE_KEYS, "vae_latent"]
    assert agent.model.trunk.out_keys == ["actor_trunk_out", "solution_features"]
    assert [
        (x.key, x.coefficient, x.prediction_key, x.forward)
        for x in agent.model.temporal_losses
    ] == [
        ("privileged_action", 0.05, "privileged_action", False),
        ("solution_features", 0.001, "solution_features", False),
    ]
    transformer = next(
        x for x in agent.model.prior.models if isinstance(x, TransformerConfig)
    )
    assert (
        transformer.latent_dim,
        transformer.num_layers,
        transformer.num_heads,
        transformer.ff_size,
    ) == (1024, 6, 4, 4096)
    assert transformer.input_and_mask_mapping == {
        "global_tokens": "masked_mimic_target_poses_masks",
        "local_tokens": "masked_mimic_local_target_poses_masks",
        "style_token": "masked_mimic_style_frame_masks",
        "velocity_tokens": "masked_mimic_velocity_command_mask",
    }
    assert agent.model.prior.in_keys == [
        "fk_max_coords_obs",
        "reduced_coords_obs",
        "last_action",
        "historical_proprioception",
        "fk_masked_mimic_target_poses",
        "masked_mimic_target_masks",
        "masked_mimic_target_times",
        "masked_mimic_local_target_poses",
        "masked_mimic_local_target_masks",
        "masked_mimic_local_target_times",
        "masked_mimic_style_poses",
        "masked_mimic_style_masks",
        "masked_mimic_style_velocities",
        "masked_mimic_velocity_command",
        "masked_mimic_velocity_command_mask",
        "masked_mimic_style_frame_masks",
        "masked_mimic_target_poses_masks",
        "masked_mimic_local_target_poses_masks",
    ]
    assert not any(key.startswith("expert_") for key in agent.model.prior.in_keys)
    assert recipe.motion_lib_config(args).motion_file == "motions.pt"


def test_expert_history_grows_the_ring_through_the_merged_bindings():
    """The ring sizes itself from the bindings: merging the expert's history
    observation into the student's table makes the ring as deep as the deeper
    read; nothing is configured by hand (the old ``_required_history`` is gone).
    """
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_reduced_coords_from_state,
    )
    from protomotions.envs.obs.state_history_buffer import history_depth
    from protomotions.envs.rewards.meta import RewardMeta
    from protomotions.envs.rewards.regularization import compute_action_rate_l2

    def history(steps: int) -> MdpComponent:
        return MdpComponent.from_kernel(
            compute_historical_reduced_coords_from_state, history_steps=steps
        )

    student = {  # previous_action: one past slot
        "rate": MdpComponent.from_kernel(
            compute_action_rate_l2, meta=RewardMeta(weight=-0.1)
        )
    }
    expert = {"expert_history": history(12)}
    assert history_depth(student) == 1
    assert history_depth({**student, **expert}) == 12
    deeper = {"deep": history(20)}
    assert history_depth({**deeper, **expert}) == 20


def test_inference_does_not_open_teacher_or_require_privileged_inputs(
    configs, monkeypatch
):
    robot, sim, env, agent, args, _ = configs

    def forbidden(_):
        raise AssertionError("Inference opened teacher")

    monkeypatch.setattr(recipe, "load_experiment_configs_from_checkpoint", forbidden)
    from protomotions.utils.experiment_configs import ExperimentConfigs

    configs = ExperimentConfigs(
        robot=robot,
        simulator=sim,
        terrain=None,
        scene_lib=None,
        motion_lib=None,
        env=env,
        agent=agent,
    )
    # Derivation applies the agent's framework defaults (the expert cleanup)
    # before the recipe hook (which drops the goal observations).
    agent.apply_inference_defaults(env)
    recipe.apply_inference_overrides(configs, args)
    assert agent.expert_model_path is None
    assert agent.rollout_actor is RolloutActor.STUDENT
    assert set(agent.model.prior.in_keys) <= set(env.observation_components)
    assert "max_coords_obs" not in env.observation_components
    assert not any(key.startswith("expert_") for key in env.observation_components)


def _small_model(config):
    config = deepcopy(config)
    for container in (config.prior, config.encoder, config.trunk):
        for module in container.models:
            if isinstance(module, MLPWithConcatConfig):
                for layer in module.layers:
                    layer.units = 16
                if module.out_keys[0] in (
                    "current_token", "history_tokens", "global_tokens", "local_tokens",
                    "style_tokens", "velocity_tokens",
                ):
                    module.num_out = 16
            elif isinstance(module, TransformerConfig):
                module.latent_dim = module.transformer_token_size = 16
                module.num_layers = 1
                module.ff_size = 32
    model = MaskedMimicModel(config)
    model.reset_rollout_context(num_envs=2, device="cpu")
    return model


# Proprioception and the privileged posterior inputs are not goal observations;
# their widths only need to be stable for lazy materialization.
SYNTHETIC_WIDTHS = {
    "fk_max_coords_obs": 45,
    "reduced_coords_obs": 61,
    "last_action": 29,
    "historical_proprioception": 680,
    "max_coords_obs": 32,
    "mimic_target_poses": 64,
}


def _goal_context(robot, control):
    """Pack a steering prompt (env 0) and a hidden one (env 1) with a reach goal."""
    from protomotions.envs.base_env.publishers import prefix_names
    from protomotions.envs.sensors.config import (
        IMUSensor,
        JointEncoderSensor,
        LocalizationSensor,
        NoNoise,
        SensorConfig,
    )
    from protomotions.envs.state_names import body_state_names
    from protomotions.envs.sensors.estimation import estimate_body_state
    from protomotions.envs.sensors.measurement import SensorReadings
    from protomotions.utils.baked_fk import BakedAnchorFK

    names = robot.kinematic_info.body_names
    ids = [names.index(name) for name in robot.trackable_bodies_subset]
    anchor = robot.anchor_body_index
    clock = torch.tensor([1.0, 2.0])
    heading = torch.tensor([0.0, torch.pi / 2])
    axis = torch.tensor([[0.0, 0.0, 1.0]]).expand(2, -1)
    rotation = rotations.quat_from_angle_axis(heading, axis, True)
    slot = control.num_masked_future_steps - 1
    goals = [
        BodyGoal(
            anchor,
            slot,
            clock + 1,
            torch.ones(2, dtype=torch.bool),
            position=torch.tensor([[1.0, 0.0, 0.8], [0.0, 1.0, 0.8]]),
            rotation=rotation,
            translation=(True, False),
            orientation=(False, True),
        ),
        BodyGoal(
            ids[1],
            slot,
            clock + 1,
            torch.tensor([True, False]),
            position=torch.tensor([[1.3, 0.2, 1.1], [0.0, 0.0, 0.0]]),
        ),
    ]
    packed = pack_goals(
        goals,
        clock=clock,
        body_count=len(names),
        conditionable_ids=ids,
        anchor_id=anchor,
        frames=control.num_masked_future_steps,
        style_frames=control.style.num_frames,
        ground_height=torch.zeros(2),
        velocity=torch.tensor([[2.0, 0.0], [0.0, 0.0]]),
        velocity_mask=torch.tensor([[True], [False]]),
    )
    dofs = torch.zeros(2, len(robot.kinematic_info.dof_names))
    readings = SensorReadings(
        joint_pos=dofs, joint_vel=torch.zeros_like(dofs), imu_quat=rotation,
        imu_gyro=torch.zeros(2, 3),
        localization=torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, 0.8]]),
    )
    exact = NoNoise("recipe test")
    sensors = SensorConfig(
        joint_encoders=JointEncoderSensor(exact, exact),
        imu=IMUSensor(exact, exact),
        localization=LocalizationSensor(names[anchor], exact),
    )
    state = estimate_body_state(
        readings,
        BakedAnchorFK(robot.kinematic_info, anchor, torch.tensor(ids)),
        sensors.mounts(robot),
    )
    return {
        **prefix_names("noisy", body_state_names(state, anchor)),
        "mimic.anchor_idx": anchor,
        **packed.names(),
    }


def _observations(model, robot, env):
    ctx = _goal_context(robot, env.control_components["masked_mimic"])
    obs = {}
    for key in model.in_keys:
        if key in SYNTHETIC_WIDTHS:
            obs[key] = torch.rand(2, SYNTHETIC_WIDTHS[key])
        else:
            obs[key] = env.observation_components[key].compute(ctx)
    obs["vae_noise"] = torch.zeros(2, 64)
    return TensorDict(obs, batch_size=[2])


def test_goal_observations_come_from_the_real_kernels(configs):
    robot, _, env, agent, _, _ = configs
    model = _small_model(agent.model).eval()
    obs = _observations(model, robot, env)
    ids = [robot.kinematic_info.body_names.index(n) for n in robot.trackable_bodies_subset]
    anchor_slot = ids.index(robot.anchor_body_index)
    frames = env.control_components["masked_mimic"].num_masked_future_steps

    velocity = obs["masked_mimic_velocity_command"]
    assert velocity.shape == (2, 2)
    torch.testing.assert_close(velocity[0], torch.tensor([2.0, 0.0]), atol=1e-6, rtol=1e-6)
    assert not velocity[1].any()
    assert obs["masked_mimic_velocity_command_mask"].tolist() == [[1.0], [0.0]]

    # Per frame: [body, property] bits in canvas order, then the four anchor
    # submode bits (XY, height, full rotation, heading).
    masks = obs["masked_mimic_target_masks"].reshape(2, frames, len(ids) * 2 + 4)
    assert not masks[:, :-1].any()
    last = masks[:, -1]
    assert last[:, 2 * anchor_slot].tolist() == [1.0, 1.0]
    assert last[:, 2 * anchor_slot + 1].tolist() == [1.0, 1.0]
    assert last[:, 2 * 1].tolist() == [1.0, 0.0]
    assert last[:, -4:].tolist() == [[1.0, 0.0, 0.0, 1.0]] * 2

    poses = obs["fk_masked_mimic_target_poses"].reshape(2, frames, len(ids), 12)
    assert not poses[:, :-1].any()
    assert poses[0, -1, 1].any() and not poses[1, -1, 1].any()
    assert obs["masked_mimic_target_poses_masks"].tolist() == [[0.0] * (frames - 1) + [1.0]] * 2
    assert not obs["masked_mimic_local_target_poses"].any()
    assert not obs["masked_mimic_style_frame_masks"].any()
    assert all(torch.isfinite(value).all() for value in obs.values())

    with torch.no_grad():
        output = model.materialize(obs.clone())
    assert output["action"].shape == (2, 29)
    assert torch.isfinite(output["action"]).all()


class _RecordingBuffer:
    def __init__(self):
        self.data = {}

    def update_data(self, key, step, value):
        self.data[(key, step)] = value.clone()


class _ConstantExpert:
    deployable_action_key = "mean_action"

    def forward_deployable(self, tensordict):
        tensordict["mean_action"] = torch.full((tensordict.batch_size[0], 29), 7.0)
        return tensordict


def test_external_expert_labels_every_state_and_drives_sampled_episodes(configs):
    robot, _, env, agent_cfg, _, _ = configs
    model = _small_model(agent_cfg.model).eval()
    obs = _observations(model, robot, env).exclude("vae_noise")
    obs["expert_fk_max_coords_obs"] = obs["fk_max_coords_obs"]
    agent = object.__new__(SupervisedAgent)
    agent.config = agent_cfg
    agent.model = agent.training_model = model
    agent.model_output_keys = model.experience_buffer_keys()
    agent.expert_model = agent.expert_actor = _ConstantExpert()
    agent.expert_actor_in_keys = ["fk_max_coords_obs"]
    agent.device = torch.device("cpu")
    agent.experience_buffer = _RecordingBuffer()
    agent._expert_episode_mask = torch.tensor([True, False])

    out = SupervisedAgent.collect_rollout_step(agent, obs, step=3)

    stored = agent.experience_buffer.data
    torch.testing.assert_close(stored[("expert_actions", 3)], torch.full((2, 29), 7.0))
    torch.testing.assert_close(out["action"][0], torch.full((29,), 7.0))
    torch.testing.assert_close(out["action"][1], stored[("privileged_action", 3)][1])
    assert not torch.allclose(stored[("privileged_action", 3)][0], torch.full((29,), 7.0))
    for key in ("privileged_action", "solution_features"):
        assert (key, 3) in stored


def test_student_materializes_trains_and_reloads_for_prior_only_inference(configs, tmp_path):
    robot, _, env, agent, _, _ = configs
    model = _small_model(agent.model)
    obs = _observations(model, robot, env)
    model.eval()
    output = model(obs.clone())
    assert output["privileged_action"].shape == (2, 29)
    rollout = {
        key: torch.stack([output[key].detach() + step for step in range(3)])
        for key in ("privileged_action", "solution_features")
    }
    dones = torch.zeros(3, 2, dtype=torch.bool)
    dones[0, 0] = True
    rollout["dones"] = dones
    targets = model.build_training_targets(rollout)
    assert set(targets) == {
        "previous_target_privileged_action",
        "previous_target_solution_features",
        "previous_target_valid",
    }
    # Pairs straddling the reset after env 0's first action are excluded, as
    # are the buffer edges.
    assert targets["previous_target_valid"].tolist() == [[False, False], [False, True], [True, True]]
    assert not targets["previous_target_privileged_action"][1, 0].any()

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    output = model(obs.clone())
    for key, value in targets.items():
        output[key] = value[1]
    loss, log_dict = model.compute_model_loss(
        output, current_epoch=1000, zero_loss=output["action"].sum() * 0
    )
    assert {
        "model/temporal_previous_privileged_action",
        "model/temporal_previous_solution_features",
    } <= set(log_dict)
    loss = loss + output["privileged_action"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for component in (model._prior, model._encoder, model._trunk):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in component.parameters()
        )
    optimizer.step()
    model.eval()
    deployment_obs = obs.select(*model.deployable_in_keys, "vae_noise")
    expected = model.forward_deployable(deployment_obs.clone())["action"]
    torch.save(model.state_dict(), tmp_path / "student.pt")
    torch.save(agent.model, tmp_path / "resolved_configs.pt")
    saved_config = torch.load(tmp_path / "resolved_configs.pt", weights_only=False)
    assert saved_config.vae.latent_geometry is LatentGeometry.SPHERE
    restored = _small_model(saved_config).eval()
    restored.materialize(deployment_obs.clone())
    BaseAgent._load_model_state_dict(
        SimpleNamespace(model=restored), torch.load(tmp_path / "student.pt")
    )
    torch.testing.assert_close(
        restored.forward_deployable(deployment_obs.clone())["action"], expected
    )


def test_full_size_recipe_materializes(configs):
    robot, _, env, agent, _, _ = configs
    model = MaskedMimicModel(agent.model).eval()
    model.reset_rollout_context(num_envs=2, device="cpu")
    with torch.no_grad():
        output = model.materialize(_observations(model, robot, env))
    assert output["action"].shape == (2, 29)
    assert torch.isfinite(output["action"]).all()


def test_style_tokens_are_permutation_equivariant_after_learning_statistics():
    modules = recipe._goal_modules(
        ["pose", "properties", "velocity"], "tokens", 2, ["pose", "velocity"]
    )
    modules[-1].num_out = 8
    for layer in modules[-1].layers:
        layer.units = 8
    encoder = ModuleContainer(recipe._container(modules, ["tokens"]))
    inputs = TensorDict({
        "pose": torch.tensor([[1., 2.], [3., 8.]])[..., None].expand(2, 2, 9).clone(),
        "properties": torch.ones(2, 2, 3),
        "frames": torch.ones(2, 2),
        "velocity": torch.zeros(2, 2, 6),
    }, [2])
    encoder(inputs.clone())
    encoder.eval()
    permuted = TensorDict({key: value.flip(1) for key, value in inputs.items()}, [2])
    original = encoder(inputs.clone())["tokens"]
    shuffled = encoder(permuted)["tokens"]
    torch.testing.assert_close(original.flip(1), shuffled)
    torch.testing.assert_close(original.mean(1), shuffled.mean(1))
    assert encoder(inputs.clone())["pose_sequence"].shape == (2, 2, 9)


def test_frame_masks_exclude_absent_goal_tokens_from_prior(configs):
    robot, _, env, agent, _, _ = configs
    model = _small_model(agent.model).eval()
    obs = _observations(model, robot, env)
    for key in (
        "masked_mimic_target_poses_masks", "masked_mimic_local_target_poses_masks",
        "masked_mimic_style_frame_masks", "masked_mimic_velocity_command_mask",
    ):
        obs[key].zero_()
    baseline = model.forward_deployable(obs.clone())["action"]
    changed = obs.clone()
    for key in (
        "fk_masked_mimic_target_poses", "masked_mimic_local_target_poses",
        "masked_mimic_style_poses", "masked_mimic_style_velocities",
        "masked_mimic_velocity_command",
    ):
        changed[key] = torch.randn_like(changed[key]) * 100
    torch.testing.assert_close(model.forward_deployable(changed)["action"], baseline)


def test_masked_control_restores_prompts_without_height_filter_state():
    from protomotions.tests.test_structured_target_sampling import make_control
    from protomotions.envs.control.masked_mimic_control import (
        MaskedMimicStyleConfig,
        MaskedMimicVelocityCommandConfig,
        MaskedMimicControl,
        MaskedMimicControlConfig,
    )

    control = MaskedMimicControl(
        MaskedMimicControlConfig(
            reference_height_smoothing_seconds=0.2,
            anchor_pose_submasking=True,
            local_target_condition_probability=0.5,
            style=MaskedMimicStyleConfig(),
            velocity_command=MaskedMimicVelocityCommandConfig(),
        ),
        make_control().inputs,
    )
    control.reset_rows(torch.arange(3), None)
    saved = control.save_state()
    control.reset_rows(torch.tensor([0, 2]), None)
    control.load_state(saved)
    assert not hasattr(control, "ref_height_ema")
    # The registry's snapshot: every slot, the two streams as children.
    assert {"global_stream", "local_stream"} <= set(saved)
    for name, tensor in control.state.tensors():
        value = saved
        for part in name.split("."):
            value = value[part]
        torch.testing.assert_close(tensor, value)


def test_goal_preprocessing_preserves_frame_order_masks_and_times():
    modules = recipe._goal_modules(
        ["pose", "properties", "time"], "tokens", 3, ["pose"]
    )
    modules[-1].num_out = 8
    for layer in modules[-1].layers:
        layer.units = 8
    encoder = ModuleContainer(recipe._container(modules, ["tokens"]))
    pose = torch.arange(24, dtype=torch.float).reshape(2, 12)
    properties = torch.tensor([[1., 0., 0., 1., 1., 1.]]).expand(2, -1)
    times = torch.tensor([[0.1, 0.5, 1.0]]).expand(2, -1)
    output = encoder(TensorDict({"pose": pose, "properties": properties, "time": times}, [2]))
    assert output["tokens"].shape == (2, 3, 8)
    torch.testing.assert_close(output["properties_sequence"], properties.reshape(2, 3, 2))
    torch.testing.assert_close(output["time_sequence"], times.unsqueeze(-1))
    # Shared statistics, not a separate running mean for each time slot.
    normalizers = [module.running_obs_norm for module in encoder.modules()
                   if hasattr(module, "running_obs_norm") and module.running_obs_norm is not None]
    assert len(normalizers) == 1
    assert normalizers[0].mean.shape == (4,)
