# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment interface template; fill in env_config() and agent_config() to use it.

Fresh training and warm starts build the config bundle through one build
sequence (config_builder.build_configs): robot and simulator configs, the
optional configure_robot_and_simulator hook, then terrain, scene, motion,
environment and agent configs. It returns a typed ExperimentConfigs read by
attribute (configs.env, configs.agent, ...); tests and tools get the same
bundle from build_experiment(argv), and it is saved to resolved_configs.json:
one tagged JSON tree with every value resolved (each dataclass under its class
name, enums, tensors and module-level functions tagged), rebuilt on load through
the real constructors. No pickle, no YAML sidecar.

Overrides. --overrides takes root.path=value over the seven roots (robot,
simulator, terrain, scene_lib, motion_lib, env, agent). Each leaf is typed by
its dataclass annotation, never by the runtime value, so a wrong type or a typo
fails at parse time naming the field, and a derived field (a property such as
the anchor index) is refused. robot.* and simulator.* apply after the robot
hook and before the other configs are built, so those builders see the
overridden robot; the rest apply after the build, and a recipe step that
silently undoes an early override is named. For example:

    --overrides robot.anchor_body_name=torso_link \\
        env.reward_components.pow_rew.meta.weight=2e-6

Resume loads resolved_configs.json without rebuilding this file. Inference derives
its view from the same bundle at load (framework defaults, then --overrides,
then the recipe's apply_inference_overrides and an optional --inference-config
hook); there is no second saved file.
"""

import argparse

from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.utils.experiment_configs import ExperimentConfigs
from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """
    Configure robot and simulator for this experiment.

    Called during training AFTER factory creation, BEFORE env/agent configs are built.
    Results are saved to resolved_configs.json (NOT called during eval - uses frozen configs).

    This is part of config BUILDING, not overrides. Think of it like env_config() and agent_config(),
    but for robot and simulator (which have factories instead of being built from scratch).

    Args:
        robot_cfg: Robot configuration object (from factory)
        simulator_cfg: Simulator configuration object (from factory)
        args: Command line arguments

    Examples:
        # Robot configuration
        robot_cfg.asset.asset_file_name = "mjcf/g1_bm.xml"
        robot_cfg.asset.self_collisions = False
        robot_cfg.update_fields(
            contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"],
            contact_partners=["terrain"],
        )

        # Simulator configuration
        simulator_cfg.domain_randomization = DomainRandomizationConfig(
            center_of_mass=CenterOfMassDomainRandomizationConfig(
                com_range={"x": (-0.025, 0.025)},
                body_names=robot_cfg.common_naming_to_robot_body_names["torso_body_name"],  # Must be a list, not a single string
            ),
        )
    """
    pass


def terrain_config(args: argparse.Namespace):
    """
    Build terrain configuration.

    Must return a TerrainConfig; TerrainConfig() is flat ground. The config
    builder refuses None with a TypeError instead of substituting a default.
    """
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    """
    Build scene library configuration.

    Returns SceneLibConfig with scene_file set (or None for empty SceneLib).
    Always returns a config - empty SceneLib will be created if scene_file is None.
    """
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """
    Build motion library configuration.

    Returns MotionLibConfig with motion_file set (or None for empty MotionLib).
    Always returns a config - empty MotionLib will be created if motion_file is None.
    """
    from protomotions.components.motion_lib import MotionLibConfig

    motion_file = args.motion_file
    return MotionLibConfig(motion_file=motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """
    Build environment configuration (training defaults).

    This creates the base config - all eval-specific changes go in apply_inference_overrides().

    Components (``observation_components``, ``reward_components``,
    ``termination_components``): each entry is a kernel, a pure tensor function
    from ``protomotions/envs/obs``, ``envs/rewards`` or ``envs/terminations``,
    bound to the env's context with ``MdpComponent.from_kernel``. A kernel's
    parameter defaults are its bindings: a ``FieldPath`` default
    (``current_rigid_body_pos: Tensor = EnvContext.current.rigid_body_pos``) is
    read from the context at every step, a plain default is a recorded
    constant, a parameter without a default must be given. The recipe writes
    only what it chooses; an override that is a ``FieldPath`` rebinds the
    reading, any other value is a constant. Body lists are
    ``BodySelection.named(*names)``: the saved configuration keeps the names
    and the build resolves them for the robot. A reward carries its weight in
    ``meta=RewardMeta(...)``; a termination is a ``TerminationRule`` over a
    boolean kernel, or over a scalar kernel with a threshold. An actor that
    runs on the robot reads the sensor suite's estimates instead of the
    simulator's state: ``sensed(components, "noisy")`` repoints every
    ``current.*`` binding of the components it is given. For example::

        from protomotions.envs.body_selection import BodySelection
        from protomotions.envs.context_sources import sensed
        from protomotions.envs.mdp_component import MdpComponent
        from protomotions.envs.obs.humanoid import (
            compute_humanoid_max_coords_observations,
        )
        from protomotions.envs.rewards.meta import RewardMeta
        from protomotions.envs.rewards.regularization import compute_contact_match_rew
        from protomotions.envs.rewards.tracking import compute_gt_rew
        from protomotions.envs.terminations.rule import TerminationRule
        from protomotions.envs.terminations.tracking import max_body_pos_error

        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        observation_components = {
            "max_coords_obs": MdpComponent.from_kernel(
                compute_humanoid_max_coords_observations
            ),
            **sensed(
                {"proprioception": MdpComponent.from_kernel(...)}, "noisy"
            ),
        }
        reward_components = {
            "gt_rew": MdpComponent.from_kernel(
                compute_gt_rew, meta=RewardMeta(weight=0.5), coefficient=-25.0
            ),
            "contact_match_rew": MdpComponent.from_kernel(
                compute_contact_match_rew,
                meta=RewardMeta(weight=0.1),
                contact_body_ids=BodySelection.named(*feet),
                required_keys={"contact_body_index": feet},
            ),
        }
        termination_components = {
            "tracking_error": TerminationRule(
                MdpComponent.from_kernel(max_body_pos_error), 0.5
            ),
        }

    A binding to a name this configuration will not publish is refused from the
    configs alone, before anything is built (``context_check.py``), naming the
    missing fact (a sensor suite, a control component, contact bodies).

    Control components (``control_components``): a task's episode state lives in
    a control component, one class per task. A control is constructed as
    ``__init__(config, inputs)`` and reads ``self.inputs`` (a ``TaskInputs``:
    the batch, the robot, the read-only episode clock, the placed reference
    query, the ground-height query, its random generator, the UI exceptions)
    and nothing else. Its per-env tensors are allocated with
    ``self.state.add(name, shape, dtype, reset_value)`` so the env's row reset,
    snapshot and restore reach them; constants go through ``self.state.constant``.
    The env drives it: ``reset_rows(env_ids, state)`` after the physical reset,
    ``advance(state) -> logging`` once per step on the robot state the env read
    (a schedule change inside an episode goes through ``resample``), and a pure
    ``publish(env_ids)`` returning the names declared under its namespace in
    ``context_views.py``. Its episode-ending checks are the ``terminations(config,
    robot)`` classmethod: ``TerminationRule`` entries over kernels bound with
    ``MdpComponent.from_kernel``, run by the env beside the recipe's under
    ``<control>/<check>``. Every random draw passes
    ``generator=self.inputs.generator``.
    """
    env_config = None
    return env_config


def agent_config(
    robot_cfg: RobotConfig, env_cfg: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """
    Build agent configuration (training defaults).

    This creates the base config - all eval-specific changes go in apply_inference_overrides().

    Observation keys are listed once: on the actor (its mean network inherits
    them) and on the critic. The PPO/AMP/ASE model derives its in_keys and
    out_keys as the union over its component configs (actor, critic, and the
    discriminator and MI critic for AMP/ASE); passing the model an in_keys or
    out_keys list is a construction error.

    The evaluator's metrics and failures (``EvaluatorConfig.metric_components``,
    ``failure_components``) are bound like the env's components: a metric is a
    kernel bound with ``MdpComponent.from_kernel``, a failure a
    ``TerminationRule`` over one, checked against the same context before
    anything is built::

        from protomotions.envs.terminations.tracking import (
            max_body_pos_error,
            mean_body_pos_error,
        )

        metric_components = {"gt_error": MdpComponent.from_kernel(mean_body_pos_error)}
        failure_components = {
            "tracking_error": TerminationRule(
                MdpComponent.from_kernel(max_body_pos_error), 0.5
            ),
        }
    """
    agent_config = None
    return agent_config


def apply_inference_overrides(
    configs: ExperimentConfigs, args: argparse.Namespace
) -> None:
    """Customize the inference view, declared only when the recipe deviates.

    Define this hook only when the recipe's inference view differs from the
    framework defaults (``apply_inference_defaults`` on each config class plus
    the sensors reading exactly). Its body is presets plus the declared deltas.

    It runs at LOAD: inference, export and ``--full-eval`` all derive their view
    from the one saved (training) bundle this way, so editing this hook changes
    the inference behaviour of existing checkpoints, and there is no second saved
    file. The signature is fixed at ``(configs, args)``; the eight-parameter form
    is retired. A hook that silently undoes a CLI override is refused by name.

    Read and write the configs by attribute: ``configs.env``, ``configs.robot``,
    ``configs.simulator``, ``configs.agent`` (may be ``None``), ``configs.terrain``,
    ``configs.motion_lib``, ``configs.scene_lib``.

    Examples::

        from protomotions.envs.presets import tracking_playback

        tracking_playback(configs.env)             # play a tracked clip to its end
        configs.simulator.domain_randomization = None  # deterministic playback
    """
