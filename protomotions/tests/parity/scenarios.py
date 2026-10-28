# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scenario catalogue and environment construction for the parity harness.

Each :class:`Scenario` names an experiment file (or the fake simulator) and the
knobs the harness turns on top of it. :func:`build_env` builds the REAL
``BaseEnv`` the way ``train_agent`` does: ``build_experiment`` (one build
sequence), the scenario's config mutations, then ``build_env_from_configs``
(the friction conversion, the components, the env class), the one path from a
bundle to an env that training, inference and the examples take. The
MuJoCo scenarios only use git-tracked motions and the mesh-free G1 MJCF, so
they run on a checkout without Git LFS payloads; the fake scenarios need no
data files at all.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

import protomotions
from protomotions.envs.context_requirements import bound_components
from protomotions.envs.obs.state_history_buffer import history_depth

REPO_ROOT = Path(protomotions.__file__).resolve().parents[1]
SMPL_MOTION = "examples/data/smpl_humanoid_sit_armchair.motion"
G1_MOTION = "examples/data/g1_crouch_to_run.motion"
# The regular G1 MJCF references STL meshes that are LFS payloads.
G1_NO_MESH_ASSET = "mjcf/g1_bm_no_mesh_box_feet.xml"


@dataclass(frozen=True)
class Scenario:
    """One recorded rollout configuration.

    ``experiment`` is a path relative to the repository root, or ``None`` for
    the fake simulator (``tests/support/sensor_simulator.py``). ``drop`` lists
    components the git-tracked example motions cannot feed (they carry no
    contact labels). ``contact_bodies`` applies
    ``robot_config.update_fields(contact_bodies=..., contact_partners=...)``
    the way ``tests/test_eval_write_back_mujoco.py`` does (partners default to
    ``"all"``). ``scheduled_resets`` adds partial resets ``(env, period)`` on
    top of the env's own dones, so a multi-env scenario exercises row-subset
    code paths. ``fake_motions`` gives a fake scenario that many synthetic
    chain clips as a motion library (0 = no reference motion).
    ``fake_target_checks`` (a non-empty subset of ``TARGET_CHECKS``) gives a
    fake scenario the target-reaching control with those checks enabled (the
    fall check runs over the double's synthetic contact model), the target
    observation and the target reward. ``body_mass_randomization=False``
    clears the simulator config's body-mass randomization, which MuJoCo
    refuses (the five-body recipe asks for it).
    """

    name: str
    experiment: Optional[str]
    robot: str
    motion_file: Optional[str]
    steps: int = 120
    seed: int = 0
    stress: bool = False
    g1_no_mesh: bool = False
    body_mass_randomization: bool = True
    drop: Tuple[str, ...] = ("contact_match_rew",)
    contact_bodies: Optional[str] = None
    contact_partners: Union[str, Tuple[str, ...]] = "all"
    num_envs: int = 1
    simulator: str = "mujoco"
    eval_max_steps: int = 20
    # The recovery bank (recovery_prob 0.5 on the fakes); False builds without it,
    # so a recording pair can show which stage streams a feature touches.
    recovery: bool = True
    scheduled_resets: Tuple[Tuple[int, int], ...] = ()
    fake_motions: int = 0
    fake_target_checks: Tuple[str, ...] = ()


@dataclass
class BuiltEnv:
    """The constructed env plus the configs it was built from."""

    env: Any
    configs: Dict[str, Any]
    scenario: Scenario


@dataclass
class HarnessAgentConfig:
    """Stand-in for an agent config when the scenario has no experiment file.

    Only the ``evaluator`` field is read (by ``stub_agent.make_evaluator``).
    """

    evaluator: Any


_G1_TRACKER = Scenario(
    name="g1_tracker",
    experiment="examples/experiments/mimic/mlp_bm_l2c2.py",
    robot="g1",
    motion_file=G1_MOTION,
    g1_no_mesh=True,
)

# The target control's episode-ending checks (``TargetControlConfig``).
TARGET_CHECKS = ("fall", "gap", "stuck")

# MuJoCo supports num_envs=1 only, so row-subset mistakes (a partial reset
# touching other rows) are invisible there. The fake scenarios drive the real
# BaseEnv with three envs on the kinematic test double, with reset noise,
# the synthetic contact model and per-env spawn offsets turned on so that
# contact memory, respawn offsets and noise draws are exercised per row.
_FAKE3 = Scenario(
    name="fake3",
    experiment=None,
    robot="sensor_chain",
    motion_file=None,
    steps=60,
    num_envs=3,
    scheduled_resets=((1, 7), (2, 11)),
)

SCENARIOS: Dict[str, Scenario] = {
    "smpl_mimic": Scenario(
        name="smpl_mimic",
        experiment="examples/experiments/mimic/mlp.py",
        robot="smpl",
        motion_file=SMPL_MOTION,
    ),
    "g1_tracker": _G1_TRACKER,
    "g1_stress": replace(_G1_TRACKER, name="g1_stress", stress=True),
    # The G1 replication recipe (the owner's main-vs-main-next runs): five-body
    # target poses, the error-exp position and orientation rewards and the
    # tracking-failure-ratio rule. MuJoCo refuses the recipe's body-mass
    # randomization, so the scenario clears it; the example motion carries no
    # contact labels, so the contact-match reward is dropped as everywhere.
    "g1_five_body": Scenario(
        name="g1_five_body",
        experiment="examples/experiments/metatron/g1_teacher.py",
        robot="g1",
        motion_file=G1_MOTION,
        g1_no_mesh=True,
        body_mass_randomization=False,
    ),
    # On today's main the experiment binds ``historical.body_contacts``, which
    # only exists when the robot declares contact bodies. SMPL declares none by
    # default, so the scenario turns them on. Partners stay at the robot default
    # ("terrain") rather than "all": with all partners the simulator reports
    # complete contact labels and MimicEvaluator then writes the bool
    # ``raw/rigid_body_contacts`` extras into a float MotionMetrics buffer,
    # which raises on today's main (see the harness report).
    "masked_mimic": Scenario(
        name="masked_mimic",
        experiment="examples/experiments/masked_mimic/transformer.py",
        robot="smpl",
        motion_file=SMPL_MOTION,
        contact_bodies="all",
        contact_partners=("terrain",),
    ),
    "complex_terrain": Scenario(
        name="complex_terrain",
        experiment="examples/experiments/mimic/mlp_complex_terrain.py",
        robot="smpl",
        motion_file=SMPL_MOTION,
    ),
    # The two AMP locomotion tasks: a control component that owns a schedule
    # (heading and speed changes, a generated path) and, for the path follower,
    # the control's own episode-ending check (``enable_path_termination``).
    # Both recipes carry a BaseEvaluator with a threshold failure rule, so the
    # evaluator excursion runs on them too. The motion library only feeds the
    # agent-side discriminator; the env's reference motion stays unassigned.
    "steering": Scenario(
        name="steering",
        experiment="examples/experiments/steering/mlp.py",
        robot="smpl",
        motion_file=SMPL_MOTION,
    ),
    "path_follower": Scenario(
        name="path_follower",
        experiment="examples/experiments/path_follower/mlp.py",
        robot="smpl",
        motion_file=SMPL_MOTION,
    ),
    # A GPC task recipe without a scene file: the nearest-surface kernel binds
    # the scene surfaces by default and the recipe unbinds them with None, so
    # its terrain-only path is recorded, over the target-reaching control with
    # the control's own checks turned off and the target observation.
    "gpc_surface": Scenario(
        name="gpc_surface",
        experiment="examples/experiments/gpc/target_prior_peft.py",
        robot="smpl",
        motion_file=SMPL_MOTION,
    ),
    "fake3": _FAKE3,
    # Same three envs, plus a two-clip synthetic motion library, a mimic
    # control component with target-pose observations, tracking rewards and
    # a tracking-error termination, and the real MimicEvaluator: multi-env
    # reference resets, motion-time ticks, disable_motion_resample, the
    # evaluator's background-motion assignment and per-step re-alignment all
    # run with partial resets.
    "fake3_mimic": replace(_FAKE3, name="fake3_mimic", fake_motions=2),
    # Same three envs, plus the target-reaching control: a command source that
    # resamples the goal on its own schedule inside the episode, and the three
    # control-owned checks (fall over the synthetic contact model, gap and
    # stuck) feeding the termination manager's control pair, all per row.
    "fake3_target": replace(
        _FAKE3, name="fake3_target", fake_target_checks=TARGET_CHECKS
    ),
}


def seed_everything(seed: int) -> None:
    """Seed the three global generators the env and its components may use."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@contextmanager
def eager_mdp_kernels():
    """Run every MdpComponent kernel eagerly while active.

    ``component_manager.TORCH_COMPILE_AVAILABLE`` is read on every
    ``execute_all`` call, so this must wrap the whole recording (build and
    rollout), not just construction. The previous value is put back on exit.
    """
    import protomotions.envs.component_manager as component_manager

    previous = component_manager.TORCH_COMPILE_AVAILABLE
    component_manager.TORCH_COMPILE_AVAILABLE = False
    try:
        yield
    finally:
        component_manager.TORCH_COMPILE_AVAILABLE = previous


@contextmanager
def g1_mesh_free_assets(enabled: bool):
    """Make ``robot_config("g1")`` return the mesh-free G1 while active."""
    if not enabled:
        yield
        return
    import protomotions.robot_configs.factory as factory
    from protomotions.robot_configs.g1 import G1RobotConfig

    original = factory.robot_config

    def mesh_free(name: str, **updates):
        if name != "g1":
            return original(name, **updates)
        asset = replace(G1RobotConfig().asset, asset_file_name=G1_NO_MESH_ASSET)
        config = G1RobotConfig(asset=asset)
        if updates:
            config.update_fields(**updates)
        return config

    factory.robot_config = mesh_free
    try:
        yield
    finally:
        factory.robot_config = original


def apply_stress(env_cfg, robot_cfg, sim_cfg) -> None:
    """Turn on the rarely exercised env paths (the prototype's ``--stress``)."""
    from protomotions.envs.base_env.config import RecoveryResetConfig
    from protomotions.envs.context_sources import sensed
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid import (
        compute_humanoid_reduced_coords_observations,
    )
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_reduced_coords_from_state,
    )
    from protomotions.envs.sensors.config import (
        IMUSensor,
        JointEncoderSensor,
        LocalizationSensor,
        SensorConfig,
    )
    from protomotions.simulator.base_simulator.config import (
        DomainRandomizationConfig,
        LatencyDomainRandomizationConfig,
        PushDomainRandomizationConfig,
        RobotNoiseConfig,
    )

    env_cfg.max_episode_length = 40  # timeouts inside the rollout
    env_cfg.recovery_reset = RecoveryResetConfig(recovery_prob=0.5, fall_sim_steps=10)
    robot_cfg.reset_noise = RobotNoiseConfig(dof_pos_noise=0.05, dof_vel_noise=0.1)
    sim_cfg.pd_target_max_accel = 0.05
    sim_cfg.domain_randomization = DomainRandomizationConfig(
        latency=LatencyDomainRandomizationConfig(max_latency_ms=10.0),
        push=PushDomainRandomizationConfig(
            push_interval_range=(0.1, 0.3), max_linear_velocity=(0.5, 0.5, 0.0)
        ),
    )
    robot_cfg.imu_body_name = robot_cfg.kinematic_info.body_names[0]
    env_cfg.sensors = SensorConfig(
        joint_encoders=JointEncoderSensor(0.02, 0.1),
        imu=IMUSensor(0.01, 0.05),
        localization=LocalizationSensor(
            body_name=robot_cfg.kinematic_info.body_names[0], noise=0.02
        ),
        seed=7,
    )
    noisy = {
        "noisy_reduced": MdpComponent.from_kernel(
            compute_humanoid_reduced_coords_observations
        )
    }
    # The noisy history observation reads every past slot the recipe's own
    # bindings keep in the ring (none when the recipe reads no history).
    depth = history_depth(bound_components(env_cfg, robot_cfg))
    if depth:
        noisy["noisy_hist"] = MdpComponent.from_kernel(
            compute_historical_reduced_coords_from_state, history_steps=depth
        )
    env_cfg.observation_components.update(sensed(noisy, "noisy"))


def _root_height_metric(root_height: torch.Tensor) -> torch.Tensor:
    """Evaluation metric for ``fake3``: one scalar per env, straight from context."""
    return root_height


def fake3_evaluator_config():
    """A ``BaseEvaluator`` config with one metric, so ``evaluate()`` runs the real
    save -> rollout -> restore path on the fake simulator."""
    from protomotions.agents.evaluators.config import EvaluatorConfig
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    return EvaluatorConfig(
        metric_components={
            "root_height": MdpComponent(
                compute_func=_root_height_metric,
                dynamic_vars={"root_height": EnvContext.current.root_height},
            )
        }
    )


def fake_mimic_evaluator_config():
    """The ``MimicEvaluator`` config of ``examples/experiments/mimic/mlp.py``."""
    from protomotions.agents.evaluators.presets import mimic_tracking_benchmark

    return mimic_tracking_benchmark(0.5)


@dataclass
class _FakeTask:
    """The component tables a fake scenario's task adds to the base env."""

    observations: Dict[str, Any]
    rewards: Dict[str, Any]
    terminations: Dict[str, Any]
    controls: Dict[str, Any]
    env_kwargs: Dict[str, Any]


def _fake_mimic_task(num_motions: int) -> _FakeTask:
    """A synthetic motion library tracked by the mimic control (``fake3_mimic``)."""
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.target_poses import build_max_coords_target_poses
    from protomotions.envs.rewards.meta import RewardMeta
    from protomotions.envs.rewards.tracking import (
        compute_gav_rew,
        compute_gr_rew,
        compute_gt_rew,
        compute_gv_rew,
        compute_rh_rew,
    )
    from protomotions.envs.terminations.tracking import max_body_pos_error
    from protomotions.envs.terminations.rule import TerminationRule
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.tests.parity.chain_motions import build_chain_motion_lib

    return _FakeTask(
        observations={
            "mimic_target_poses": MdpComponent.from_kernel(
                build_max_coords_target_poses, with_velocities=True
            )
        },
        rewards={
            "gt_rew": MdpComponent.from_kernel(
                compute_gt_rew,
                meta=RewardMeta(weight=0.5),
            ),
            "gr_rew": MdpComponent.from_kernel(
                compute_gr_rew,
                meta=RewardMeta(weight=0.3),
            ),
            "gv_rew": MdpComponent.from_kernel(
                compute_gv_rew,
                meta=RewardMeta(weight=0.1),
            ),
            "gav_rew": MdpComponent.from_kernel(
                compute_gav_rew,
                meta=RewardMeta(weight=0.1),
            ),
            "rh_rew": MdpComponent.from_kernel(
                compute_rh_rew,
                meta=RewardMeta(weight=0.2),
            ),
        },
        terminations={
            "tracking_error": TerminationRule(
                MdpComponent.from_kernel(max_body_pos_error), 0.5
            )
        },
        controls={"mimic": MimicControlConfig(bootstrap_on_episode_end=True)},
        env_kwargs={
            "motion_lib": build_chain_motion_lib(num_motions),
            "motion_manager": MimicMotionManagerConfig(
                init_start_prob=0.5, realign_motion_with_humanoid_on_each_step=True
            ),
        },
    )


def _fake_target_task(checks: Tuple[str, ...]) -> _FakeTask:
    """The target-reaching control with the named checks on (``fake3_target``).

    The cuts are scaled to the kinematic double so that each check, enabled
    alone, fires somewhere in the 60-step rollout (``test_parity_harness``
    pins that per check). The double prescribes joint velocities only: its
    root keeps its initial velocity in zero gravity, so the anchor (the torso,
    0.3 m above a pelvis at z=0.95) moves about 1 cm per 6 frames and sinks
    about 0.1 mm per step, never more than a few millimetres below its height
    at reset. The synthetic contact model presses every body below 0.9 m onto
    the terrain, so a forbidden body below 0.85 m is a fall. The target
    resamples every 5 to 10 steps inside the 17-step episodes.
    """
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.target import compute_target_obs
    from protomotions.envs.rewards.meta import RewardMeta
    from protomotions.envs.rewards.task import compute_target_rew
    from protomotions.envs.control.command_source import (
        RandomTargetCommandSourceConfig,
    )
    from protomotions.envs.control.target_control import (
        TargetControlConfig,
    )

    unknown = set(checks) - set(TARGET_CHECKS)
    if unknown:
        raise ValueError(
            f"unknown target checks {sorted(unknown)}; expected a subset of "
            f"{TARGET_CHECKS}"
        )
    control = TargetControlConfig(
        command_source=RandomTargetCommandSourceConfig(
            tar_change_time_min=0.1, tar_change_time_max=0.2, tar_dist_max=2.0
        ),
        enable_fall_termination="fall" in checks,
        fall_termination_height=0.85,
        enable_gap_termination="gap" in checks,
        gap_termination_threshold=0.002,
        enable_stuck_termination="stuck" in checks,
        stuck_window_frames=6,
        stuck_movement_threshold=0.02,
        stuck_height_threshold=0.001,
    )
    return _FakeTask(
        observations={"target": MdpComponent.from_kernel(compute_target_obs)},
        rewards={
            "target": MdpComponent.from_kernel(
                compute_target_rew,
                meta=RewardMeta(weight=1.0),
            )
        },
        terminations={},
        controls={"target": control},
        env_kwargs={},
    )


def _merge_disjoint(table: Dict[str, Any], additions: Dict[str, Any], kind: str):
    """``table.update(additions)`` that refuses a name both already define."""
    clash = table.keys() & additions.keys()
    if clash:
        raise ValueError(f"two fake tasks define the {kind} {sorted(clash)}")
    table.update(additions)


def _build_fake_env(scenario: Scenario) -> BuiltEnv:
    from protomotions.envs.base_env.config import RecoveryResetConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_actions_from_state,
    )
    from protomotions.envs.rewards.meta import RewardMeta
    from protomotions.envs.rewards.regularization import compute_action_rate_l2
    from protomotions.envs.sensors.config import (
        IMUSensor,
        JointEncoderSensor,
        LocalizationDrift,
        LocalizationSensor,
        SensorConfig,
    )
    from protomotions.simulator.base_simulator.config import RobotNoiseConfig
    from protomotions.tests.support.sensor_simulator import make_sensor_test_env

    sensors = SensorConfig(
        joint_encoders=JointEncoderSensor(0.04, 0.1),
        imu=IMUSensor(0.02, 0.05),
        localization=LocalizationSensor(
            body_name="pelvis",
            noise=0.03,
            drift=LocalizationDrift(scale=0.1, yaw=0.1, log_magnitude=0.02),
        ),
        # The old block passed noise= and relied on make_sensor_test_env's
        # default seed; keep it so the sensor RNG stream is unchanged.
        seed=734,
    )
    observations = {
        "prev_actions": MdpComponent.from_kernel(
            compute_historical_actions_from_state, history_steps=1
        )
    }
    rewards = {
        "action_rate": MdpComponent.from_kernel(
            compute_action_rate_l2,
            meta=RewardMeta(weight=-0.1),
        )
    }
    terminations = {}
    controls = {}
    env_kwargs = {}
    evaluator_config = fake3_evaluator_config()
    tasks = []
    if scenario.fake_motions:
        tasks.append(_fake_mimic_task(scenario.fake_motions))
        evaluator_config = fake_mimic_evaluator_config()
    if scenario.fake_target_checks:
        tasks.append(_fake_target_task(scenario.fake_target_checks))
    for task in tasks:
        _merge_disjoint(observations, task.observations, "observation")
        _merge_disjoint(rewards, task.rewards, "reward")
        _merge_disjoint(terminations, task.terminations, "termination")
        _merge_disjoint(controls, task.controls, "control")
        _merge_disjoint(env_kwargs, task.env_kwargs, "env argument")
    env = make_sensor_test_env(
        sensors=sensors,
        num_envs=scenario.num_envs,
        recovery_reset=(
            RecoveryResetConfig(recovery_prob=0.5, fall_sim_steps=5)
            if scenario.recovery
            else None
        ),
        observations=observations,
        rewards=rewards,
        terminations=terminations,
        spawn_spread=True,
        contact_model=True,
        reset_noise=RobotNoiseConfig(dof_pos_noise=0.05, dof_vel_noise=0.1),
        control_components=controls or None,
        **env_kwargs,
    )
    env.max_episode_length = 40 if scenario.fake_motions else 17
    if scenario.fake_motions:
        # The double reports complete contact labels. With them MimicEvaluator
        # writes the bool ``raw/rigid_body_contacts`` extras into a float
        # MotionMetrics buffer, which raises on today's main (the same reason
        # ``masked_mimic`` keeps its partners at "terrain"); the fake mimic
        # scenario therefore reports incomplete labels like a terrain-only
        # contact layout would.
        env.simulator.has_complete_contact_labels = False
    configs = {
        "env": env.config,
        "robot": env.robot_config,
        "agent": HarnessAgentConfig(evaluator=evaluator_config),
    }
    return BuiltEnv(env=env, configs=configs, scenario=scenario)


def _build_experiment_env(scenario: Scenario) -> BuiltEnv:
    from protomotions.utils.component_builder import build_env_from_configs
    from protomotions.utils.config_builder import build_experiment

    if scenario.simulator == "mujoco":
        device = torch.device("cpu")
    else:
        # GPU mode: the simulator name and env count are passed straight
        # through. Untested here; GPU backends also have import-order rules
        # (see protomotions/utils/simulator_imports.py) that this builder
        # does not arrange.
        device = torch.device("cuda")

    argv = [
        "--robot-name",
        scenario.robot,
        "--simulator",
        scenario.simulator,
        "--num-envs",
        str(scenario.num_envs),
        "--batch-size",
        "64",
        "--motion-file",
        scenario.motion_file,
        "--experiment-path",
        str(REPO_ROOT / scenario.experiment),
        "--experiment-name",
        "parity",
        "--training-max-steps",
        "1000",
        "--headless",
        "true",
    ]

    def after_robot_hook(robot_cfg, simulator_cfg, args):
        if scenario.contact_bodies is not None:
            partners = scenario.contact_partners
            robot_cfg.update_fields(
                contact_bodies=scenario.contact_bodies,
                contact_partners=partners
                if isinstance(partners, str)
                else list(partners),
            )

    with g1_mesh_free_assets(scenario.g1_no_mesh):
        experiment = build_experiment(argv, after_robot_hook=after_robot_hook)

    env_cfg = experiment.env
    sim_cfg = experiment.simulator
    for group in (
        "observation_components",
        "reward_components",
        "termination_components",
    ):
        components = getattr(env_cfg, group) or {}
        for name in scenario.drop:
            components.pop(name, None)
    if (
        not scenario.body_mass_randomization
        and sim_cfg.domain_randomization is not None
    ):
        sim_cfg.domain_randomization.body_mass = None
    if scenario.stress:
        apply_stress(env_cfg, experiment.robot, sim_cfg)
    # The same path training takes from a bundle to an env; the friction
    # conversion is written back into ``experiment``, so the dict below carries
    # the converted terrain and simulator configs. The one argument that path
    # adds to the harness's old inline block is ``save_dir=configs.env.save_dir``,
    # and a non-empty one would make the build read ``env_<motion file>.ckpt``
    # as scene weights: a recipe that sets it fails here instead of being read.
    if env_cfg.save_dir != "":
        raise ValueError(
            f"{scenario.name}: the recipe sets env.save_dir={env_cfg.save_dir!r}; "
            "the parity harness builds with the empty default, as training does"
        )
    env = build_env_from_configs(experiment, device)
    # BuiltEnv.configs stays a harness-internal dict (stub_agent reads "agent").
    configs = {
        "robot": experiment.robot,
        "simulator": experiment.simulator,
        "terrain": experiment.terrain,
        "scene_lib": experiment.scene_lib,
        "motion_lib": experiment.motion_lib,
        "env": experiment.env,
        "agent": experiment.agent,
    }
    return BuiltEnv(env=env, configs=configs, scenario=scenario)


def build_env(scenario: Scenario) -> BuiltEnv:
    """Seed everything and build the scenario's env.

    Callers must wrap construction AND the rollout in :func:`eager_mdp_kernels`
    (``recorder.record`` does): the compile flag is read on every component
    execution, and this function deliberately does not touch it, so building
    an env never leaks eager mode into the rest of the process.
    """
    seed_everything(scenario.seed)
    if scenario.experiment is None:
        return _build_fake_env(scenario)
    return _build_experiment_env(scenario)
