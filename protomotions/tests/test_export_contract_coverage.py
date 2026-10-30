# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage guards for the D4b ONNX-exporter contract.

These tests pin the host-facing deployment sidecar the BeyondMimic tracker
exporter writes, and the refusals the exporter makes at the export boundary:

* the g1 ``mlp_bm_l2c2`` bundle's frozen policy-input keys/shapes/kinds (A),
* a per-key action-history lag (the legacy key lags 2, not a constant 1) (B),
* two history observations of different reach exporting at the shared max (C),
* an ONNX input the name heuristic cannot invert refusing, with no sidecar (D),
* ``_build_samples`` refusing a path with no deploy fact / a non-exportable
  kernel (E),
* the validate step refusing a graph whose output disagrees with the traced
  deterministic mean (F),
* a configured-but-unbound sensor channel being dropped from the sidecar (G),
* the host rule: a privileged recipe (the SOMA tracker) exports for a
  simulation host and is refused for the default robot host; a robot sidecar
  carries no ``host`` key (H).

They export the DEFAULT g1 anchor (``torso_link``); the sibling file
``test_export_anchor_body.py`` covers the pelvis-anchor rename.
"""

import argparse
import dataclasses
import logging
from types import SimpleNamespace

import pytest
import torch
import yaml

# The frozen host-facing policy inputs the default g1 mlp_bm_l2c2 bundle exports
# today, in order: (context key, ONNX shape, deploy kind). This is the
# strict-superset keys guard -- a dropped input, a changed shape, or a regressed
# kind label breaks the equality.
# (key, shape, kind, provider). The provider is pinned here too: the deleted
# agreement test was the last thing checking the derived channel of the sensor
# paths, so the g1 export's providers (imu/joint_encoders for the noisy-sourced
# current state, reference_motion for the mimic futures, policy_history for the
# action history) are frozen against regression here.
EXPECTED_G1_POLICY_INPUTS = [
    (
        "mimic.future_anchor_rot",
        (1, 4, 4),
        "reference_motion_anchor_rot",
        "reference_motion",
    ),
    (
        "mimic.future_dof_pos",
        (1, 4, 29),
        "reference_motion_joint_pos",
        "reference_motion",
    ),
    (
        "mimic.future_dof_vel",
        (1, 4, 29),
        "reference_motion_joint_vel",
        "reference_motion",
    ),
    ("current.anchor_rot", (1, 4), "anchor_rot", "imu"),
    ("current.dof_pos", (1, 29), "joint_pos", "joint_encoders"),
    ("current.dof_vel", (1, 29), "joint_vel", "joint_encoders"),
    ("current.root_local_ang_vel", (1, 3), "local_root_ang_vel", "imu"),
    ("processed_action_history", (1, 1, 29), "last_actions", "policy_history"),
]


def _build_g1_configs(mutate_env=None, mutate_agent=None):
    """Build robot/env/agent for g1 mlp_bm_l2c2 at its DEFAULT anchor.

    Unlike ``test_export_anchor_body._g1_tracker_configs`` this does NOT move the
    anchor to the pelvis, so the exported bundle is the shipped-default one.
    ``mutate_env`` / ``mutate_agent`` hook in before the configs are returned so a
    test can add an observation or retarget the actor.
    """
    from examples.experiments.mimic import mlp_bm_l2c2 as recipe
    from protomotions.robot_configs.factory import robot_config

    robot = robot_config("g1")
    args = argparse.Namespace(
        motion_file="motions.pt",
        scenes_file=None,
        batch_size=32,
        training_max_steps=1024,
    )
    recipe.configure_robot_and_simulator(robot, SimpleNamespace(), args)
    assert robot.anchor_body_name == "torso_link"
    env = recipe.env_config(robot, args)
    agent = recipe.agent_config(robot, env, args)
    if mutate_env is not None:
        mutate_env(env)
    if mutate_agent is not None:
        mutate_agent(agent)
    return robot, env, agent


def _keep_initial_actor_weights(monkeypatch):
    """Let an empty checkpoint load: the actor keeps its freshly built weights."""
    load_state_dict = torch.nn.Module.load_state_dict

    def load(self, state_dict, *args, **kwargs):
        if not state_dict:
            state_dict = self.state_dict()
        return load_state_dict(self, state_dict, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "load_state_dict", load)


def _set_actor_keys(agent, keys):
    """Point the actor at ``keys``; the mean network inherits them explicitly.

    ``PPOActorConfig.__post_init__`` copies ``in_keys`` onto ``mu_model.in_keys``
    at construction, so a post-hoc change must set both or the mean network still
    reads the recipe's original keys.
    """
    agent.model.actor.in_keys = list(keys)
    agent.model.actor.mu_model.in_keys = list(keys)


def _build_soma_configs():
    """Build robot/env/agent for the privileged SOMA tracker (``mimic/mlp.py``).

    The shipped ``soma-bones`` bundle is this recipe: its observations read the
    simulator's own body state (``current.rigid_body_*``, ``current.ground_heights``)
    and the reference body velocities, none of which a robot can feed.
    """
    from examples.experiments.mimic import mlp as recipe
    from protomotions.robot_configs.factory import robot_config

    robot = robot_config("soma23")
    args = argparse.Namespace(
        motion_file="motions.pt",
        scenes_file=None,
        batch_size=32,
        training_max_steps=1024,
    )
    recipe.configure_robot_and_simulator(robot, SimpleNamespace(), args)
    env = recipe.env_config(robot, args)
    agent = recipe.agent_config(robot, env, args)
    return robot, env, agent


def _export(robot, env, agent, out_dir, validate=True, host="robot"):
    """Save a config bundle + empty ckpt under ``out_dir`` and export the tracker."""
    from deployment import export_to_onnx
    from protomotions.utils.experiment_configs import (
        ExperimentConfigs,
        save_experiment_configs,
    )

    configs = ExperimentConfigs(
        robot=robot,
        simulator=None,
        terrain=None,
        scene_lib=None,
        motion_lib=None,
        env=env,
        agent=agent,
    )
    save_experiment_configs(configs, out_dir)
    torch.save({"model": {}}, out_dir / "last.ckpt")
    return export_to_onnx.export_tracker(
        str(out_dir / "last.ckpt"), str(out_dir / "out"), validate=validate, host=host
    )


def _sidecar(onnx_path):
    return yaml.safe_load(onnx_path.with_suffix(".yaml").read_text())


# ---------------------------------------------------------------------------
# A. The g1 sidecar carries today's keys, per-input shapes and kinds.
# ---------------------------------------------------------------------------


def test_g1_sidecar_policy_inputs_match_frozen_contract(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    _keep_initial_actor_weights(monkeypatch)
    robot, env, agent = _build_g1_configs()
    onnx_path = _export(robot, env, agent, tmp_path)

    sidecar = _sidecar(onnx_path)
    got = [
        (entry["key"], tuple(entry["shape"]), entry["kind"], entry["provider"])
        for entry in sidecar["policy_inputs"]
    ]
    # Exact, ordered: guards the inputs and each one's shape, kind and provider.
    assert got == EXPECTED_G1_POLICY_INPUTS
    # A robot export writes no host key: omitted means robot, so the sidecar is
    # byte-for-byte what it was before --host existed.
    assert "host" not in sidecar
    # The documented top-level layout the host reads.
    for key in (
        "schema",
        "policy_inputs",
        "policy_outputs",
        "robot",
        "control",
        "timing",
        "motion",
        "sensors",
    ):
        assert key in sidecar


def test_plain_tracker_has_no_fk_graph_to_export(tmp_path, monkeypatch):
    """A recipe without an FK kernel exports the four control outputs and no FK graph.

    The policy graph never carries body-pose outputs; a host gets them from
    ``fk.onnx``, which exists only for a recipe that binds ``BakedAnchorFK`` as a
    goal's ``fk=``. The plain tracker binds none, so the FK exporter refuses with
    the reason instead of writing an empty graph.
    """
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    _keep_initial_actor_weights(monkeypatch)
    robot, env, agent = _build_g1_configs()
    onnx_path = _export(robot, env, agent, tmp_path)

    control_outputs = ["actions", "joint_pos_targets", "stiffness_targets", "damping_targets"]
    sidecar = _sidecar(onnx_path)
    assert [entry["name"] for entry in sidecar["policy_outputs"]] == control_outputs
    assert sidecar["_runtime"]["onnx_out_names"] == control_outputs
    from deployment.export_fk_to_onnx import export_fk

    with pytest.raises(ValueError, match="binds an FK kernel .* nothing to export"):
        export_fk(str(tmp_path / "last.ckpt"), str(tmp_path / "fk"))
    graph = onnx.load(str(onnx_path)).graph
    assert [output.name for output in graph.output] == control_outputs


# ---------------------------------------------------------------------------
# B. first_action_lag is per-key: the legacy key lags 2, not a constant 1.
# ---------------------------------------------------------------------------


def test_legacy_action_history_entry_lags_two(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from deployment.tracker_inputs import (
        LEGACY_ACTION_HISTORY_KEY,
        action_history_steps_back,
    )
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_actions_from_state,
    )

    # Unit contrast: the first lag is per-key, not a constant. The current
    # processed history reaches 1 call back; the legacy key reaches 2.
    assert action_history_steps_back("processed_action_history", 3)[0] == 1
    assert action_history_steps_back(LEGACY_ACTION_HISTORY_KEY, 3)[0] == 2

    def mut_env(env):
        # An actor observation that reads the legacy command-history path.
        env.observation_components["legacy_prev_actions"] = MdpComponent(
            compute_func=compute_historical_actions_from_state,
            dynamic_vars={
                "historical_actions": EnvContext.historical.processed_actions
            },
            static_params={"history_steps": 1},
        )

    def mut_agent(agent):
        _set_actor_keys(
            agent,
            [
                "noisy_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses",
                "legacy_prev_actions",
            ],
        )

    _keep_initial_actor_weights(monkeypatch)
    robot, env, agent = _build_g1_configs(mut_env, mut_agent)
    onnx_path = _export(robot, env, agent, tmp_path)

    entries = {entry["key"]: entry for entry in _sidecar(onnx_path)["policy_inputs"]}
    assert LEGACY_ACTION_HISTORY_KEY in entries
    legacy = entries[LEGACY_ACTION_HISTORY_KEY]
    # The fix writes first_action_lag = steps_back[0]; for the legacy key that is
    # 2. A constant 1 (the pre-fix behaviour) would be caught here.
    assert legacy["first_action_lag"] == 2
    assert legacy["steps_back"][0] == 2
    assert legacy["output_key"] == "joint_pos_targets"


# ---------------------------------------------------------------------------
# C. Two history reaches export at the shared max (history_slots = max reach).
# ---------------------------------------------------------------------------


def test_two_history_reaches_export_at_the_max(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_actions_from_state,
    )

    def mut_env(env):
        # A reach-5 raw history and a reach-1 processed history. The reach-1 one
        # is added LAST on purpose: a "last-visited wins" bug would size both at
        # 1, not 5, and fail this test.
        env.observation_components["prev_raw_5"] = MdpComponent.from_kernel(
            compute_historical_actions_from_state, history_steps=5
        )
        env.observation_components["prev_proc_1"] = MdpComponent.from_kernel(
            compute_historical_actions_from_state,
            history_steps=1,
            historical_actions=EnvContext.processed_action_history,
        )

    def mut_agent(agent):
        _set_actor_keys(
            agent,
            [
                "noisy_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses",
                "prev_raw_5",
                "prev_proc_1",
            ],
        )

    _keep_initial_actor_weights(monkeypatch)
    robot, env, agent = _build_g1_configs(mut_env, mut_agent)
    onnx_path = _export(robot, env, agent, tmp_path)

    sidecar = _sidecar(onnx_path)
    num_dofs = len(sidecar["joint_names"])
    histories = {
        entry["key"]: entry
        for entry in sidecar["policy_inputs"]
        if entry["key"] in ("raw_action_history", "processed_action_history")
    }
    assert set(histories) == {"raw_action_history", "processed_action_history"}
    # Both inputs get the SINGLE history_slots, which is the MAX reach (5), not
    # the last-visited reach (1).
    for key, entry in histories.items():
        assert tuple(entry["shape"]) == (1, 5, num_dofs), (key, entry["shape"])


# ---------------------------------------------------------------------------
# D. An un-invertible ONNX input refuses, and no sidecar is written.
# ---------------------------------------------------------------------------


def test_unmappable_onnx_input_refuses_without_sidecar(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    import onnxruntime as ort

    real_get_inputs = ort.InferenceSession.get_inputs

    class _Stray:
        name = "stray_unmappable_input"

    def fake_get_inputs(self):
        # A graph input whose name maps to no context path; the name heuristic
        # cannot invert it.
        return list(real_get_inputs(self)) + [_Stray()]

    monkeypatch.setattr(ort.InferenceSession, "get_inputs", fake_get_inputs)

    _keep_initial_actor_weights(monkeypatch)
    robot, env, agent = _build_g1_configs()

    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="context path"):
        _export(robot, env, agent, tmp_path, validate=True)
    # The refusal happens before the sidecar is written: a short sidecar that
    # omits a required input must never reach disk.
    assert not (out_dir / "unified_pipeline.yaml").exists()


# ---------------------------------------------------------------------------
# E. _build_samples refuses an undeclared path and a non-exportable kernel.
# ---------------------------------------------------------------------------


def test_build_samples_refuses_path_without_deploy_fact():
    import torch

    from deployment.export_to_onnx import _build_samples
    from protomotions.envs.context_views import EnvContext, deploy_for
    from protomotions.envs.mdp_component import MdpComponent

    # contact_forces is a real context name with NO deploy fact. Published as a
    # tensor (a graph input), binding it in an exportable observation must be
    # refused: a tensor input the host cannot feed needs a declaration.
    assert deploy_for("contact_forces") is None

    def _reads_contacts(x):
        return x

    component = MdpComponent(_reads_contacts, {"x": EnvContext.contact_forces})
    context = {"contact_forces": torch.zeros(1, 3)}
    with pytest.raises(ValueError, match="no deployment declaration"):
        _build_samples({"bad_obs": component}, context, set())


def test_build_samples_folds_a_published_constant_without_a_fact():
    import torch

    from deployment.export_to_onnx import _build_samples
    from protomotions.envs.context_views import EnvContext, deploy_for
    from protomotions.envs.mdp_component import MdpComponent

    # anchor_idx is a published Python int (the anchor body index) every
    # max-coords / target-pose / masked-mimic kernel carries; it is bare (no
    # deploy fact) because the export module folds a non-tensor in as a constant,
    # never a graph input, so it needs no declaration. The reduced-coords default
    # pattern -- one tensor input plus the anchor index -- must export cleanly.
    assert deploy_for("current.anchor_idx") is None

    def _reads(dof_pos, anchor_body_index):
        return dof_pos

    component = MdpComponent(
        _reads,
        {
            "dof_pos": EnvContext.current.dof_pos,
            "anchor_body_index": EnvContext.current.anchor_idx,
        },
    )
    context = {"current.dof_pos": torch.zeros(1, 2), "current.anchor_idx": 3}
    samples = _build_samples({"tiny": component}, context, set())
    # The constant is taken as-is (no refusal); the tensor is its published value.
    assert samples["current.anchor_idx"] == 3
    assert torch.is_tensor(samples["current.dof_pos"])


def test_build_samples_refuses_non_exportable_kernel():
    from deployment.export_to_onnx import _build_samples
    from protomotions.envs.mdp_component import MdpComponent, is_exportable
    from protomotions.envs.obs.terrain import compute_terrain_height_map

    # The terrain height kernel is @non_exportable: it needs the whole height
    # field, which the deployable graph cannot carry.
    assert not is_exportable(compute_terrain_height_map)

    component = MdpComponent.from_kernel(compute_terrain_height_map)
    with pytest.raises(ValueError, match="non-exportable"):
        _build_samples({"terrain": component}, {}, set())


def test_build_samples_refuses_path_the_context_does_not_publish():
    from deployment.export_to_onnx import _build_samples
    from protomotions.envs.context_views import EnvContext, deploy_for
    from protomotions.envs.mdp_component import MdpComponent

    # A declared, exportable path the sample context did not publish (here an
    # empty context) is refused before tracing: a host cannot feed a name no
    # publisher produces.
    assert deploy_for("current.dof_pos") is not None

    def _reads(x):
        return x

    component = MdpComponent(_reads, {"x": EnvContext.current.dof_pos})
    with pytest.raises(ValueError, match="does not publish"):
        _build_samples({"self": component}, {}, set())


def test_build_samples_refuses_a_history_axis_without_a_replay_rule():
    import torch

    from deployment.export_to_onnx import _build_samples
    from protomotions.envs.context_views import EnvContext, deploy_for
    from protomotions.envs.mdp_component import MdpComponent

    # A slot-axis path needs an action feedback or measurement replay rule.
    # A current-state path has neither, so reject it before tracing, not after
    # the ONNX is on disk.
    assert deploy_for("current.dof_pos") is not None

    def _reads(x):
        return x

    component = MdpComponent(_reads, {"x": EnvContext.current.dof_pos})
    context = {"current.dof_pos": torch.zeros(1, 5, 2)}
    with pytest.raises(ValueError, match="history axis the host cannot feed"):
        _build_samples({"self": component}, context, {"current.dof_pos"})


def test_sensor_historical_history_specs_for_the_five_metatron_leaves():
    # Measurements start at s(t-1); their paired actions start at a(t-2).
    from deployment.contract import provider_for_path
    from deployment.input_history import history_specs
    from protomotions.envs.context_views import deploy_for

    state = {
        "noisy_historical.dof_pos": ("joint_pos", "joint_encoders"),
        "noisy_historical.dof_vel": ("joint_vel", "joint_encoders"),
        "noisy_historical.root_local_ang_vel": ("local_root_ang_vel", "imu"),
        "noisy_historical.anchor_rot": ("anchor_rot", "imu"),
    }
    for path, (kind, provider) in state.items():
        assert deploy_for(path) == kind, path
        assert provider_for_path(path) == provider, path
        action, measurement = history_specs(path, (1, 8, 29))
        assert action is None, path
        assert measurement.current_key == path.replace("_historical", ""), path
        assert measurement.steps_back == tuple(range(1, 9)), path
        assert measurement.reset_fill == "initial_measurement", path

    # The command-history leaf is the policy's own past actions fed back, lag 2.
    assert deploy_for("noisy_historical.raw_actions") == "last_actions"
    assert provider_for_path("noisy_historical.raw_actions") == "policy_history"
    spec, measurement = history_specs("noisy_historical.raw_actions", (1, 8, 29))
    assert measurement is None
    assert spec.output_key == "actions"
    assert spec.first_action_lag == 2
    assert spec.reset_fill == "zeros"
    assert spec.steps_back == (2, 3, 4, 5, 6, 7, 8, 9)


def test_sensor_history_entries_reach_the_sidecar_through_build_contract(
    tmp_path, monkeypatch
):
    # Export and reload the sidecar to verify the host gets the replay rules,
    # rather than only checking the helper's in-memory result.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from deployment.contract import DeploymentContract
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_actions_from_state,
    )

    def mut_env(env):
        # One observation per sensor-historical leaf. The two command-history
        # leaves are the policy's own past actions (policy_history); the state
        # leaf is the noisy sensor's own history ring (joint_encoders).
        for name, path in (
            ("prev_raw_actions", EnvContext.noisy_historical.raw_actions),
            ("prev_processed_actions", EnvContext.noisy_historical.processed_actions),
            ("prev_dof_pos", EnvContext.noisy_historical.dof_pos),
        ):
            env.observation_components[name] = MdpComponent(
                compute_func=compute_historical_actions_from_state,
                dynamic_vars={"historical_actions": path},
                static_params={"history_steps": 1},
            )

    def mut_agent(agent):
        _set_actor_keys(
            agent,
            [
                "noisy_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses",
                "prev_raw_actions",
                "prev_processed_actions",
                "prev_dof_pos",
            ],
        )

    _keep_initial_actor_weights(monkeypatch)
    robot, env, agent = _build_g1_configs(mut_env, mut_agent)
    onnx_path = _export(robot, env, agent, tmp_path)

    contract = DeploymentContract.from_yaml(onnx_path.with_suffix(".yaml"))
    by_key = {spec.key: spec for spec in contract.inputs}

    # raw_actions: the policy's own past actions, replayed from the raw-action
    # output at lag 2 (steps_back [2] for a one-slot window), zeroed on reset.
    raw = by_key["noisy_historical.raw_actions"]
    assert raw.kind == "last_actions" and raw.provider == "policy_history"
    assert raw.history is not None
    assert raw.history.output_key == "actions"
    assert raw.history.slots == 1
    assert raw.history.first_action_lag == 2
    assert raw.history.steps_back == (2,)
    assert raw.history.reset_fill == "zeros"
    assert raw.history.reset_value == 0.0

    # processed_actions: same lag-2 command history, replaying the joint targets.
    proc = by_key["noisy_historical.processed_actions"]
    assert proc.kind == "last_actions" and proc.provider == "policy_history"
    assert proc.history is not None
    assert proc.history.output_key == "joint_pos_targets"
    assert proc.history.slots == 1
    assert proc.history.first_action_lag == 2
    assert proc.history.steps_back == (2,)
    assert proc.history.reset_fill == "zeros"
    assert proc.history.reset_value == 0.0

    # dof_pos: past measurements, one call back, repeating the reset reading.
    dof = by_key["noisy_historical.dof_pos"]
    assert dof.kind == "joint_pos" and dof.provider == "joint_encoders"
    assert dof.history is None
    assert dof.observation_history is not None
    assert dof.observation_history.current_key == "noisy.dof_pos"
    assert dof.observation_history.steps_back == (1,)
    assert dof.observation_history.reset_fill == "initial_measurement"


# ---------------------------------------------------------------------------
# F. The validate step accepts the deterministic mean and refuses a stochastic
#    graph at the export boundary.
# ---------------------------------------------------------------------------


def test_export_validation_accepts_the_deterministic_mean(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    _keep_initial_actor_weights(monkeypatch)
    robot, env, agent = _build_g1_configs()
    # validate=True internally checks ONNX output == traced deployable (mean)
    # output within 1e-4; a deterministic policy passes and writes both files.
    onnx_path = _export(robot, env, agent, tmp_path, validate=True)
    assert onnx_path.exists()
    assert onnx_path.with_suffix(".yaml").exists()


def test_export_validation_refuses_a_stochastic_graph(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from protomotions.agents.ppo.model import PPOActor

    real_forward_deployable = PPOActor.forward_deployable

    def stochastic_forward_deployable(self, tensordict, *args, **kwargs):
        tensordict = real_forward_deployable(self, tensordict, *args, **kwargs)
        action = tensordict["mean_action"]
        # Make the "deterministic" path non-deterministic: the eager forward and
        # the traced constant then disagree far beyond 1e-4.
        tensordict["mean_action"] = action + torch.randn_like(action) * 10.0
        return tensordict

    monkeypatch.setattr(PPOActor, "forward_deployable", stochastic_forward_deployable)

    _keep_initial_actor_weights(monkeypatch)
    robot, env, agent = _build_g1_configs()
    # The boundary RAISES (it no longer only warns) when the ONNX graph's output
    # disagrees with the traced deployable output.
    with pytest.raises(ValueError, match="disagrees with the traced"):
        _export(robot, env, agent, tmp_path, validate=True)


# ---------------------------------------------------------------------------
# G. A configured sensor channel the actor never binds is dropped.
# ---------------------------------------------------------------------------


def test_configured_but_unbound_localization_is_dropped(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from protomotions.envs.sensors.config import LocalizationSensor

    def mut_env(env):
        # Configure a localization sensor. The actor still reads only joint
        # encoders + the IMU (no localization path), so the bound-paths filter
        # must drop this channel from the sidecar.
        env.sensors = dataclasses.replace(
            env.sensors,
            localization=LocalizationSensor(body_name="pelvis", noise=0.05),
        )

    _keep_initial_actor_weights(monkeypatch)
    robot, env, agent = _build_g1_configs(mut_env)
    # The channel IS configured; the drop below is the filter at work, not an
    # absent sensor.
    assert env.sensors.localization is not None

    onnx_path = _export(robot, env, agent, tmp_path)
    sidecar = _sidecar(onnx_path)
    assert sidecar["sensors"]["localization"] is None


# ---------------------------------------------------------------------------
# H. The exporter derives configs from the JSON bundle and warns about nothing.
# ---------------------------------------------------------------------------


def test_export_on_a_json_only_run_directory_is_quiet(tmp_path, monkeypatch, caplog):
    """No inference-pickle fallback warning when exporting a JSON-only run dir.

    The exporter derives its configs from the JSON training bundle (B2); the
    deleted fallback probed for the inference pickle and warned about domain
    randomization on every JSON-only run directory -- which is every run
    directory now -- naming files the loader never reads.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    _keep_initial_actor_weights(monkeypatch)
    robot, env, agent = _build_g1_configs()
    with caplog.at_level(logging.WARNING):
        _export(robot, env, agent, tmp_path)

    noisy = [
        record.getMessage()
        for record in caplog.records
        if "resolved_configs" in record.getMessage()
        or "Domain randomization" in record.getMessage()
    ]
    assert not noisy, noisy


# ---------------------------------------------------------------------------
# H. The MaskedMimic deployable helpers: the deployable in_keys (prior + trunk
#    minus prior outputs), with strict checkpoint loading. The full build and
#    ONNX/LEAPP roundtrip are covered by test_metatron_generic_export.py.
# ---------------------------------------------------------------------------


def test_deployable_policy_in_keys_mimic_and_masked():
    from types import SimpleNamespace

    from deployment.export_to_onnx import _deployable_policy_in_keys

    # A plain actor exposes its own in_keys and is returned for the caller to build.
    actor_cfg = SimpleNamespace(in_keys=["x", "y"])
    got_cfg, keys = _deployable_policy_in_keys(SimpleNamespace(actor=actor_cfg))
    assert got_cfg is actor_cfg
    assert keys == ["x", "y"]

    # MaskedMimic has no actor: the deployable inputs are the prior's plus the
    # trunk's, deduped, minus the internal latent key the prior produces.
    from protomotions.agents.common.config import ModuleContainerConfig
    from protomotions.agents.common.latent import VAE_LATENT_KEY
    from protomotions.agents.supervised.masked_mimic_config import (
        MaskedMimicModelConfig,
    )

    model_config = MaskedMimicModelConfig(
        prior=ModuleContainerConfig(in_keys=["a", "b"]),
        trunk=ModuleContainerConfig(in_keys=["b", VAE_LATENT_KEY, "c"]),
    )
    got_cfg, keys = _deployable_policy_in_keys(model_config)
    assert got_cfg is None
    assert keys == ["a", "b", "c"]  # vae_latent dropped, "b" deduped


def test_load_deployable_weights_checks_the_full_masked_model():
    from deployment.export_to_onnx import _load_deployable_weights
    from protomotions.tests.test_masked_mimic_model import (
        _masked_mimic_model_with_kld_schedule,
    )

    model = _masked_mimic_model_with_kld_schedule(None)
    for module in (model._prior, model._trunk, model._encoder):
        module.add_module("weights", torch.nn.Linear(2, 2))
    state = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
    _load_deployable_weights(model, None, state)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key])
    # The encoder is not executed during export, but an incomplete full training
    # checkpoint must still be rejected rather than silently accepted.
    del state["_encoder.weights.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        _load_deployable_weights(model, None, state)


def _historical_proprio_fk(dof_pos, anchor_rot):
    """A trivial FK for the real history kernel; baked FK has separate coverage."""
    return torch.zeros(dof_pos.shape[0], 3, device=dof_pos.device, dtype=dof_pos.dtype)


def test_real_proprioception_kernel_pins_the_five_sensor_history_entries(
    tmp_path, monkeypatch
):
    # The #248 re-review's kernel-level guard: drive the REAL
    # compute_historical_proprioception kernel (not a stand-in) through the
    # exporter and pin the five Metatron sensor-history entries it reads. dt is
    # bound too and must be folded into the graph as a constant, never a host
    # input.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from deployment.contract import DeploymentContract
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_proprioception,
    )

    def mut_env(env):
        env.observation_components["deployable_proprio"] = MdpComponent(
            compute_func=compute_historical_proprioception,
            dynamic_vars={
                "historical_dof_pos": EnvContext.noisy_historical.dof_pos,
                "historical_dof_vel": EnvContext.noisy_historical.dof_vel,
                "historical_root_local_ang_vel": (
                    EnvContext.noisy_historical.root_local_ang_vel
                ),
                "historical_anchor_rot": EnvContext.noisy_historical.anchor_rot,
                "historical_actions": EnvContext.noisy_historical.raw_actions,
                "dt": EnvContext.dt,
            },
            static_params={"fk": _historical_proprio_fk, "history_steps": [1]},
        )

    def mut_agent(agent):
        _set_actor_keys(
            agent,
            [
                "noisy_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses",
                "deployable_proprio",
            ],
        )

    _keep_initial_actor_weights(monkeypatch)
    robot, env, agent = _build_g1_configs(mut_env, mut_agent)
    onnx_path = _export(robot, env, agent, tmp_path)

    contract = DeploymentContract.from_yaml(onnx_path.with_suffix(".yaml"))
    by_key = {spec.key: spec for spec in contract.inputs}

    # Measurements replay the previous sensor reading (lag 1), with the initial
    # reading repeated after reset. The incoming-action history has lag 2.
    state = {
        "noisy_historical.dof_pos": ("joint_pos", "joint_encoders"),
        "noisy_historical.dof_vel": ("joint_vel", "joint_encoders"),
        "noisy_historical.root_local_ang_vel": ("local_root_ang_vel", "imu"),
        "noisy_historical.anchor_rot": ("anchor_rot", "imu"),
    }
    for key, (kind, provider) in state.items():
        spec = by_key[key]
        assert spec.kind == kind and spec.provider == provider
        assert spec.history is None
        assert spec.observation_history.current_key == key.replace(
            "noisy_historical.", "noisy."
        )
        assert spec.observation_history.steps_back == (1,)
        assert spec.observation_history.reset_fill == "initial_measurement"

    raw = by_key["noisy_historical.raw_actions"]
    assert raw.kind == "last_actions" and raw.provider == "policy_history"
    assert raw.history.output_key == "actions"
    assert raw.history.first_action_lag == 2
    assert raw.history.reset_fill == "zeros"

    # dt is a scalar bound by the kernel; it is folded into the graph, never fed.
    assert "dt" not in by_key
    assert not any(spec.source_namespace == "dt" for spec in contract.inputs)


def test_current_action_exports_as_a_single_slot_action_feedback(tmp_path, monkeypatch):
    # The policy's own last raw action -- the last_action observation reads
    # EnvContext.current_action -- exports as a single-slot action feedback from
    # the "actions" output at lag 1, declared flat [1, D] (no slot axis). This is
    # the current_action half of the #248 "dt + current_action" note.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from deployment.contract import DeploymentContract
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.cast import to_float

    def mut_env(env):
        env.observation_components["prev_action"] = MdpComponent.from_kernel(
            to_float, x=EnvContext.current_action
        )

    def mut_agent(agent):
        _set_actor_keys(
            agent,
            [
                "noisy_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses",
                "prev_action",
            ],
        )

    _keep_initial_actor_weights(monkeypatch)
    robot, env, agent = _build_g1_configs(mut_env, mut_agent)
    onnx_path = _export(robot, env, agent, tmp_path)

    contract = DeploymentContract.from_yaml(onnx_path.with_suffix(".yaml"))
    spec = {s.key: s for s in contract.inputs}["current_action"]
    assert spec.kind == "last_actions" and spec.provider == "policy_history"
    assert len(spec.shape) == 2  # flat [1, D], not a [1, N, D] window
    assert spec.history is not None
    assert spec.history.output_key == "actions"
    assert spec.history.slots == 1
    assert spec.history.first_action_lag == 1
    assert spec.history.steps_back == (1,)
    assert spec.history.reset_fill == "zeros"
    assert spec.history.reset_value == 0.0


def test_deployable_in_keys_drops_a_prior_emitted_decoder_condition():
    # A conditional-VAE prior can emit a decoder condition alongside the latent
    # (the Metatron v1 demo model does). The trunk reads it as internal data, not
    # a host input, so it must be filtered out of the deployable in_keys together
    # with the latent -- otherwise the exporter would demand a graph input the
    # host cannot feed.
    from deployment.export_to_onnx import _deployable_policy_in_keys
    from protomotions.agents.common.config import ModuleContainerConfig
    from protomotions.agents.common.latent import VAE_LATENT_KEY
    from protomotions.agents.supervised.masked_mimic_config import (
        MaskedMimicModelConfig,
    )

    model_config = MaskedMimicModelConfig(
        prior=ModuleContainerConfig(
            in_keys=["a", "b"], out_keys=["mu", "logvar", "prior_condition"]
        ),
        trunk=ModuleContainerConfig(
            in_keys=["b", VAE_LATENT_KEY, "prior_condition", "c"]
        ),
    )
    actor_cfg, keys = _deployable_policy_in_keys(model_config)
    assert actor_cfg is None
    # "prior_condition" and the latent are prior-produced -> dropped; "b" deduped.
    assert keys == ["a", "b", "c"]
    from protomotions.tests.test_masked_mimic_model import (
        _masked_mimic_model_with_kld_schedule,
    )

    model = _masked_mimic_model_with_kld_schedule(None)
    model._prior.in_keys = model_config.prior.in_keys
    model._prior.out_keys = model_config.prior.out_keys
    model._trunk.in_keys = model_config.trunk.in_keys
    assert model.deployable_in_keys == keys


# ---------------------------------------------------------------------------
# H. Which host: the simulator's body state exports only for a simulation host.
# ---------------------------------------------------------------------------

# The SOMA tracker's graph inputs for a simulation host, in the exporter's
# order (sorted context path): (context key, ONNX shape, kind, provider).
# soma23 has 66 dofs and 23 bodies; the recipe's mimic control publishes one
# future step. Every ``current.*`` body-state input carries a ``simulated_*``
# kind, which is why the default robot export refuses this recipe.
EXPECTED_SOMA_SIMULATION_INPUTS = [
    ("current.ground_heights", (1,), "simulated_ground_heights", "simulator_truth"),
    (
        "current.rigid_body_ang_vel",
        (1, 23, 3),
        "simulated_body_ang_vel",
        "simulator_truth",
    ),
    ("current.rigid_body_pos", (1, 23, 3), "simulated_body_pos", "simulator_truth"),
    ("current.rigid_body_rot", (1, 23, 4), "simulated_body_rot", "simulator_truth"),
    ("current.rigid_body_vel", (1, 23, 3), "simulated_body_vel", "simulator_truth"),
    (
        "mimic.future_ang_vel",
        (1, 1, 23, 3),
        "reference_motion_body_ang_vel",
        "reference_motion",
    ),
    (
        "mimic.future_pos",
        (1, 1, 23, 3),
        "reference_motion_body_pos_future",
        "reference_motion",
    ),
    ("mimic.future_rot", (1, 1, 23, 4), "reference_motion_body_rot", "reference_motion"),
    ("mimic.future_vel", (1, 1, 23, 3), "reference_motion_body_vel", "reference_motion"),
    ("raw_action_history", (1, 1, 66), "last_actions", "policy_history"),
]


def test_privileged_tracker_exports_for_a_simulation_host(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from deployment.contract import DeploymentContract

    _keep_initial_actor_weights(monkeypatch)
    robot, env, agent = _build_soma_configs()
    onnx_path = _export(robot, env, agent, tmp_path, host="simulation")

    sidecar = _sidecar(onnx_path)
    # The host is recorded right after the file's type, so a reader sees what
    # the graph expects to be fed before it reads the inputs.
    assert list(sidecar)[:3] == ["schema", "type", "host"]
    assert sidecar["host"] == "simulation"
    got = [
        (entry["key"], tuple(entry["shape"]), entry["kind"], entry["provider"])
        for entry in sidecar["policy_inputs"]
    ]
    assert got == EXPECTED_SOMA_SIMULATION_INPUTS
    # Every simulated input has a host-facing description and body labels in
    # the robot's own body order; the ground height is a single lane.
    by_key = {entry["key"]: entry for entry in sidecar["policy_inputs"]}
    assert by_key["current.rigid_body_pos"]["element_names"] == [
        list(robot.kinematic_info.body_names),
        ["x", "y", "z"],
    ]
    assert by_key["current.rigid_body_rot"]["element_names"][1] == ["x", "y", "z", "w"]
    assert "element_names" not in by_key["current.ground_heights"]
    for key in ("current.rigid_body_pos", "current.ground_heights", "mimic.future_vel"):
        assert by_key[key]["source"], key
    # No sensor suite is declared and none is claimed.
    assert sidecar["sensors"] is None
    contract = DeploymentContract.from_yaml(onnx_path.with_suffix(".yaml"))
    assert contract.host == "simulation"


def test_privileged_tracker_is_refused_for_the_robot_host(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    _keep_initial_actor_weights(monkeypatch)
    robot, env, agent = _build_soma_configs()
    with pytest.raises(ValueError) as refusal:
        _export(robot, env, agent, tmp_path)
    message = str(refusal.value)
    assert "actor observation 'max_coords_obs' binds 'current.rigid_body_pos'" in message
    assert "host kind 'simulated_body_pos', which only a simulation host can feed" in message
    assert "this export is for a robot host" in message
    assert "--host simulation" in message
    # Refused before anything is written.
    assert not (tmp_path / "out").exists()


def test_unknown_host_is_refused_before_loading(tmp_path):
    from deployment import export_to_onnx

    with pytest.raises(ValueError, match="unknown deployment host 'browser'"):
        export_to_onnx.export_from_configs(
            None, str(tmp_path / "last.ckpt"), str(tmp_path / "out"), host="browser"
        )
    # The public entry point refuses before it opens the bundle: no checkpoint
    # exists here, so a load attempt would raise FileNotFoundError instead.
    with pytest.raises(ValueError, match="unknown deployment host 'browser'"):
        export_to_onnx.export_tracker(
            str(tmp_path / "missing" / "last.ckpt"),
            str(tmp_path / "out"),
            host="browser",
        )
    assert not (tmp_path / "out").exists()


def test_privileged_history_is_refused_by_the_history_rule_for_every_host():
    import torch

    from deployment.export_to_onnx import _build_samples
    from protomotions.envs.context_views import EnvContext, deploy_for
    from protomotions.envs.mdp_component import MdpComponent

    # The historical privileged view inherits the simulated_* kinds, but no host
    # can feed past simulator truth (no replay rule is declared for it), so the
    # refusal names the history rule instead of sending the user to a host that
    # refuses the same binding.
    assert deploy_for("historical.rigid_body_pos") == "simulated_body_pos"

    def _reads_history(x):
        return x

    component = MdpComponent(
        _reads_history, {"x": EnvContext.historical.rigid_body_pos}
    )
    context = {"historical.rigid_body_pos": torch.zeros(1, 8, 33, 3)}
    for host in ("robot", "simulation"):
        with pytest.raises(ValueError, match="history axis the host cannot feed"):
            _build_samples(
                {"past": component}, context, {"historical.rigid_body_pos"}, host=host
            )

