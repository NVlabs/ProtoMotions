# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The state publisher's derived names, the vocabulary, and the kernels' bindings."""

import pytest
import torch

from protomotions.envs.base_env.publishers import StatePublisher
from protomotions.envs.context import FrozenContext
from protomotions.envs.context_paths import FieldPath, NestedField, resolve_path
from protomotions.envs.context_sources import sensed
from protomotions.envs.context_views import (
    CONTROL_NAMESPACES,
    NAMESPACES,
    EnvContext,
    SensorStateView,
    declared_paths,
    namespace_owner,
    namespace_requires,
)
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.humanoid import (
    compute_humanoid_max_coords_contact_observations,
    compute_humanoid_max_coords_observations,
    compute_humanoid_reduced_coords_observations,
)
from protomotions.envs.obs.humanoid_historical import (
    compute_historical_actions_from_state,
    compute_historical_max_coords_from_state,
    compute_historical_reduced_coords_from_state,
)
from protomotions.envs.obs.path import compute_path_obs
from protomotions.envs.obs.state_history_buffer import HistorySample, StateHistoryBuffer
from protomotions.envs.obs.steering import compute_steering_obs
from protomotions.envs.obs.target import compute_target_obs
from protomotions.envs.obs.target_poses import (
    build_deploy_target_poses,
    build_max_coords_target_poses,
    build_max_coords_target_poses_future_rel,
    build_reduced_coords_target_poses,
    compute_odom_offset_local,
)
from protomotions.envs.state import BodyState


def test_masked_mimic_float_casts_remain_eager():
    from protomotions.envs.obs.cast import to_float
    from protomotions.envs.obs.masked_mimic import masked_mimic_goal_observations
    from protomotions.tests.test_anchor_local_goals import canvas

    # The goal builder's two mask casts, and the style and velocity-command
    # casts a recipe writes itself, all run `to_float` eagerly: a bool-to-float
    # cast is not worth a compiled graph.
    components = masked_mimic_goal_observations(canvas("a", "b"))
    casts = [c for c in components.values() if c.compute_func is to_float]
    assert len(casts) == 2
    assert all(c.compute_func.run_eagerly for c in casts)
    assert MdpComponent.from_kernel(
        to_float, x=EnvContext.masked_mimic.style_frame_masks
    ).compute_func.run_eagerly
    for shape in ((2, 3), (2, 3, 4), (2, 3, 4, 5)):
        value = torch.ones(shape, dtype=torch.bool)
        assert torch.equal(to_float(value), value.float())


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def _body_state(*lead: int, num_bodies: int = 3, num_dofs: int = 4) -> BodyState:
    """A fully populated privileged state with distinguishable values."""
    count = 1
    for dim in lead:
        count *= dim
    pos = torch.arange(count * num_bodies * 3, dtype=torch.float).reshape(
        *lead, num_bodies, 3
    )
    return BodyState(
        rigid_body_rot=_identity_quat(*lead, num_bodies),
        rigid_body_ang_vel=torch.ones(*lead, num_bodies, 3) * 2.0,
        dof_pos=torch.ones(*lead, num_dofs),
        dof_vel=torch.ones(*lead, num_dofs) * 2.0,
        rigid_body_pos=pos,
        rigid_body_vel=torch.ones(*lead, num_bodies, 3),
        ground_heights=torch.ones(*lead) * 0.5,
        rigid_body_contacts=torch.arange(count * num_bodies).reshape(*lead, num_bodies)
        % 2
        == 0,
        dof_forces=torch.ones(*lead, num_dofs) * 3.0,
    )


# =============================================================================
# The state publisher
# =============================================================================


def _publish(states, anchor_idx: int, *, history_steps: int = 0, num_actions: int = 5):
    """What the state publisher publishes from a ring seeded with ``states``."""
    envs = next(iter(states.values())).dof_pos.shape[0]
    actions = torch.zeros(envs, num_actions)
    ring = StateHistoryBuffer(
        history_steps, HistorySample(states, actions, actions.clone())
    )
    return StatePublisher(ring, anchor_idx).publish(None)


def test_state_publisher_derives_root_and_anchor_names_from_the_state():
    state = _body_state(2)

    published = _publish({"privileged": state}, anchor_idx=2)

    assert torch.equal(published["current.root_pos"], state.rigid_body_pos[:, 0])
    assert torch.equal(published["current.root_height"], state.rigid_body_pos[:, 0, 2])
    assert torch.equal(published["current.anchor_pos"], state.rigid_body_pos[:, 2])
    assert torch.equal(published["current.anchor_vel"], state.rigid_body_vel[:, 2])
    # Identity orientations: local angular velocity equals the world one.
    assert torch.equal(
        published["current.root_local_ang_vel"], state.rigid_body_ang_vel[:, 0]
    )
    assert torch.equal(
        published["current.anchor_local_ang_vel"], state.rigid_body_ang_vel[:, 2]
    )
    assert torch.equal(published["current.ground_heights"], state.ground_heights)
    # The top-level ground height is the privileged one, the same object.
    assert published["ground_heights"] is published["current.ground_heights"]
    assert published["current.anchor_idx"] == 2
    # Without history: no historical namespace and no previous command.
    assert not any(key.startswith("historical.") for key in published)
    assert "previous_action" not in published
    assert published["current_action"].shape == (2, 5)
    assert EnvContext.current.anchor_pos.path == "current.anchor_pos"
    assert EnvContext.noisy_historical.dof_pos.path == "noisy_historical.dof_pos"
    assert (
        EnvContext.mimic.ref_state.rigid_body_pos.path
        == "mimic.ref_state.rigid_body_pos"
    )


def test_state_publisher_omits_quantities_a_sensor_source_does_not_have():
    sensor_like = BodyState(
        rigid_body_rot=_identity_quat(2, 3),
        rigid_body_ang_vel=torch.ones(2, 3, 3),
        dof_pos=torch.ones(2, 4),
        dof_vel=torch.ones(2, 4),
    )

    published = _publish({"privileged": _body_state(2), "noisy": sensor_like}, 1)

    assert torch.equal(published["noisy.anchor_rot"], sensor_like.rigid_body_rot[:, 1])
    declared = declared_paths("noisy")
    for name in (
        "rigid_body_pos",
        "root_pos",
        "anchor_pos",
        "root_height",
        "root_vel",
        "anchor_vel",
        "ground_heights",
    ):
        assert f"noisy.{name}" not in published
        # Absent, and declared with what a suite needs to produce it.
        assert declared[f"noisy.{name}"].requires, name
    # Contacts and joint forces are not sensor names at all.
    assert "noisy.rigid_body_contacts" not in declared
    assert "noisy.dof_forces" not in declared
    # The privileged source still has them.
    assert "current.root_pos" in published


def test_state_publisher_keeps_the_time_axis_and_state_aligned_commands():
    initial = _body_state(2)
    ring = StateHistoryBuffer(
        2, HistorySample({"privileged": initial}, torch.zeros(2, 5), torch.zeros(2, 5))
    )
    for step in (7.0, 8.0):
        state = _body_state(2)
        state.dof_pos.fill_(step)
        raw = torch.full((2, 5), step * 10)
        ring.push(HistorySample({"privileged": state}, raw, raw * 2))

    published = StatePublisher(ring, 1).publish(None)

    # Slots 1 onward, oldest last; the command stored with each state.
    assert published["historical.root_pos"].shape == (2, 2, 3)
    assert published["historical.root_local_ang_vel"].shape == (2, 2, 3)
    assert published["historical.dof_pos"][0, :, 0].tolist() == [7.0, 1.0]
    assert published["historical.raw_actions"][0, :, 0].tolist() == [70.0, 0.0]
    assert published["historical.processed_actions"][0, :, 0].tolist() == [140.0, 0.0]
    assert torch.equal(
        published["historical.anchor_rot"],
        published["historical.rigid_body_rot"][:, :, 1],
    )
    assert (
        published["historical.body_contacts"]
        is published["historical.rigid_body_contacts"]
    )
    # The command history starts at slot 0 (the policy's latest output).
    assert published["raw_action_history"][0, :, 0].tolist() == [80.0, 70.0, 0.0]
    assert published["current_action"][0, 0] == 80.0
    assert published["previous_action"][0, 0] == 70.0
    assert published["previous_processed_action"][0, 0] == 140.0

    subset = StatePublisher(ring, 1).publish(torch.tensor([1]))
    assert subset["historical.root_pos"].shape == (1, 2, 3)
    assert torch.equal(subset["current.dof_pos"], published["current.dof_pos"][[1]])


# =============================================================================
# The flat context and the vocabulary
# =============================================================================


def test_context_distinguishes_absent_from_explicit_none():
    context = FrozenContext.merge(
        {"control": {"masked_mimic.ref_rot": None}}, env_ids=None, dt=0.1
    )

    assert "masked_mimic.ref_rot" in context
    assert context["masked_mimic.ref_rot"] is None
    assert resolve_path(context, "masked_mimic.ref_rot") is None
    assert "masked_mimic.ref_pos" not in context
    with pytest.raises(AttributeError, match="masked_mimic.ref_pos"):
        resolve_path(context, "masked_mimic.ref_pos")
    assert EnvContext.masked_mimic.local_ref_pos.path == "masked_mimic.local_ref_pos"


def test_vocabulary_declares_namespaces_and_tags_what_each_name_requires():
    assert set(NAMESPACES) == {
        "",
        "current",
        "clean",
        "noisy",
        "historical",
        "clean_historical",
        "noisy_historical",
        "terrain",
        "scene",
        "mimic",
        "masked_mimic",
        "steering",
        "path",
        "target",
    }
    # What a namespace as a whole needs is on its NestedField; the names
    # inside carry their own requirements.
    for namespace in ("", "current", "scene"):
        assert namespace_requires(namespace) == ()
    assert namespace_requires("clean") == namespace_requires("noisy") == ("sensors",)
    assert namespace_requires("historical") == ("history",)
    assert namespace_requires("noisy_historical") == ("sensors", "history")
    assert namespace_requires("terrain") == ("terrain_heightfield",)
    for namespace in CONTROL_NAMESPACES:
        assert namespace_requires(namespace) == ()
        assert namespace_owner(namespace) == "control"
    assert namespace_owner("current") is None and namespace_owner("") is None
    # Simulator truth promises every body quantity; a sensor source promises
    # what the suite measures, and nothing about contacts or joint forces.
    current, noisy = declared_paths("current"), declared_paths("noisy")
    assert current["current.dof_pos"].requires == ()
    assert current["current.root_pos"].requires == ()
    assert current["current.dof_forces"].requires == ()
    assert current["current.rigid_body_contacts"].requires == ("contacts",)
    assert "current.odom_disp_start" not in current
    assert noisy["noisy.dof_pos"].requires == ()
    assert noisy["noisy.root_pos"].requires == ("localization",)
    assert noisy["noisy.odom_disp_start"].requires == ("localization",)
    assert noisy["noisy.root_vel"].requires == ("base_velocity",)
    assert noisy["noisy.ground_heights"].requires == ("localization", "height_sensor")
    assert "noisy.rigid_body_contacts" not in noisy and "noisy.dof_forces" not in noisy
    historical = declared_paths("historical")
    assert historical["historical.root_pos"].requires == ()
    assert historical["historical.raw_actions"].requires == ()
    assert historical["historical.body_contacts"].requires == ("contacts",)
    assert historical["historical.rigid_body_contacts"].requires == ("contacts",)
    sensor_historical = declared_paths("noisy_historical")
    assert sensor_historical["noisy_historical.root_pos"].requires == ("localization",)
    assert "noisy_historical.body_contacts" not in sensor_historical
    masked = declared_paths("masked_mimic")
    assert masked["masked_mimic.ref_pos"].requires == ()
    assert masked["masked_mimic.local_ref_pos"].requires == ("local_stream",)
    assert masked["masked_mimic.style_frame_masks"].requires == ("style",)
    assert masked["masked_mimic.average_velocity"].requires == ("velocity_command",)
    assert masked["masked_mimic.anchor_rotation_mask"].requires == (
        "anchor_pose_submasking",
    )
    mimic = declared_paths("mimic")
    assert mimic["mimic.ref_state.dof_pos"].requires == ()
    assert mimic["mimic.ref_state.rigid_body_contacts"].requires == ("motion_file",)
    assert mimic["mimic.ref_state.local_rigid_body_rot"].requires == ("motion_file",)
    assert "mimic.ref_state.dof_forces" not in mimic
    assert mimic["mimic.smoothed_ref_anchor_height"].requires == (
        "reference_height_smoothing_seconds",
    )
    top = declared_paths("")
    assert top["current_action"].requires == ()
    assert top["previous_action"].requires == ("history",)
    assert top["contact_forces"].requires == ("contacts",)
    assert top["dt"].requires == () and top["env_ids"].requires == ()
    with pytest.raises(KeyError):
        declared_paths("bogus")
    with pytest.raises(KeyError):
        namespace_requires("bogus")


def test_vocabulary_classes_are_declarations_not_containers():
    view = SensorStateView()
    with pytest.raises(AttributeError, match="'dof_pos' is a declared context name"):
        view.dof_pos
    with pytest.raises(AttributeError, match="declared context name"):
        view.dof_pos = torch.zeros(1)
    with pytest.raises(AttributeError, match="context namespace"):
        EnvContext().current
    # A name with no deploy fact reprs as its path and requires alone.
    assert repr(EnvContext.noisy.ground_heights) == (
        "FieldPath('noisy.ground_heights', requires=('localization', 'height_sensor'))"
    )
    assert repr(EnvContext.current.dof_forces) == "FieldPath('current.dof_forces')"
    # A name that carries a deploy host kind shows it after the path and requires.
    assert repr(EnvContext.noisy.anchor_pos) == (
        "FieldPath('noisy.anchor_pos', requires='localization', deploy='anchor_pos')"
    )
    assert repr(vars(EnvContext)["noisy"]) == "NestedField('noisy', requires='sensors')"
    assert repr(vars(EnvContext)["scene"]) == "NestedField('scene')"
    # requires= takes one name or several; anything else is refused.
    assert FieldPath(requires=("a", "b")).requires == ("a", "b")
    with pytest.raises(TypeError, match="requirement names"):
        FieldPath(requires=(1,))


def test_a_namespace_nested_inside_a_view_cannot_carry_requires():
    """Only a namespace hung on ``EnvContext`` is checked for its own
    ``requires=``; a tag on a namespace nested inside another view would be
    read by nothing, so building the paths refuses it instead of dropping it."""

    class Inner:
        value: torch.Tensor = FieldPath()

    class Outer:
        inner: Inner = NestedField(Inner, requires="something")

    outer = NestedField(Outer)
    outer.name = "outer"
    with pytest.raises(
        TypeError,
        match="'outer.inner': a nested namespace cannot carry requires=; put the "
        "tag on its names",
    ):
        outer.__get__(None)

    class Plain:
        inner: Inner = NestedField(Inner)

    plain = NestedField(Plain)
    plain.name = "plain"
    assert plain.__get__(None).inner.value.path == "plain.inner.value"


# =============================================================================
# Kernel bindings and sensed()
# =============================================================================


def _bindings(component):
    return component.get_bindings_dict()


def _params(component):
    return component.get_params()


def _sensed(component, source):
    return sensed({"c": component}, source)["c"]


def test_observation_kernels_bind_the_selected_source():
    """What each retired observation factory bound is its kernel's own default
    now, repointed at a sensor source by ``sensed``."""
    max_coords = _sensed(
        MdpComponent.from_kernel(
            compute_humanoid_max_coords_observations, local_obs=False
        ),
        "noisy",
    )
    assert _bindings(max_coords)["body_pos"] == "noisy.rigid_body_pos"
    assert _bindings(max_coords)["ground_height"] == "noisy.ground_heights"
    assert _bindings(max_coords)["anchor_body_index"] == "noisy.anchor_idx"
    assert _params(max_coords)["local_obs"] is False
    assert _params(max_coords)["body_contacts"] is None

    privileged_contacts = MdpComponent.from_kernel(
        compute_humanoid_max_coords_contact_observations
    )
    assert _bindings(privileged_contacts)["contact_forces"] == "contact_forces"
    # The simulator already selected the rows; don't subset them a second time.
    assert "contact_body_ids" not in _bindings(privileged_contacts)
    with pytest.raises(ValueError, match="only the simulator produces"):
        _sensed(privileged_contacts, "noisy")

    # Without the height the ground is unbound too, else it stays an input.
    no_height = MdpComponent.from_kernel(
        compute_humanoid_max_coords_observations,
        root_height_obs=False,
        ground_height=None,
    )
    assert "ground_height" not in _bindings(no_height)
    assert _params(no_height)["ground_height"] is None

    reduced = MdpComponent.from_kernel(
        compute_humanoid_reduced_coords_observations,
        root_height_obs=True,
        root_vel_obs=True,
        root_pos=EnvContext.current.root_pos,
        ground_height=EnvContext.current.ground_heights,
        root_rot=EnvContext.current.root_rot,
        root_vel=EnvContext.current.root_vel,
    )
    assert _bindings(reduced)["root_pos"] == "current.root_pos"
    assert _bindings(reduced)["root_vel"] == "current.root_vel"
    assert _bindings(reduced)["anchor_rot"] == "current.anchor_rot"
    clean_reduced = _sensed(
        MdpComponent.from_kernel(compute_humanoid_reduced_coords_observations),
        "clean",
    )
    assert _bindings(clean_reduced)["dof_pos"] == "clean.dof_pos"
    assert "root_pos" not in _bindings(clean_reduced)

    hist_max = _sensed(
        MdpComponent.from_kernel(
            compute_historical_max_coords_from_state, history_steps=[1, 3]
        ),
        "noisy",
    )
    assert (
        _bindings(hist_max)["historical_rigid_body_pos"]
        == "noisy_historical.rigid_body_pos"
    )
    assert _bindings(hist_max)["anchor_body_index"] == "noisy_historical.anchor_idx"
    # A step selection is normalised once, at construction: a list or an int N
    # becomes the tuple of 1-based steps the kernel receives.
    assert _params(hist_max)["history_steps"] == (1, 3)

    hist_reduced = MdpComponent.from_kernel(
        compute_historical_reduced_coords_from_state, history_steps=3
    )
    assert _bindings(hist_reduced)["historical_dof_pos"] == "historical.dof_pos"
    assert _params(hist_reduced)["history_steps"] == (1, 2, 3)

    # Command history starts at the most recent policy output, not at the
    # state-aligned slot one; the processed variant rebinds the history.
    previous = MdpComponent.from_kernel(
        compute_historical_actions_from_state,
        history_steps=3,
        historical_actions=EnvContext.processed_action_history,
    )
    assert _bindings(previous) == {"historical_actions": "processed_action_history"}
    assert _params(previous)["history_steps"] == (1, 2, 3)
    raw_previous = MdpComponent.from_kernel(
        compute_historical_actions_from_state, history_steps=1
    )
    assert _bindings(raw_previous) == {"historical_actions": "raw_action_history"}

    max_target = _sensed(
        MdpComponent.from_kernel(
            build_max_coords_target_poses,
            with_velocities=False,
            with_relative=False,
            future_steps=2,
        ),
        "noisy",
    )
    assert _bindings(max_target)["current_state_body_pos"] == "noisy.rigid_body_pos"
    assert _bindings(max_target)["anchor_body_index"] == "mimic.anchor_idx"
    assert _params(max_target)["future_steps"] == (1, 2)

    future_rel = _sensed(
        MdpComponent.from_kernel(
            build_max_coords_target_poses_future_rel, future_steps=4
        ),
        "clean",
    )
    assert _bindings(future_rel)["current_state_body_rot"] == "clean.rigid_body_rot"
    assert _params(future_rel)["future_steps"] == (1, 2, 3, 4)

    # The anchor position is bound only when the XY offset is actually computed
    # (the recipe binds it), so an encoder/IMU-only suite can serve targets
    # without an offset.
    offset_target = _sensed(
        MdpComponent.from_kernel(
            build_reduced_coords_target_poses,
            include_xy_offset=True,
            current_state_anchor_pos=EnvContext.current.anchor_pos,
        ),
        "noisy",
    )
    assert _bindings(offset_target)["current_state_anchor_pos"] == "noisy.anchor_pos"
    zeroed_target = _sensed(
        MdpComponent.from_kernel(
            build_reduced_coords_target_poses,
            include_xy_offset=True,
            zero_xy_offset=True,
        ),
        "noisy",
    )
    assert "current_state_anchor_pos" not in _bindings(zeroed_target)
    assert _params(zeroed_target)["zero_xy_offset"] is True
    plain_target = _sensed(
        MdpComponent.from_kernel(build_reduced_coords_target_poses), "noisy"
    )
    assert "current_state_anchor_pos" not in _bindings(plain_target)
    assert _bindings(plain_target)["mimic_ref_anchor_pos"] == "mimic.future_anchor_pos"

    deploy = _sensed(
        MdpComponent.from_kernel(
            build_deploy_target_poses, include_dof_vel=False, future_steps=[1, 4]
        ),
        "noisy",
    )
    assert _bindings(deploy)["current_anchor_rot"] == "noisy.anchor_rot"
    assert _params(deploy)["include_dof_vel"] is False

    # The task kernels read the anchor, not the root, and their control's names.
    target = MdpComponent.from_kernel(compute_target_obs)
    assert _bindings(target)["root_pos"] == "current.anchor_pos"
    steering = MdpComponent.from_kernel(compute_steering_obs)
    assert _bindings(steering)["root_rot"] == "current.anchor_rot"
    path = MdpComponent.from_kernel(compute_path_obs)
    assert _bindings(path)["traj_samples"] == "path.traj_samples"


def test_odometry_kernel_estimates_position_but_uses_true_heading():
    """The odometer inputs are keyword-only sensor names a recipe binds to its
    source; the heading keeps the kernel's default, simulator truth, so the
    component is written unsensed."""
    noisy = MdpComponent.from_kernel(
        compute_odom_offset_local,
        odom_disp_start=EnvContext.noisy.odom_disp_start,
        odom_start_xy=EnvContext.noisy.odom_start_xy,
        odom_start_heading_inv=EnvContext.noisy.odom_start_heading_inv,
    )
    assert noisy.get_compute_func() is compute_odom_offset_local
    bindings = _bindings(noisy)
    assert bindings["odom_disp_start"] == "noisy.odom_disp_start"
    # The offset is expressed in the true heading frame, as before #124.
    assert bindings["current_state_anchor_rot"] == "current.anchor_rot"
    assert bindings["odom_start_xy"] == "noisy.odom_start_xy"
    assert bindings["odom_start_heading_inv"] == "noisy.odom_start_heading_inv"
    # Only the reference anchor is needed, never the full reference body array.
    assert bindings["ref_anchor_pos"] == "mimic.ref_anchor_pos"
    assert "ref_rigid_body_pos" not in bindings
    assert _params(noisy) == {"w_last": True}

    clean = MdpComponent.from_kernel(
        compute_odom_offset_local,
        odom_disp_start=EnvContext.clean.odom_disp_start,
        odom_start_xy=EnvContext.clean.odom_start_xy,
        odom_start_heading_inv=EnvContext.clean.odom_start_heading_inv,
    )
    assert _bindings(clean)["odom_disp_start"] == "clean.odom_disp_start"
    assert _bindings(clean)["current_state_anchor_rot"] == "current.anchor_rot"


def test_reward_kernels_carry_the_bindings_their_factories_used_to_add():
    """Each retired reward factory's bindings are its kernel's own defaults now;
    ``from_kernel`` records the plain defaults a recipe may override."""
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards import regularization, task, tracking
    from protomotions.envs.rewards.meta import RewardMeta

    def bound(kernel, **overrides):
        return MdpComponent.from_kernel(
            kernel, meta=RewardMeta(weight=1.0), **overrides
        )

    rate = bound(regularization.compute_action_rate_l2)
    assert _bindings(rate) == {
        "raw_action": "current_action",
        "previous_raw_action": "previous_action",
    }
    smooth = bound(regularization.compute_action_smoothness)
    assert _bindings(smooth) == {
        "current_processed_action": "current_processed_action",
        "previous_processed_action": "previous_processed_action",
    }

    # The tracking rewards read one current quantity and its reference.
    for kernel, quantity in [
        (tracking.compute_gt_rew, "rigid_body_pos"),
        (tracking.compute_gr_rew, "rigid_body_rot"),
        (tracking.compute_gv_rew, "rigid_body_vel"),
        (tracking.compute_gav_rew, "rigid_body_ang_vel"),
    ]:
        assert _bindings(bound(kernel)) == {
            f"current_{quantity}": f"current.{quantity}",
            f"ref_{quantity}": f"mimic.ref_state.{quantity}",
        }
    # The height reward reads the declared anchor on both sides, not body 0.
    assert _bindings(bound(tracking.compute_rh_rew)) == {
        "anchor_pos": "current.anchor_pos",
        "ref_anchor_pos": "mimic.ref_anchor_pos",
    }
    assert _params(bound(tracking.compute_gt_rew)) == {"coefficient": -100.0}
    assert _params(bound(tracking.compute_gt_rew, coefficient=-25.0)) == {
        "coefficient": -25.0
    }

    # The anchor-relative kernels read the published anchor slices on both
    # sides; none takes the anchor index and slices the reference itself.
    rel = bound(tracking.compute_gt_rel_rew, body_indices=[0, 2])
    assert _bindings(rel)["current_anchor_pos"] == "current.anchor_pos"
    assert _bindings(rel)["ref_anchor_pos"] == "mimic.ref_anchor_pos"
    assert _bindings(rel)["ref_anchor_rot"] == "mimic.ref_anchor_rot"
    assert "anchor_idx" not in _bindings(rel)
    assert _params(rel) == {"coefficient": -100.0, "body_indices": [0, 2]}

    anchor_xy = bound(tracking.compute_anchor_xy_rew, coefficient=-3.0)
    assert _bindings(anchor_xy) == {
        "current_anchor_pos": "current.anchor_pos",
        "ref_anchor_pos": "mimic.ref_anchor_pos",
    }
    assert _params(anchor_xy) == {"coefficient": -3.0}

    power = bound(regularization.compute_pow_rew, use_torque_squared=True)
    assert _bindings(power)["dof_forces"] == "current.dof_forces"
    assert _params(power) == {"use_torque_squared": True}

    contact_match = bound(regularization.compute_contact_match_rew)
    assert _bindings(contact_match) == {
        "contact_forces": "contact_forces",
        "ref_contacts": "mimic.ref_state.rigid_body_contacts",
        "contact_body_ids": "contact_body_ids",
    }
    force_change = bound(regularization.compute_contact_force_change_rew)
    assert (
        _bindings(force_change)["prev_contact_force_magnitudes"]
        == "prev_contact_force_magnitudes"
    )
    assert _params(force_change) == {"threshold": 30.0}

    assert (
        _bindings(bound(task.compute_target_rew))["tar_proximity_threshold"]
        == "target.tar_proximity_threshold"
    )
    assert _bindings(bound(task.compute_heading_velocity_rew))["dt"] == "dt"
    assert (
        _bindings(bound(task.compute_path_following_rew))["height_conditioned"]
        == "path.height_conditioned"
    )

    for kernel in [
        tracking.compute_global_anchor_pos_rew,
        tracking.compute_global_anchor_ori_rew,
        tracking.compute_relative_body_pos_rew,
        tracking.compute_relative_body_ori_rew,
        tracking.compute_global_body_lin_vel_rew,
        tracking.compute_global_body_ang_vel_rew,
    ]:
        component = bound(kernel, sigma=1.3)
        assert _params(component)["sigma"] == 1.3
        assert any(path.startswith("mimic.") for path in _bindings(component).values())
        assert "anchor_idx" not in _bindings(component)
        assert component.meta.weight == 1.0


def test_termination_and_metric_kernels_bind_metadata_and_wrappers():
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.terminations import tracking
    from protomotions.envs.terminations.base import named_contact_fall_termination
    from protomotions.envs.terminations.rule import TerminationRule
    from protomotions.envs.terminations.task import path_distance

    # The tracking termination is a rule over the max-joint-error metric.
    tracking_term = TerminationRule(
        MdpComponent.from_kernel(tracking.max_body_pos_error), 0.6
    )
    assert (
        _bindings(tracking_term.component)["current_rigid_body_pos"]
        == "current.rigid_body_pos"
    )
    assert tracking_term.threshold == 0.6 and tracking_term.fail_above

    fall = MdpComponent.from_kernel(
        named_contact_fall_termination,
        required_keys={
            "contact_body_index": ["torso"],
            "contact_partner_index": ["terrain"],
        },
        body_ids=torch.tensor([1]),
        body_names=["torso"],
        partner_names=["terrain"],
        termination_height=0.2,
    )
    assert _bindings(fall)["progress_buf"] == "progress_buf"
    assert _params(fall)["termination_height"] == 0.2
    # A boolean kernel's rule takes no threshold.
    assert TerminationRule(fall).threshold is None

    for kernel in [
        tracking.mean_body_pos_error,
        tracking.max_body_pos_error,
        tracking.mean_body_rot_error,
        tracking.anchor_pos_error_value,
        tracking.anchor_ori_error_value,
        tracking.relative_body_pos_max_error,
        tracking.anchor_height_error_value,
    ]:
        metric = MdpComponent.from_kernel(kernel)
        # Every input is a context reading; the threshold is the rule's.
        assert _params(metric) == {}
        assert "anchor_idx" not in _bindings(metric)
        rule = TerminationRule(metric, 0.75)
        assert rule.threshold == 0.75
        assert rule.component is metric

    path_metric = MdpComponent.from_kernel(path_distance, min_progress=3)
    assert _bindings(path_metric)["target_pos"] == "path.tar_pos"
    assert _bindings(path_metric)["progress_buf"] == "progress_buf"
    assert _params(path_metric) == {"min_progress": 3}
    distance = path_metric.get_compute_func()(
        head_pos=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        target_pos=torch.zeros(2, 3),
        progress_buf=torch.tensor([10, 10]),
        min_progress=3,
    )
    assert torch.equal(
        TerminationRule(path_metric, 1.5).compare(distance),
        torch.tensor([False, True]),
    )

    from protomotions.envs.terminations.task import (
        steering_direction_alignment,
        steering_speed_error,
    )

    speed = MdpComponent.from_kernel(steering_speed_error)
    alignment = MdpComponent.from_kernel(steering_direction_alignment)
    assert _bindings(speed)["prev_root_pos"] == "steering.prev_root_pos"
    assert _bindings(alignment)["tar_dir"] == "steering.tar_dir"
    assert _params(speed) == {} and _params(alignment) == {}
    kwargs = dict(
        root_pos=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        prev_root_pos=torch.zeros(2, 3),
        dt=1.0,
    )
    too_fast = TerminationRule(speed, 0.5).compare(
        speed.get_compute_func()(tar_speed=torch.tensor([1.0, 1.0]), **kwargs)
    )
    off_course = TerminationRule(alignment, 0.7, fail_above=False).compare(
        alignment.get_compute_func()(
            tar_dir=torch.tensor([[1.0, 0.0], [1.0, 0.0]]), **kwargs
        )
    )
    assert torch.equal(too_fast, torch.tensor([False, False]))
    assert torch.equal(off_course, torch.tensor([False, True]))
