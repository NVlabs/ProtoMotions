# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene-box observations, the placeholder scene-box control and their export."""

import io
import math
from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.context_views import declared_paths, deploy_for
from protomotions.envs.control.scene_boxes_control import (
    SceneBoxesControl,
    SceneBoxesControlConfig,
)
from protomotions.envs.control.task_inputs import TaskInputs
from protomotions.envs.obs.scene_boxes import (
    ROOT_BODY,
    compute_fk_scene_box_corners,
    compute_scene_box_mask,
    observed_object_pose,
    scene_box_corners_in_heading_frame,
)
from protomotions.utils import rotations

IDENTITY = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
# The deployment kind of every name, by hand: renaming either side changes
# what a host reads in the sidecar.
SCENE_KINDS = {
    "scene_boxes.centers": "scene_box_centers",
    "scene_boxes.extents": "scene_box_extents",
    "scene_boxes.valid": "scene_box_valid",
    "scene_boxes.visible": "scene_object_visible",
    "scene_boxes.object_pos": "scene_object_pos",
    "scene_boxes.object_rot": "scene_object_rot",
    "scene_boxes.position_offset": "scene_position_offset",
    "scene_boxes.yaw_offset": "scene_yaw_offset",
}
NO_SCENE = SimpleNamespace(has_scene_objects=False)
SCENE_PATHS = (
    "scene_boxes.centers",
    "scene_boxes.extents",
    "scene_boxes.valid",
    "scene_boxes.visible",
    "scene_boxes.object_pos",
    "scene_boxes.object_rot",
    "scene_boxes.position_offset",
    "scene_boxes.yaw_offset",
)


def _inputs(num_envs: int, reference=NO_SCENE) -> TaskInputs:
    return TaskInputs(
        num_envs=num_envs,
        device=torch.device("cpu"),
        dt=0.02,
        robot=None,
        clock=None,
        reference=reference,
        ground_height=None,
        generator=None,
        headless=True,
        ui=None,
        viewer_reset=None,
        restart_episode=None,
        teleport=None,
    )


def _yaw_quat(yaw: float) -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]])


def _random_unit_quats(n: int, generator: torch.Generator) -> torch.Tensor:
    q = torch.randn(n, 4, generator=generator)
    return q / q.norm(dim=-1, keepdim=True)


def _random_scene(
    num_envs: int, generator: torch.Generator, objects: int = 1, boxes: int = 2
):
    """A random scene at the stream's shapes, with one padding box per object."""
    valid = torch.ones(num_envs, objects, boxes, dtype=torch.bool)
    valid[..., -1] = False
    return dict(
        object_pos=torch.randn(num_envs, objects, 3, generator=generator),
        object_rot=_random_unit_quats(num_envs * objects, generator).reshape(
            num_envs, objects, 4
        ),
        centers=torch.randn(num_envs, objects, boxes, 3, generator=generator),
        extents=torch.rand(num_envs, objects, boxes, 3, generator=generator) + 0.1,
        valid=valid,
        position_offset=0.1 * torch.randn(num_envs, 3, generator=generator),
        yaw_offset=0.5 * torch.randn(num_envs, generator=generator),
    )


# --- the corners kernel ------------------------------------------------------


def test_corners_keep_the_object_orientation_and_zero_the_padding_box():
    yawed = _yaw_quat(math.pi / 2).reshape(1, 1, 4)
    corners = scene_box_corners_in_heading_frame(
        object_pos=torch.zeros(1, 1, 3),
        object_rot=yawed,
        centers=torch.zeros(1, 1, 2, 3),
        extents=torch.tensor([[[[2.0, 4.0, 6.0], [2.0, 4.0, 6.0]]]]),
        valid=torch.tensor([[[True, False]]]),
        position_offset=torch.zeros(1, 3),
        yaw_offset=torch.zeros(1),
        root_pos=torch.zeros(1, 3),
        root_rot=IDENTITY,
    )
    assert corners.shape == (1, 2, 24)
    per_corner = corners.reshape(1, 2, 8, 3)
    # Corner (-1, -2, -3) of the object, yawed by 90 degrees: (2, -1, -3).
    torch.testing.assert_close(per_corner[0, 0, 0], torch.tensor([2.0, -1.0, -3.0]))
    # Lexicographic sign order: the last corner is (+, +, +) -> (-2, 1, 3).
    torch.testing.assert_close(per_corner[0, 0, 7], torch.tensor([-2.0, 1.0, 3.0]))
    assert per_corner[0, 1].count_nonzero() == 0


def test_offsets_bias_the_observed_object_pose():
    corners = scene_box_corners_in_heading_frame(
        object_pos=torch.tensor([[[2.0, 3.0, 0.0]]]),
        object_rot=IDENTITY.reshape(1, 1, 4),
        centers=torch.tensor([[[[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]]]]),
        extents=torch.tensor([[[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0]]]]),
        valid=torch.tensor([[[True, False]]]),
        position_offset=torch.tensor([[0.1, -0.2, 0.05]]),
        yaw_offset=torch.tensor([math.pi / 2]),
        root_pos=torch.zeros(1, 3),
        root_rot=IDENTITY,
    ).reshape(1, 2, 8, 3)
    # Local corner (-0.5, -1, 0) yawed by 90 degrees is (1, -0.5, 0); plus the
    # biased object position (2.1, 2.8, 0.05).
    torch.testing.assert_close(corners[0, 0, 0], torch.tensor([3.1, 2.3, 0.05]))


def test_corners_are_in_the_root_heading_frame_and_ignore_root_tilt():
    # A unit box centred one metre to the robot's left; the robot faces +y.
    scene = dict(
        object_pos=torch.tensor([[[1.0, 1.0, 0.0]]]),
        object_rot=IDENTITY.reshape(1, 1, 4),
        centers=torch.zeros(1, 1, 1, 3),
        extents=torch.ones(1, 1, 1, 3),
        valid=torch.ones(1, 1, 1, dtype=torch.bool),
        position_offset=torch.zeros(1, 3),
        yaw_offset=torch.zeros(1),
    )
    root_pos = torch.tensor([[1.0, 0.0, 0.5]])
    heading = _yaw_quat(math.pi / 2)
    flat = scene_box_corners_in_heading_frame(
        **scene, root_pos=root_pos, root_rot=heading
    )
    centre = flat.reshape(1, 1, 8, 3).mean(dim=2)[0, 0]
    # World (0, 1, -0.5) relative to the root, turned into the +y heading: ahead by one.
    torch.testing.assert_close(centre, torch.tensor([1.0, 0.0, -0.5]))
    pitched = rotations.quat_mul(
        heading,
        rotations.quat_from_angle_axis(
            torch.tensor([0.5]), torch.tensor([[0.0, 1.0, 0.0]]), True
        ),
        True,
    )
    tilted = scene_box_corners_in_heading_frame(
        **scene, root_pos=root_pos, root_rot=pitched
    )
    torch.testing.assert_close(tilted, flat)


def test_mask_is_visible_and_valid_per_box_flattened():
    mask = compute_scene_box_mask(
        visible=torch.tensor([[[True]], [[False]]]),
        valid=torch.tensor([[[True, False]], [[True, True]]]),
    )
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, False], [False, False]]


def _g1_fk():
    from protomotions.robot_configs.factory import robot_config
    from protomotions.utils.baked_fk import BakedAnchorFK

    cfg = robot_config("g1")
    bodies = torch.tensor(cfg.body_indices(*cfg.trackable_bodies_subset))
    return cfg, BakedAnchorFK(cfg.kinematic_info, cfg.anchor_body_index, bodies)


def _g1_readings(cfg, num_envs: int, generator: torch.Generator):
    return dict(
        dof_pos=0.4
        * torch.randn(num_envs, cfg.kinematic_info.num_dofs, generator=generator),
        anchor_pos=torch.randn(num_envs, 3, generator=generator),
        anchor_rot=_random_unit_quats(num_envs, generator),
    )


def test_deployable_corners_use_the_root_the_state_publisher_names_on_g1():
    pytest.importorskip("dm_control")
    from protomotions.envs.state import BodyState
    from protomotions.envs.state_names import body_state_names

    cfg, fk = _g1_fk()
    generator = torch.Generator().manual_seed(7)
    readings = _g1_readings(cfg, 5, generator)
    scene = _random_scene(5, generator)
    # The oracle: the body poses the FK places (pinned against MuJoCo in
    # test_baked_fk_velocities), named the way the state publisher names every
    # source's bodies, so "root" below is whatever the environment publishes as
    # current.root_pos / root_rot, not this module's ROOT_BODY.
    body_pos, body_rot = fk.world_body_poses(**readings)
    names = body_state_names(
        BodyState(
            rigid_body_rot=body_rot,
            rigid_body_ang_vel=torch.zeros_like(body_pos),
            dof_pos=readings["dof_pos"],
            dof_vel=torch.zeros_like(readings["dof_pos"]),
            rigid_body_pos=body_pos,
        ),
        cfg.anchor_body_index,
    )
    expected = scene_box_corners_in_heading_frame(
        **scene, root_pos=names["root_pos"], root_rot=names["root_rot"]
    )
    deployable = compute_fk_scene_box_corners(
        **scene, **readings, anchor_body_index=cfg.anchor_body_index, fk=fk
    )
    torch.testing.assert_close(deployable, expected, atol=1e-6, rtol=0)
    assert torch.equal(names["root_pos"], body_pos[:, ROOT_BODY])
    # The FK placed the anchor where the readings said it is.
    torch.testing.assert_close(
        names["anchor_pos"], readings["anchor_pos"], atol=1e-5, rtol=0
    )


def test_deployable_corners_refuse_a_kernel_anchored_on_another_body():
    pytest.importorskip("dm_control")
    from protomotions.utils.baked_fk import BakedAnchorFK

    cfg, fk = _g1_fk()
    assert cfg.anchor_body_index != ROOT_BODY  # torso_link, not the pelvis
    generator = torch.Generator().manual_seed(3)
    readings = _g1_readings(cfg, 2, generator)
    scene = _random_scene(2, generator)
    bodies = torch.tensor(cfg.body_indices(*cfg.trackable_bodies_subset))
    wrong_anchor = BakedAnchorFK(cfg.kinematic_info, ROOT_BODY, bodies)
    with pytest.raises(ValueError, match="anchored on body 0 but the state"):
        compute_fk_scene_box_corners(
            **scene,
            **readings,
            anchor_body_index=cfg.anchor_body_index,
            fk=wrong_anchor,
        )
    # The same readings through the right kernel are fine.
    compute_fk_scene_box_corners(
        **scene, **readings, anchor_body_index=cfg.anchor_body_index, fk=fk
    )


def test_deployable_corners_trace_to_a_static_graph():
    pytest.importorskip("dm_control")
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from protomotions.robot_configs.factory import robot_config
    from protomotions.utils.baked_fk import BakedAnchorFK

    cfg = robot_config("g1")
    ki = cfg.kinematic_info
    bodies = torch.tensor(cfg.body_indices(*cfg.trackable_bodies_subset))
    fk = BakedAnchorFK(ki, cfg.anchor_body_index, bodies)
    generator = torch.Generator().manual_seed(11)
    scene = _random_scene(1, generator)
    scene["valid"][:] = True
    state = dict(
        dof_pos=0.3 * torch.randn(1, ki.num_dofs, generator=generator),
        anchor_pos=torch.randn(1, 3, generator=generator),
        anchor_rot=_random_unit_quats(1, generator),
    )
    names = [*scene, *state]
    values = [*scene.values(), *state.values()]

    class Wrapped(torch.nn.Module):
        def forward(self, *args):
            kwargs = dict(zip(names, args))
            return compute_fk_scene_box_corners(
                **kwargs, anchor_body_index=cfg.anchor_body_index, fk=fk
            )

    buffer = io.BytesIO()
    torch.onnx.export(
        Wrapped(),
        tuple(values),
        buffer,
        input_names=names,
        output_names=["corners"],
        opset_version=17,
        dynamo=False,
    )
    model = onnx.load_from_string(buffer.getvalue())
    ops = {node.op_type for node in model.graph.node}
    assert not ops & {"NonZero", "Loop", "If"}
    session = ort.InferenceSession(
        buffer.getvalue(), providers=["CPUExecutionProvider"]
    )
    feed = {name: value.numpy() for name, value in zip(names, values)}
    (traced,) = session.run(None, feed)
    expected = compute_fk_scene_box_corners(
        **scene, **state, anchor_body_index=cfg.anchor_body_index, fk=fk
    )
    torch.testing.assert_close(torch.from_numpy(traced), expected, atol=1e-5, rtol=0)


# --- the placeholder control -------------------------------------------------


def test_placeholder_control_publishes_exactly_the_declared_names_at_the_graph_shapes():
    control = SceneBoxesControl(SceneBoxesControlConfig(), _inputs(3))
    published = control.publish(None)
    assert set(published) == set(declared_paths("scene_boxes")) == set(SCENE_PATHS)
    shapes = {name: tuple(value.shape) for name, value in published.items()}
    assert shapes == {
        "scene_boxes.centers": (3, 1, 2, 3),
        "scene_boxes.extents": (3, 1, 2, 3),
        "scene_boxes.valid": (3, 1, 2),
        "scene_boxes.visible": (3, 1, 1),
        "scene_boxes.object_pos": (3, 1, 3),
        "scene_boxes.object_rot": (3, 1, 4),
        "scene_boxes.position_offset": (3, 3),
        "scene_boxes.yaw_offset": (3,),
    }
    assert published["scene_boxes.valid"].dtype == torch.bool
    assert published["scene_boxes.visible"].dtype == torch.bool
    assert not published["scene_boxes.valid"].any()
    assert not published["scene_boxes.visible"].any()
    torch.testing.assert_close(
        published["scene_boxes.object_rot"], IDENTITY.expand(3, 1, 4).clone()
    )
    assert control.advance(None) == {}
    subset = control.publish(torch.tensor([2]))
    assert all(value.shape[0] == 1 for value in subset.values())
    # Over the placeholder the policy sees no boxes: zero corners, all-false mask.
    kwargs = {path.split(".", 1)[1]: value for path, value in published.items()}
    visible = kwargs.pop("visible")
    corners = scene_box_corners_in_heading_frame(
        **kwargs, root_pos=torch.zeros(3, 3), root_rot=IDENTITY.expand(3, 4)
    )
    assert corners.shape == (3, 2, 24) and not corners.any()
    assert not compute_scene_box_mask(visible, kwargs["valid"]).any()


def test_capacity_sizes_the_stream_and_must_be_positive():
    control = SceneBoxesControl(
        SceneBoxesControlConfig(max_objects=2, max_boxes_per_object=3), _inputs(1)
    )
    assert tuple(control.publish(None)["scene_boxes.centers"].shape) == (1, 2, 3, 3)
    with pytest.raises(ValueError, match="at least one object"):
        SceneBoxesControlConfig(max_objects=0)
    with pytest.raises(ValueError, match="at least one object"):
        SceneBoxesControlConfig(max_boxes_per_object=0)


# --- export plumbing ---------------------------------------------------------


def test_every_scene_name_has_a_robot_kind_with_the_scene_provider():
    from deployment.contract import (
        ELEMENT_LABELS,
        SCENE,
        element_labels,
        host_can_feed,
        provider_for_path,
    )

    published = SceneBoxesControl(SceneBoxesControlConfig(), _inputs(1)).publish(None)
    for path in SCENE_PATHS:
        kind = deploy_for(path)
        assert kind == SCENE_KINDS[path] and kind in ELEMENT_LABELS, path
        assert provider_for_path(path) == SCENE == "scene"
        assert host_can_feed("robot", kind) and host_can_feed("simulation", kind)
        # The label rule labels the published tensor's own coordinate axis: xyz
        # for a position or box dimension, xyzw for the quaternion, nothing for
        # the flags and the scalar yaw.
        labels = element_labels(kind, ("j",), ("b",), "b", published[path].shape[-1])
        width = published[path].shape[-1]
        if path in (
            "scene_boxes.valid",
            "scene_boxes.visible",
            "scene_boxes.yaw_offset",
        ):
            assert labels == (), path
        else:
            assert labels == (("x", "y", "z", "w")[:width],), path


def test_placeholder_publishes_the_pinned_scene_graph_inputs():
    # The 2026-10-25 scene-capable graph's eight scene inputs, as the native
    # contract pins them: same keys, trailing shapes and dtypes, and the one
    # name rule gives the graph's input names.
    from deployment.export_to_onnx import sanitize_onnx_name
    from deployment.native.examples.metatron_v1_contract import (
        BOOL_INPUTS,
        INPUT_SHAPES,
    )

    pinned = {key for key in INPUT_SHAPES if key.startswith("scene_boxes.")}
    assert pinned == set(SCENE_PATHS)
    published = SceneBoxesControl(SceneBoxesControlConfig(), _inputs(1)).publish(None)
    for path in SCENE_PATHS:
        assert tuple(published[path].shape) == (1, *INPUT_SHAPES[path]), path
        assert (published[path].dtype == torch.bool) == (path in BOOL_INPUTS), path
        assert sanitize_onnx_name(path) == "scene_boxes_" + path.split(".", 1)[1]


def _g1_scene_sample_context(with_control: bool):
    from deployment.sample_context import build_sample_context
    from protomotions.envs.control.masked_mimic_control import MaskedMimicControlConfig
    from protomotions.envs.sensors.config import (
        IMUSensor,
        JointEncoderSensor,
        LocalizationSensor,
        SensorConfig,
    )
    from protomotions.robot_configs.factory import robot_config
    from protomotions.utils.baked_fk import BakedAnchorFK
    from examples.experiments.masked_mimic.deployable_g1_observations import (
        deployable_scene_box_observations,
    )

    robot = robot_config("g1")
    ki = robot.kinematic_info
    bodies = torch.tensor(robot.body_indices(*robot.trackable_bodies_subset))
    fk = BakedAnchorFK(ki, robot.anchor_body_index, bodies)
    controls = {"masked_mimic": MaskedMimicControlConfig()}
    if with_control:
        controls["scene_boxes"] = SceneBoxesControlConfig()
    env = SimpleNamespace(
        sensors=SensorConfig(
            joint_encoders=JointEncoderSensor(pos_noise=0.01, vel_noise=0.01),
            imu=IMUSensor(quat_noise=0.01, gyro_noise=0.01),
            localization=LocalizationSensor(body_name=ki.body_names[0], noise=0.01),
        ),
        control_components=controls,
    )
    actor_obs = deployable_scene_box_observations(fk)
    context, _future, slot_names = build_sample_context(robot, env, actor_obs, "cpu")
    return actor_obs, context, slot_names


def test_recipe_helper_binds_the_sensor_source_and_the_fk_kernel():
    pytest.importorskip("dm_control")
    actor_obs, _context, _slots = _g1_scene_sample_context(with_control=True)
    corners = actor_obs["scene_box_corners"]
    assert corners.get_bindings_dict() == {
        "object_pos": "scene_boxes.object_pos",
        "object_rot": "scene_boxes.object_rot",
        "centers": "scene_boxes.centers",
        "extents": "scene_boxes.extents",
        "valid": "scene_boxes.valid",
        "position_offset": "scene_boxes.position_offset",
        "yaw_offset": "scene_boxes.yaw_offset",
        "dof_pos": "noisy.dof_pos",
        "anchor_pos": "noisy.anchor_pos",
        "anchor_rot": "noisy.anchor_rot",
        "anchor_body_index": "noisy.anchor_idx",
    }
    assert "fk" in corners.static_params
    assert actor_obs["scene_box_mask"].get_bindings_dict() == {
        "visible": "scene_boxes.visible",
        "valid": "scene_boxes.valid",
    }


def test_sample_context_publishes_the_scene_stream_only_with_its_control():
    pytest.importorskip("dm_control")
    from deployment.export_to_onnx import _build_samples

    actor_obs, context, slot_names = _g1_scene_sample_context(with_control=True)
    assert tuple(context["scene_boxes.centers"].shape) == (1, 1, 2, 3)
    assert tuple(context["scene_boxes.visible"].shape) == (1, 1, 1)
    assert tuple(context["scene_boxes.yaw_offset"].shape) == (1,)
    samples = _build_samples(actor_obs, context, slot_names)
    assert set(SCENE_PATHS) <= set(samples)
    assert {"noisy.dof_pos", "noisy.anchor_pos", "noisy.anchor_rot"} <= set(samples)

    actor_obs, context, slot_names = _g1_scene_sample_context(with_control=False)
    assert not any(path in context for path in SCENE_PATHS)
    with pytest.raises(ValueError, match="does not publish"):
        _build_samples(actor_obs, context, slot_names)


# --- in an environment ---------------------------------------------------------


def _fake_scene_env(num_envs: int = 3):
    """The real BaseEnv on the kinematic test double with the placeholder
    control and the recipe helper's two observations bound to its sensors."""
    from protomotions.components.pose_lib import extract_kinematic_info
    from protomotions.tests.support.sensor_simulator import (
        CHAIN_XML,
        make_sensor_test_env,
    )
    from protomotions.utils.baked_fk import BakedAnchorFK
    from examples.experiments.masked_mimic.deployable_g1_observations import (
        deployable_scene_box_observations,
    )

    # The double's robot is anchored on body 1 (SensorTestRobot); so is this FK.
    fk = BakedAnchorFK(extract_kinematic_info(str(CHAIN_XML)), 1, torch.tensor([3]))
    return make_sensor_test_env(
        num_envs=num_envs,
        observations=deployable_scene_box_observations(fk),
        control_components={"scene_boxes": SceneBoxesControlConfig()},
    )


def test_placeholder_runs_in_an_environment_for_every_env_and_a_reset_subset():
    env = _fake_scene_env()
    try:
        env.reset()
        obs, *_ = env.step(torch.zeros(env.num_envs, 2))
        assert obs["scene_box_corners"].shape == (3, 2, 24)
        assert not obs["scene_box_corners"].any()
        assert obs["scene_box_mask"].shape == (3, 2)
        assert obs["scene_box_mask"].dtype == torch.bool
        assert not obs["scene_box_mask"].any()
        context = env.context
        assert set(SCENE_PATHS) <= set(context)
        assert tuple(context["scene_boxes.visible"].shape) == (3, 1, 1)
        assert context["noisy.anchor_idx"] == 1
        # A row-subset reset publishes the stream for just those rows (the
        # build checks that subset against the declared names too) and
        # recomputes their observations; the returned buffers stay env-wide.
        after_reset, _ = env.reset(torch.tensor([1]))
        assert after_reset["scene_box_corners"].shape == (3, 2, 24)
        assert not after_reset["scene_box_corners"].any()
        assert after_reset["scene_box_mask"].shape == (3, 2)
    finally:
        env.close()


def test_config_check_accepts_scene_bindings_only_with_the_control():
    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.envs.context_check import check_bindings_against_config

    env = _fake_scene_env(num_envs=1)
    try:
        robot = env.robot_config
        config = env.config
        with_control = check_bindings_against_config(config, robot, None)
        assert with_control.errors == []
        assert not any("scene_boxes" in path for path in with_control.unknown)
        without = EnvConfig(
            sensors=config.sensors,
            observation_components={
                key: config.observation_components[key]
                for key in ("scene_box_corners", "scene_box_mask")
            },
        )
        refused = check_bindings_against_config(without, robot, None)
    finally:
        env.close()
    assert refused.errors
    assert all(
        "no control component publishes 'scene_boxes'" in e for e in refused.errors
    )
    assert any("scene_boxes.centers" in e for e in refused.errors)


def test_placeholder_refuses_a_scene_library_with_objects():
    from protomotions.envs.base_env.reset.placed_reference import PlacedReference

    with pytest.raises(ValueError, match="scene library has objects"):
        SceneBoxesControl(
            SceneBoxesControlConfig(),
            _inputs(1, reference=SimpleNamespace(has_scene_objects=True)),
        )
    # The env's reference answers from the scene library's scene count.
    motion = SimpleNamespace(motion_manager=None, motion_lib=None)
    for scenes, expected in ((0, False), (2, True)):
        reference = PlacedReference(
            motion, None, SimpleNamespace(num_scenes=lambda: scenes), 0.0, 1, "cpu"
        )
        assert reference.has_scene_objects is expected


def test_yaw_bias_turns_the_object_about_the_vertical_after_its_own_rotation():
    # An object rolled 90 degrees about x, then the observed-pose bias yaws it
    # 90 degrees about world z: the body x axis (untouched by the roll) ends on
    # world +y and the body y axis (sent to +z by the roll) stays on +z. The
    # other composition order would send the body x axis to +z.
    roll = torch.tensor([[[math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]]])
    pos, rot = observed_object_pose(
        torch.tensor([[[1.0, 2.0, 3.0]]]),
        roll,
        torch.tensor([[0.5, 0.0, 0.0]]),
        torch.tensor([math.pi / 2]),
    )
    torch.testing.assert_close(pos, torch.tensor([[[1.5, 2.0, 3.0]]]))
    axes = torch.eye(3)[:2]
    turned = rotations.quat_rotate(rot.reshape(1, 4).expand(2, 4), axes, True)
    torch.testing.assert_close(
        turned, torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), atol=1e-6, rtol=0
    )
