# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from protomotions.components.pose_lib import (
    extract_kinematic_info,
    fk_batch_mjcf_with_velocities,
)
from data.scripts.from_soma_to_anatomical_soma.soma23_to_anatomical_soma_retargeting import (
    CANONICAL_FRAME,
    SOMA_TO_CANONICAL,
    canonical_soma_poses,
    canonical_soma_scenes,
    hinge_seed,
    semantic_names,
)


ASSET = Path(__file__).resolve().parents[1] / "data/assets/mjcf/anatomic_soma.xml"


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(ASSET))


def test_asset_contract(model):
    tree = ET.parse(ASSET)
    assert (
        tree.find("custom/text[@name='coordinate_system']").get("data")
        == CANONICAL_FRAME
    )
    bodies = list(tree.find("worldbody").iter("body"))
    assert len(bodies) == 53
    assert model.nq == 59 and model.nv == 58 and model.nu == 52
    assert len(semantic_names(ASSET)) == 23
    assert all(len(b.findall("joint")) == 1 for b in bodies)
    assert all(b.find("joint").get("type") == "hinge" for b in bodies[1:])
    assert sum(b.find("geom") is None for b in bodies) == 30
    assert (model.body_mass[1:] > 0).all()
    assert (model.body_inertia[1:] > 0).all()
    assert (model.dof_armature[6:] > 0).all()
    assert model.ngeom == 39
    assert sum(model.geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE) == 38
    assert sum(model.geom_type == mujoco.mjtGeom.mjGEOM_BOX) == 1
    neck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Neck2_yaw")
    np.testing.assert_allclose(model.jnt_range[neck], np.deg2rad([-30, 30]))


def test_pose_lib_fk_matches_mujoco(model):
    kin = extract_kinematic_info(str(ASSET))
    rng = np.random.default_rng(4)
    qpos = np.zeros((24, model.nq))
    qpos[:, :3] = rng.uniform(-1, 1, (24, 3))
    qpos[:, 3:7] = Rotation.random(24, random_state=rng).as_quat()[:, [3, 0, 1, 2]]
    qpos[:, 7:] = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1], (24, 52))
    state = fk_batch_mjcf_with_velocities(
        kin, torch.tensor(qpos, dtype=torch.float32), compute_velocities=False
    )
    data = mujoco.MjData(model)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            state.rigid_body_pos[frame], data.xpos[1:], atol=1e-6
        )
        expected = Rotation.from_quat(data.xquat[1:, [1, 2, 3, 0]])
        actual = Rotation.from_quat(state.rigid_body_rot[frame].numpy())
        assert (expected.inv() * actual).magnitude().max() < 2e-6


def test_seed_recovers_small_admissible_motion(model):
    rng = np.random.default_rng(5)
    q = rng.uniform(-0.05, 0.05, (12, 52))
    q = np.clip(q, model.jnt_range[1:, 0], model.jnt_range[1:, 1])
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in semantic_names(ASSET)
    ]
    rotations = []
    data = mujoco.MjData(model)
    for row in q:
        data.qpos[7:] = row
        mujoco.mj_forward(model, data)
        rotations.append(data.xquat[ids][:, [1, 2, 3, 0]].copy())
    seed, names = hinge_seed(ASSET, np.array(rotations))
    assert names == extract_kinematic_info(str(ASSET)).dof_names
    np.testing.assert_allclose(seed, q, atol=1e-6)


def test_robot_config_uses_consistent_motor_classes(model):
    from protomotions.robot_configs.factory import robot_config

    config = robot_config("anatomic_soma")
    assert config.kinematic_info.num_dofs == 52
    assert config.asset.self_collisions
    assert config.semantic_forward_axis_xy == (1.0, 0.0)
    assert config.kinematic_info.body_names[0] == "Hips"
    assert set(config.trackable_bodies_subset) <= set(semantic_names(ASSET))
    for joint in ET.parse(ASSET).iter("joint"):
        if joint.get("type") != "hinge":
            continue
        control = config.control.control_info[joint.get("name")]
        assert (
            sum(
                re.fullmatch(pattern, joint.get("name")) is not None
                for pattern in config.control.override_control_info
            )
            == 1
        )
        assert control.stiffness == pytest.approx(
            control.armature * (20 * math.pi) ** 2
        )
        assert control.damping == pytest.approx(4 * control.armature * 20 * math.pi)
        assert control.effort_limit > 0
        assert control.velocity_limit in (30.0, 33.0, 48.0, 55.5)


@pytest.mark.parametrize(
    "joint,armature,effort,velocity",
    [
        ("Chest_x", 1.25 * 0.010177520, 110.0, 48.0),
        ("LeftLeg_x", 1.25 * 0.010177520, 110.0, 48.0),
        ("RightLeg_y", 0.03137740625, 173.75, 30.0),
        ("LeftShin_x", 0.03137740625, 173.75, 30.0),
        ("RightFoot_x", 2 * 0.00451215625, 62.5, 55.5),
        ("LeftArm_x", 0.00451215625, 31.25, 55.5),
        ("RightForeArm_z", 0.00451215625, 31.25, 55.5),
        ("LeftHand_x", 0.0053125, 6.25, 33.0),
    ],
)
def test_motor_classes_are_pinned(joint, armature, effort, velocity):
    # The robot carries its own motor numbers: no other robot's config may move
    # them. The ankle has two motors in parallel (scale=2 doubles the armature).
    from protomotions.robot_configs.factory import robot_config

    control = robot_config("anatomic_soma").control.control_info[joint]
    assert control.armature == pytest.approx(armature)
    assert control.effort_limit == pytest.approx(effort)
    assert control.velocity_limit == pytest.approx(velocity)
    assert control.stiffness == pytest.approx(armature * (20 * math.pi) ** 2)
    assert control.damping == pytest.approx(4 * armature * 20 * math.pi)


def test_physx_body_speed_cap_matches_g1():
    """Anatomical SOMA uses the same PhysX body angular-speed cap as G1."""
    from protomotions.robot_configs.factory import robot_config

    assert (
        robot_config("anatomic_soma").asset.max_angular_velocity
        == robot_config("g1").asset.max_angular_velocity
    )


def test_implicit_pd_dissipates_perturbation_without_gravity_or_contacts():
    from protomotions.robot_configs.anatomic_soma import AnatomicSomaRobotConfig
    from protomotions.robot_configs.base import ControlType
    from protomotions.simulator.mujoco.simulator import MujocoSimulator

    config = AnatomicSomaRobotConfig()
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    model.opt.timestep = 1 / config.simulation_params.mujoco.fps
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.gravity[:] = 0
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    model.jnt_stiffness[:] = 0
    model.dof_damping[:] = 0
    sim = object.__new__(MujocoSimulator)
    sim.model = model
    sim.data = mujoco.MjData(model)
    sim.robot_config = config
    sim.control_type = ControlType.BUILT_IN_PD
    sim._has_free_joint = True
    sim._neutralize_actuator_gear()
    sim._override_joint_properties()
    sim._configure_actuators_for_pd()
    sim.data.qpos[7:] = 0.2 * np.sin(np.arange(model.nu))
    for _ in range(20 * config.simulation_params.mujoco.fps):
        mujoco.mj_step(model, sim.data)
        assert np.isfinite(sim.data.qpos).all()
        assert np.isfinite(sim.data.qvel).all()
    assert sim.data.warning.number.sum() == 0
    assert np.abs(sim.data.qpos[7:]).max() < 1e-3
    assert np.abs(sim.data.qvel[6:]).max() < 1e-3


@pytest.mark.parametrize(
    "seed,amplitude,bm_actions", [(0, 0.25, False), (3, 1.0, False), (5, 4.0, True)]
)
def test_random_targets_at_30_hz_remain_finite_and_settle(seed, amplitude, bm_actions):
    import torch

    from protomotions.envs.action import make_bm_pd_action_config
    from protomotions.robot_configs.anatomic_soma import AnatomicSomaRobotConfig
    from protomotions.robot_configs.base import ControlType
    from protomotions.simulator.mujoco.simulator import MujocoSimulator

    config = AnatomicSomaRobotConfig()
    spec = mujoco.MjSpec.from_file(str(ASSET))
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1]
    )
    sim = object.__new__(MujocoSimulator)
    sim.model = spec.compile()
    sim.data = mujoco.MjData(sim.model)
    sim.robot_config = config
    sim.control_type = ControlType.BUILT_IN_PD
    sim._has_free_joint = True
    sim._zero_passive_forces()
    sim._neutralize_actuator_gear()
    sim._override_joint_properties()
    sim._configure_actuators_for_pd()
    params = config.simulation_params.mujoco
    # Decimation is derived on main as physics fps over the control rate.
    decimation = params.fps // config.control_hz
    assert params.fps == decimation * config.control_hz == decimation * 30
    sim.model.opt.timestep = 1 / params.fps
    sim.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    sim.data.qpos[2] = 1.0
    lower, upper = sim.model.jnt_range[1:].T
    rng = np.random.default_rng(seed)
    action_config = make_bm_pd_action_config(config)
    effort = np.array(
        [
            config.control.control_info[name].effort_limit
            for name in config.kinematic_info.dof_names
        ]
    )
    targets_outside_limits = False
    previous_time = 0.0
    late_speeds = []
    # Falls and contacts are intentional. Hold the final command for 20 seconds
    # to distinguish commanded motion from sustained controller oscillation.
    for frame in range(50 * 30):
        if frame < 30 * 30:
            if bm_actions:
                raw = torch.tensor(
                    rng.normal(0, amplitude, (1, sim.model.nu)), dtype=torch.float32
                )
                sim.data.ctrl[:] = action_config.apply(raw)["processed_action"][
                    0
                ].numpy()
            else:
                sim.data.ctrl[:] = (lower + upper) / 2 + (
                    amplitude
                    * 0.45
                    * (upper - lower)
                    * rng.uniform(-1, 1, sim.model.nu)
                )
            targets_outside_limits |= bool(
                ((sim.data.ctrl < lower) | (sim.data.ctrl > upper)).any()
            )
        for _ in range(decimation):
            mujoco.mj_step(sim.model, sim.data)
            assert np.isfinite(sim.data.qpos).all()
            assert np.isfinite(sim.data.qvel).all()
            assert sim.data.time > previous_time  # Detect MuJoCo's automatic reset.
            previous_time = sim.data.time
            assert (np.abs(sim.data.actuator_force) <= effort + 1e-5).all()
        if frame >= 45 * 30:
            late_speeds.append(np.abs(sim.data.qvel[6:]).max())
    assert sim.data.warning.number.sum() == 0
    # Sustained PD oscillation shows up at rad/s scale. A body that fell during
    # the random phase keeps stick-slipping against the floor at a few
    # hundredths of a rad/s, so the bound sits well above that creep.
    assert max(late_speeds) < 0.1  # rad/s, after targets have stopped changing
    assert targets_outside_limits == bm_actions


def test_soma23_remains_a_separate_robot():
    from protomotions.robot_configs.factory import robot_config

    soma = robot_config("soma23")
    anatomical = robot_config("anatomic_soma")
    assert soma.asset.asset_file_name == "mjcf/soma23_humanoid.xml"
    assert anatomical.asset.asset_file_name == "mjcf/anatomic_soma.xml"
    assert soma.kinematic_info.num_dofs == 66
    assert anatomical.kinematic_info.num_dofs == 52
    assert soma.semantic_forward_axis_xy == (0.0, -1.0)
    assert anatomical.semantic_forward_axis_xy == (1.0, 0.0)
    # SOMA23 keeps the root anchor and target set its shipped policies were
    # trained with; the anatomical model anchors on the chest.
    assert soma.anchor_body_name is None
    assert anatomical.anchor_body_name == "Chest"
    assert "Hips" in soma.trackable_bodies_subset
    assert "Chest" in anatomical.trackable_bodies_subset
    assert "Hips" not in anatomical.trackable_bodies_subset
    assert anatomical.simulation_params.mujoco == soma.simulation_params.mujoco
    assert anatomical.simulation_params.mujoco.integrator == "implicitfast"
    assert anatomical.simulation_params.mujoco.fps == 960
    assert anatomical.simulation_params.mujoco.fps // anatomical.control_hz == 32
    assert anatomical.simulation_params.isaaclab.fps == 240
    assert anatomical.simulation_params.isaaclab.fps // anatomical.control_hz == 8
    assert anatomical.simulation_params.isaaclab.physx.num_position_iterations == 8
    assert anatomical.simulation_params.isaaclab.physx.num_velocity_iterations == 4
    assert soma.simulation_params.isaaclab.fps == 120
    for backend in ("isaacgym", "newton"):
        assert getattr(anatomical.simulation_params, backend) == getattr(
            soma.simulation_params, backend
        )
    assert (
        anatomical.common_naming_to_robot_body_names
        == soma.common_naming_to_robot_body_names
    )
    assert anatomical.trackable_bodies_subset == [
        "Chest",
        "Head",
        "LeftFoot",
        "RightFoot",
        "LeftHand",
        "RightHand",
    ]
    assert soma.trackable_bodies_subset == [
        "Hips",
        "Head",
        "LeftFoot",
        "RightFoot",
        "LeftHand",
        "RightHand",
    ]


def test_canonical_rest_positions_match_source_soma(model):
    source = mujoco.MjModel.from_xml_path(str(ASSET.with_name("soma23_humanoid.xml")))
    a, b = mujoco.MjData(source), mujoco.MjData(model)
    mujoco.mj_forward(source, a)
    mujoco.mj_forward(model, b)
    names = semantic_names(ASSET)
    old_ids = [source.body(name).id for name in names]
    new_ids = [model.body(name).id for name in names]
    # Compare root-relative offsets because initial ground placement is separate.
    expected = (a.xpos[old_ids] - a.xpos[old_ids[0]]) @ SOMA_TO_CANONICAL.T
    actual = b.xpos[new_ids] - b.xpos[new_ids[0]]
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_pose_conversion_rotates_both_world_and_body_frames():
    rng = np.random.default_rng(7)
    positions = rng.normal(size=(8, 23, 3))
    rotations = Rotation.random(8 * 23, random_state=rng).as_quat().reshape(8, 23, 4)
    p, q = canonical_soma_poses(positions, rotations)
    source_matrix = Rotation.from_quat(rotations.reshape(-1, 4)).as_matrix()
    target_matrix = Rotation.from_quat(q.reshape(-1, 4)).as_matrix()
    C = SOMA_TO_CANONICAL
    np.testing.assert_allclose(p @ C, positions, atol=1e-12)
    np.testing.assert_allclose(C.T @ target_matrix @ C, source_matrix, atol=1e-12)
    np.testing.assert_allclose(C @ [0.0, -1.0, 0.0], [1.0, 0.0, 0.0])


def test_no_penetrating_nonexcluded_rest_contacts(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert all(c.dist >= -1e-6 for c in data.contact)


def test_passive_dynamics_remain_finite():
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    model.opt.gravity[:] = 0
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    data.qpos[7] = 0.03
    for _ in range(500):
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert np.abs(data.qvel).max() < 1


def test_scene_conversion_preserves_robot_object_relative_geometry():
    rng = np.random.default_rng(31)
    p, q = rng.normal(size=(4, 3)), Rotation.random(4, random_state=rng).as_quat()
    scene = {
        "num_original_scenes": 1,
        "num_objects_per_scene": 1,
        "original_scenes": [
            {
                "offset": [2.0, -3.0],
                "humanoid_motion_id": 7,
                "objects": [
                    {
                        "translation": p.tolist(),
                        "rotation": q.tolist(),
                        "scale": [0.7, 0.8, 0.9],
                        "object_path": "mesh.obj",
                    }
                ],
            }
        ],
    }
    converted = canonical_soma_scenes(scene)
    actual = converted["original_scenes"][0]
    np.testing.assert_allclose(actual["offset"], [3.0, 2.0])
    obj = actual["objects"][0]
    points = rng.normal(size=(10, 3)) * [0.7, 0.8, 0.9]
    source_world = (
        np.einsum("tij,pj->tpi", Rotation.from_quat(q).as_matrix(), points) + p[:, None]
    )
    actual_world = (
        np.einsum(
            "tij,pj->tpi", Rotation.from_quat(obj["rotation"]).as_matrix(), points
        )
        + np.asarray(obj["translation"])[:, None]
    )
    np.testing.assert_allclose(
        actual_world, source_world @ SOMA_TO_CANONICAL.T, atol=1e-12
    )
    assert actual["humanoid_motion_id"] == 7
    assert obj["scale"] == [0.7, 0.8, 0.9] and obj["object_path"] == "mesh.obj"
    assert scene["original_scenes"][0]["offset"] == [2.0, -3.0]
    np.testing.assert_array_equal(
        scene["original_scenes"][0]["objects"][0]["rotation"], q
    )


@pytest.mark.parametrize("heightfield", [False, True])
def test_scene_file_roundtrip_and_overwrite_protection(tmp_path, heightfield):
    import importlib.util

    path = (
        ASSET.parents[4]
        / "data/scripts/from_soma_to_anatomical_soma/convert_soma_scene.py"
    )
    spec = importlib.util.spec_from_file_location("convert_soma_scene", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, output = tmp_path / "source.pt", tmp_path / "canonical.pt"
    data = {
        "num_original_scenes": 1,
        "num_objects_per_scene": 1,
        "original_scenes": [
            {
                "offset": [0.0, 0.0],
                "humanoid_motion_id": 0,
                "objects": [
                    {
                        "type": "BoxSceneObject",
                        "translation": [[0.0, -2.0, 0.5]],
                        "rotation": [[0.0, 0.0, 0.0, 1.0]],
                        "fps": 30,
                        "width": 1.0,
                        "depth": 2.0,
                        "height": 0.5,
                    }
                ],
            }
        ],
    }
    if heightfield:
        data["original_scenes"][0]["objects"][0].update(
            type="HeightfieldSceneObject",
            heightfield=torch.arange(12).reshape(3, 4).float(),
            horizontal_scale=0.1,
            min_point=[-0.1, -0.2],
        )
    torch.save(data, source)
    module.convert(source, output)
    actual = torch.load(output, weights_only=True)
    np.testing.assert_allclose(
        actual["original_scenes"][0]["objects"][0]["translation"], [[2.0, 0.0, 0.5]]
    )
    assert module.same_scene_data(torch.load(source, weights_only=True), data)
    if heightfield:
        obj = actual["original_scenes"][0]["objects"][0]
        torch.testing.assert_close(
            obj["heightfield"], torch.arange(12).reshape(3, 4).float()
        )
        # Heights/grid remain object-local; the object pose rotates the grid.
        assert obj["min_point"] == [-0.1, -0.2]
    with pytest.raises(ValueError, match="overwrite"):
        module.convert(source, output)


def test_scene_conversion_keeps_a_source_that_shares_the_temporary_stem(tmp_path):
    # A fixed "<stem>.partial.pt" scratch name once overwrote this source and then
    # renamed it away; the output must be written without touching it.
    import importlib.util

    path = (
        ASSET.parents[4]
        / "data/scripts/from_soma_to_anatomical_soma/convert_soma_scene.py"
    )
    spec = importlib.util.spec_from_file_location("convert_soma_scene", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, output = tmp_path / "scene.partial.pt", tmp_path / "scene.pt"
    data = {
        "num_original_scenes": 1,
        "num_objects_per_scene": 1,
        "original_scenes": [
            {
                "offset": [0.0, 0.0],
                "humanoid_motion_id": 0,
                "objects": [
                    {
                        "type": "BoxSceneObject",
                        "translation": [[0.0, -2.0, 0.5]],
                        "rotation": [[0.0, 0.0, 0.0, 1.0]],
                        "fps": 30,
                        "width": 1.0,
                        "depth": 2.0,
                        "height": 0.5,
                    }
                ],
            }
        ],
    }
    torch.save(data, source)
    before = source.read_bytes()
    module.convert(source, output)
    assert source.read_bytes() == before
    assert module.same_scene_data(torch.load(source, weights_only=True), data)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "scene.metadata.json",
        "scene.partial.pt",
        "scene.pt",
    ]


def test_retarget_resume_and_failure_reporting(tmp_path, monkeypatch):
    import importlib.util
    import json
    from contextlib import contextmanager

    spec = importlib.util.spec_from_file_location(
        "soma_driver",
        ASSET.parents[4]
        / "data/scripts/from_soma_to_anatomical_soma/retarget_motions.py",
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    source_model = mujoco.MjModel.from_xml_path(
        str(ASSET.with_name("soma23_humanoid.xml"))
    )
    state = mujoco.MjData(source_model)
    mujoco.mj_forward(source_model, state)
    frames = 6
    packed = {
        "gts": torch.tensor(
            np.repeat(state.xpos[None, 1:], frames, axis=0), dtype=torch.float32
        ),
        "grs": torch.tensor(
            np.repeat(state.xquat[None, 1:, [1, 2, 3, 0]], frames, axis=0),
            dtype=torch.float32,
        ),
        "contacts": None,
        "motion_num_frames": torch.tensor([3, 3]),
        "motion_dt": torch.tensor([1 / 30, 1 / 30]),
        "motion_lengths": torch.tensor([2 / 30, 2 / 30]),
        "motion_weights": torch.ones(2),
        "length_starts": torch.tensor([0, 3]),
        "motion_files": ("first", "second"),
    }
    source = tmp_path / "source.pt"
    torch.save(packed, source)
    calls = []
    reject_first = True

    @contextmanager
    def fake_worker(command):
        def solve(input_path, output_path):
            calls.append(input_path)
            if reject_first and len(calls) == 1:
                raise ValueError("test quality rejection")
            with np.load(input_path) as inputs:
                np.savez(output_path, dof_pos=inputs["seed"], diagnostics="{}")

        yield solve

    monkeypatch.setattr(driver, "solver_worker", fake_worker)
    output, cache = tmp_path / "result.pt", tmp_path / "resume"
    with pytest.raises(ValueError, match="no incomplete pack"):
        driver.retarget(source, output, "unused", resume_dir=cache)
    assert len(calls) == 2 and not output.exists()
    assert (cache / "00000001.npz").exists()
    assert (
        json.loads(output.with_suffix(".audit.jsonl").read_text().splitlines()[0])[
            "status"
        ]
        == "failed"
    )
    reject_first = False
    driver.retarget(source, output, "unused", resume_dir=cache)
    assert len(calls) == 3  # Only the rejected clip was retried.
    result = torch.load(output, weights_only=True)
    assert result["motion_files"] == packed["motion_files"]
    assert torch.equal(result["motion_num_frames"], packed["motion_num_frames"])
    driver.retarget(source, tmp_path / "cached.pt", "unused", resume_dir=cache)
    assert len(calls) == 3

    with pytest.raises(ValueError, match="contract changed"):
        driver.retarget(
            source,
            tmp_path / "different.pt",
            "unused",
            iterations=501,
            resume_dir=cache,
        )
    with pytest.raises(ValueError, match="contract changed"):
        driver.retarget(
            source,
            tmp_path / "bucketed.pt",
            "unused",
            frame_bucket_size=128,
            resume_dir=cache,
        )
    packed["gts"][0, 0, 0] += 0.01
    torch.save(packed, source)
    with pytest.raises(ValueError, match="Cached source changed"):
        driver.retarget(source, tmp_path / "modified.pt", "unused", resume_dir=cache)
