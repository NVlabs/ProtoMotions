# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the native fixed-root contract without meshes or external fixtures."""

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")

from protomotions.components.pose_lib import ControlInfo  # noqa: E402
from protomotions.components.scene_lib import SceneLib  # noqa: E402
from protomotions.robot_configs.base import (  # noqa: E402
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
)
from protomotions.simulator.base_simulator.config import ProjectileConfig  # noqa: E402
from protomotions.simulator.base_simulator.simulator_state import (  # noqa: E402
    ResetState,
    StateConversion,
)
from protomotions.simulator.mujoco.config import (  # noqa: E402
    MujocoSimParams,
    MujocoSimulatorConfig,
)
from protomotions.simulator.mujoco.simulator import MujocoSimulator  # noqa: E402


def _asset(tmp_path, free_joint="<freejoint/>", armature=0.0):
    # Two hinges with reversed motor order, placed out of contact.
    path = tmp_path / "robot.xml"
    path.write_text(f"""<mujoco>
      <compiler angle="radian"/>
      <worldbody><body name="base" pos="1 2 3">
        {free_joint}<geom type="box" size="0.1 0.1 0.1" mass="1"/>
        <body name="left" pos="0 0.4 0">
          <joint name="left_joint" axis="0 0 1" range="-1 1" armature="{armature}"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
        <body name="right" pos="0 -0.4 0">
          <joint name="right_joint" axis="0 0 1" range="-1 1" armature="{armature}"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </body></worldbody>
      <actuator><motor joint="right_joint"/><motor joint="left_joint"/></actuator>
    </mujoco>""")
    return path


def _build(path, fixed=True, projectiles=0, mode="torque", armature=None):
    rc = RobotConfig(
        asset=RobotAssetConfig(
            asset_root=str(path.parent), asset_file_name=path.name, fix_base_link=fixed
        ),
        control_hz=200,
        semantic_forward_axis_xy=(1.0, 0.0),
        common_naming_to_robot_body_names={
            name: ["base"]
            for name in (
                "all_left_foot_bodies",
                "all_right_foot_bodies",
                "all_left_hand_bodies",
                "all_right_hand_bodies",
                "head_body_name",
                "torso_body_name",
            )
        },
        control=ControlConfig(
            control_type=ControlType.TORQUE
            if mode == "torque"
            else ControlType.BUILT_IN_PD,
            override_control_info={
                ".*": ControlInfo(
                    stiffness=2.0,
                    damping=0.1,
                    effort_limit=100,
                    velocity_limit=100,
                    armature=armature,
                )
            },
        ),
    )
    sim = MujocoSimulator(
        config=MujocoSimulatorConfig(
            num_envs=1,
            headless=True,
            experiment_name="fixed_base_contract",
            sim=MujocoSimParams(fps=200),
            projectile=ProjectileConfig(num_projectiles=projectiles),
            use_implicit_pd=mode != "explicit_pd",
        ),
        robot_config=rc,
        terrain=None,
        device=torch.device("cpu"),
        scene_lib=SceneLib.empty(num_envs=1, device="cpu", terrain=None),
    )
    sim._initialize_with_markers({})
    return sim


def _reset(sim, positions=(0.1, -0.2), velocities=(0.3, -0.4)):
    sim.reset_envs(
        ResetState(
            root_pos=torch.tensor([[4.0, 5.0, 6.0]]),
            root_rot=torch.tensor([[0.0, 0.0, 0.6, 0.8]]),
            root_vel=torch.tensor([[0.1, 0.2, 0.3]]),
            root_ang_vel=torch.tensor([[0.3, -0.2, 0.1]]),
            dof_pos=torch.tensor([positions]),
            dof_vel=torch.tensor([velocities]),
            state_conversion=StateConversion.COMMON,
        )
    )


@pytest.mark.parametrize("free_joint", ["<freejoint/>", '<joint type="free"/>'])
@pytest.mark.parametrize("fixed", [True, False, None])
def test_native_reset_step_root_and_projectile_indices(tmp_path, free_joint, fixed):
    """Fixed roots keep their authored pose while joints and projectiles still move."""
    path = _asset(tmp_path, free_joint)
    original = path.read_bytes()
    sim = _build(path, fixed=fixed, projectiles=2)
    assert path.read_bytes() == original
    root_qpos = 0 if fixed else 7
    root_qvel = 0 if fixed else 6
    assert sim.model.nq == root_qpos + 2 + 14
    assert sim.model.nv == root_qvel + 2 + 12
    assert sim._proj_qpos_start == root_qpos + 2
    assert sim._proj_qvel_start == root_qvel + 2
    np.testing.assert_array_equal(sim._dof_to_actuator, [1, 0])
    for index in range(2):
        body = sim.model.body(f"projectile_{index}")
        joint = int(body.jntadr[0])
        assert sim.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
        assert sim.model.jnt_qposadr[joint] == sim._proj_qpos_start + 7 * index
        assert sim.model.jnt_dofadr[joint] == sim._proj_qvel_start + 6 * index

    _reset(sim)
    state = sim.get_robot_state()
    np.testing.assert_allclose(state.dof_pos, [[0.1, -0.2]])
    np.testing.assert_allclose(state.dof_vel, [[0.3, -0.4]])
    expected_root = [[1, 2, 3]] if fixed else [[4, 5, 6]]
    np.testing.assert_allclose(state.root_pos, expected_root)
    np.testing.assert_allclose(state.rigid_body_pos[:, 0], expected_root)
    np.testing.assert_allclose(
        state.root_rot, [[0, 0, 0, 1]] if fixed else [[0, 0, 0.6, 0.8]]
    )
    np.testing.assert_allclose(
        state.root_vel, [[0, 0, 0]] if fixed else [[0.1, 0.2, 0.3]]
    )
    np.testing.assert_allclose(
        state.root_ang_vel, [[0, 0, 0]] if fixed else [[0.3, -0.2, 0.1]], atol=1e-7
    )

    qpos_before = sim.data.qpos[: root_qpos + 2].copy()
    sim._set_projectile_root_states(
        torch.tensor([1]),
        torch.tensor([[10.0, 10.0, 8.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.zeros(1, 3),
        torch.tensor([0]),
    )
    np.testing.assert_array_equal(sim.data.qpos[: root_qpos + 2], qpos_before)
    sim.step(torch.tensor([[0.01, -0.02]]))
    assert np.isfinite(sim.data.qpos).all()
    assert sim.data.qpos[sim._proj_qpos_start + 7] > 10.0
    assert not np.allclose(sim.get_robot_state().dof_pos, state.dof_pos)
    if fixed:
        np.testing.assert_allclose(sim.get_root_state().root_pos, expected_root)
        np.testing.assert_allclose(sim.get_root_state().root_rot, [[0, 0, 0, 1]])
        np.testing.assert_allclose(sim.get_robot_state().rigid_body_vel[:, 0], 0)
        before = sim.data.qvel.copy()
        sim._apply_root_velocity_impulse(
            torch.ones(1, 3), torch.ones(1, 3), torch.tensor([0])
        )
        np.testing.assert_array_equal(sim.data.qvel, before)
    _reset(sim, positions=(-0.2, 0.1), velocities=(0.0, 0.0))
    np.testing.assert_allclose(sim.get_robot_state().dof_pos, [[-0.2, 0.1]])
    np.testing.assert_allclose(sim.get_robot_state().dof_vel, 0)
    np.testing.assert_allclose(sim.get_root_state().root_pos, expected_root)
    assert path.read_bytes() == original


@pytest.mark.parametrize("mode", ["implicit_pd", "explicit_pd"])
@pytest.mark.parametrize("fixed", [True, False])
def test_pd_steps_exclude_projectile_coordinates(tmp_path, mode, fixed):
    """Both PD modes control only robot joints with fixed or floating roots."""
    sim = _build(_asset(tmp_path), fixed=fixed, projectiles=2, mode=mode)
    _reset(sim, positions=(0.0, 0.0), velocities=(0.0, 0.0))
    for _ in range(3):
        sim.step(torch.tensor([[0.1, -0.2]]))
    assert sim.get_robot_state().dof_vel[0, 0] > 0
    assert sim.get_robot_state().dof_vel[0, 1] < 0
    assert np.isfinite(sim.data.qpos).all()


def _scramble_projectiles(sim):
    """Give the projectile coordinates values no robot joint could have.

    Explicit PD reads qpos/qvel by position, so a slice that reaches past the
    robot joints picks these up: either as a shape error or, if the slice is
    merely shifted, as torques computed from the wrong coordinates.
    """
    sim.data.qpos[sim._proj_qpos_start :] = 100.0
    sim.data.qvel[sim._proj_qvel_start :] = -50.0
    mujoco.mj_forward(sim.model, sim.data)


@pytest.mark.parametrize("fixed", [True, False])
def test_explicit_pd_torque_is_the_pd_law_on_the_robot_joints_only(tmp_path, fixed):
    """With projectiles present, the torque is the PD law on the robot joints alone."""
    sim = _build(_asset(tmp_path), fixed=fixed, projectiles=2, mode="explicit_pd")
    _reset(sim, positions=(0.1, -0.2), velocities=(0.3, -0.4))
    _scramble_projectiles(sim)

    sim._apply_simulator_pd_targets(torch.tensor([[0.5, -0.6]]))

    # Gains from _build: stiffness 2.0, damping 0.1, in the asset's joint order.
    expected = 2.0 * (np.array([0.5, -0.6]) - np.array([0.1, -0.2])) - 0.1 * np.array(
        [0.3, -0.4]
    )
    np.testing.assert_allclose(sim.data.ctrl[sim._dof_to_actuator], expected, atol=1e-6)
    mujoco.mj_forward(sim.model, sim.data)
    assert np.all(sim.data.qfrc_actuator[sim._proj_qvel_start :] == 0.0)
    np.testing.assert_allclose(
        sim.data.qfrc_actuator[sim._proj_qvel_start - 2 : sim._proj_qvel_start],
        expected,
        atol=1e-6,
    )


@pytest.mark.parametrize("fixed", [True, False])
def test_explicit_pd_matches_implicit_pd_with_projectiles(tmp_path, fixed):
    """Both PD modes apply the same actuator force on one state, projectiles present."""
    forces = {}
    for mode in ("implicit_pd", "explicit_pd"):
        sim = _build(_asset(tmp_path), fixed=fixed, projectiles=2, mode=mode)
        _reset(sim, positions=(0.1, -0.2), velocities=(0.3, -0.4))
        _scramble_projectiles(sim)
        sim._apply_simulator_pd_targets(torch.tensor([[0.5, -0.6]]))
        mujoco.mj_forward(sim.model, sim.data)
        forces[mode] = sim.data.qfrc_actuator.copy()
    np.testing.assert_allclose(forces["explicit_pd"], forces["implicit_pd"], atol=1e-6)


@pytest.mark.parametrize("free_joint", ["<freejoint/>", '<joint type="free"/>'])
def test_loader_welds_only_named_robot_root(tmp_path, free_joint):
    """Welding preserves the asset and all bodies except the named root free joint."""
    path = _asset(tmp_path, free_joint)
    xml = path.read_text().replace(
        "<worldbody>",
        '<worldbody><body name="unrelated" pos="5 0 2"><freejoint name="other_free"/>'
        '<geom size="0.1"/></body>',
    )
    path.write_text(xml)
    source_model = mujoco.MjModel.from_xml_path(str(path))
    model = MujocoSimulator._load_mjcf_stripped(
        str(path), ProjectileConfig(num_projectiles=2), fixed_root_body="base"
    )
    assert model.body("base").jntnum[0] == 0
    assert model.joint("other_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("left_joint").type[0] == mujoco.mjtJoint.mjJNT_HINGE
    assert model.joint("right_joint").type[0] == mujoco.mjtJoint.mjJNT_HINGE
    assert np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 3
    for name in ("base", "left", "right", "unrelated"):
        source_body = source_model.body(name)
        body = model.body(name)
        np.testing.assert_array_equal(body.pos, source_body.pos)
        np.testing.assert_array_equal(body.quat, source_body.quat)
        np.testing.assert_array_equal(body.mass, source_body.mass)
        np.testing.assert_array_equal(body.inertia, source_body.inertia)
        source_geom = source_model.geom(int(source_body.geomadr[0]))
        geom = model.geom(int(body.geomadr[0]))
        for attribute in ("type", "size", "pos", "quat", "contype", "conaffinity"):
            np.testing.assert_array_equal(
                getattr(geom, attribute), getattr(source_geom, attribute)
            )
    assert path.read_text() == xml


@pytest.mark.parametrize(
    "root_name, free_joint, message",
    [
        ("missing", "<freejoint/>", "not found"),
        ("base", "", "exactly one authored free joint"),
    ],
)
def test_loader_rejects_missing_fixed_root_contract(
    tmp_path, root_name, free_joint, message
):
    """A fixed robot must name an authored root with exactly one free joint."""
    path = _asset(tmp_path, free_joint)
    with pytest.raises(ValueError, match=message):
        MujocoSimulator._load_mjcf_stripped(str(path), fixed_root_body=root_name)


@pytest.mark.parametrize("projectiles", [0, 2])
@pytest.mark.parametrize("coordinates", ["qpos", "qvel", "both"])
def test_fixed_root_discards_unused_keyframe_coordinates(
    tmp_path, projectiles, coordinates
):
    """Welding accepts authored keyframes and resets through the simulator state."""
    path = _asset(tmp_path)
    attributes = []
    if coordinates in ("qpos", "both"):
        attributes.append('qpos="1 2 3 1 0 0 0 .3 -.4"')
    if coordinates in ("qvel", "both"):
        attributes.append('qvel=".1 .2 .3 .4 .5 .6 .7 .8"')
    keyframe = f'<keyframe><key name="home" {" ".join(attributes)}/></keyframe>'
    path.write_text(path.read_text().replace("</mujoco>", keyframe + "</mujoco>"))
    original = path.read_bytes()
    assert mujoco.MjModel.from_xml_path(str(path)).nkey == 1
    sim = _build(path, projectiles=projectiles)
    assert sim.model.nkey == 0
    _reset(sim)
    np.testing.assert_allclose(sim.get_robot_state().dof_pos, [[0.1, -0.2]])
    sim.step(torch.tensor([[0.01, -0.02]]))
    assert np.isfinite(sim.data.qpos).all()
    np.testing.assert_allclose(sim.get_root_state().root_pos, [[1, 2, 3]])
    assert path.read_bytes() == original


@pytest.mark.parametrize("projectiles", [0, 2])
def test_fixed_root_preserves_named_body_with_static_fusion(tmp_path, projectiles):
    """Welding retains the named robot root even when its asset enables fusion."""
    path = _asset(tmp_path)
    path.write_text(
        path.read_text().replace('angle="radian"', 'angle="radian" fusestatic="true"')
    )
    original = path.read_bytes()
    assert mujoco.MjModel.from_xml_path(str(path)).body("base").id > 0
    sim = _build(path, projectiles=projectiles)
    assert sim.model.body("base").id > 0
    _reset(sim)
    sim.step(torch.tensor([[0.01, -0.02]]))
    np.testing.assert_allclose(sim.get_root_state().root_pos, [[1, 2, 3]])
    np.testing.assert_allclose(sim.get_robot_state().rigid_body_pos[:, 0], [[1, 2, 3]])
    assert np.isfinite(sim.data.qpos).all()
    assert path.read_bytes() == original


def test_floating_root_retains_keyframes_and_static_fusion(tmp_path):
    """Floating models retain authored keyframes and compiler fusion behavior."""
    path = _asset(tmp_path)
    xml = path.read_text().replace('angle="radian"', 'angle="radian" fusestatic="true"')
    xml = xml.replace(
        "<worldbody>", '<worldbody><body name="static_marker"><geom size=".1"/></body>'
    ).replace(
        "</mujoco>",
        '<keyframe><key name="home" qpos="1 2 3 1 0 0 0 .3 -.4" '
        'qvel=".1 .2 .3 .4 .5 .6 .7 .8"/></keyframe></mujoco>',
    )
    path.write_text(xml)
    original = path.read_bytes()
    source = mujoco.MjModel.from_xml_path(str(path))
    model = MujocoSimulator._load_mjcf_stripped(str(path))
    assert model.nkey == 1
    np.testing.assert_array_equal(model.key_qpos, source.key_qpos)
    np.testing.assert_array_equal(model.key_qvel, source.key_qvel)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "static_marker") == -1
    assert path.read_bytes() == original


@pytest.mark.parametrize("fixed", [True, False])
def test_armature_override_matches_authored_inertia_and_dynamics(tmp_path, fixed):
    """An armature override matches compiled inertia and constrained dynamics."""
    path = _asset(tmp_path)
    overridden = _build(path, fixed=fixed, armature=2.0)
    authored = _build(_asset(tmp_path, armature=2.0), fixed=fixed)
    for sim in (overridden, authored):
        _reset(sim, positions=(0.0, 0.0), velocities=(0.0, 0.0))
        sim.step(torch.tensor([[1.0, -2.0]]))
        if fixed:
            # Each sphere's hinge inertia is 2/5 * mass * radius^2 + armature.
            np.testing.assert_allclose(
                sim.get_robot_state().dof_vel,
                [[0.005 / 2.004, -0.01 / 2.004]],
                rtol=1e-6,
            )
        _reset(sim, positions=(1.1, -1.1), velocities=(0.2, -0.1))
        for _ in range(5):
            sim.step(torch.tensor([[1.0, -2.0]]))
    np.testing.assert_allclose(overridden.data.qpos, authored.data.qpos, atol=1e-12)
    np.testing.assert_allclose(overridden.data.qvel, authored.data.qvel, atol=1e-12)
    np.testing.assert_allclose(
        overridden.model.dof_invweight0, authored.model.dof_invweight0
    )


def test_constant_refresh_uses_scratch_data_only_at_construction(tmp_path, monkeypatch):
    """Constants refresh once using scratch data without overwriting reset state."""
    original = mujoco.mj_setConst
    refreshed_data = []

    def record_refresh(model, data):
        refreshed_data.append(data)
        original(model, data)

    monkeypatch.setattr(mujoco, "mj_setConst", record_refresh)
    sim = _build(_asset(tmp_path), fixed=False, armature=2.0, projectiles=2)
    assert len(refreshed_data) == 1
    assert refreshed_data[0] is not sim.data
    _reset(sim)
    sim.step(torch.tensor([[0.01, -0.02]]))
    _reset(sim)
    assert len(refreshed_data) == 1
    np.testing.assert_allclose(sim.get_root_state().root_pos, [[4, 5, 6]])
    np.testing.assert_allclose(sim.get_root_state().root_rot, [[0, 0, 0.6, 0.8]])
    np.testing.assert_allclose(sim.get_robot_state().dof_pos, [[0.1, -0.2]])


def test_welded_root_readback_keeps_authored_transform(tmp_path):
    """Fixed-root engine readback returns the authored position and orientation."""
    path = _asset(tmp_path)
    path.write_text(
        path.read_text().replace('pos="1 2 3"', 'pos="1 2 3" quat="0.8 0.6 0 0"')
    )
    sim = object.__new__(MujocoSimulator)
    sim.model = MujocoSimulator._load_mjcf_stripped(str(path), fixed_root_body="base")
    sim.data = mujoco.MjData(sim.model)
    sim._has_free_joint = False
    sim._body_names = ["base", "left", "right"]
    mujoco.mj_forward(sim.model, sim.data)
    state = sim._get_simulator_root_state()
    assert state.state_conversion is StateConversion.SIMULATOR
    np.testing.assert_allclose(state.root_pos, [[1, 2, 3]])
    np.testing.assert_allclose(state.root_rot, [[0.8, 0.6, 0, 0]])
    np.testing.assert_array_equal(state.root_vel, np.zeros((1, 3)))
    np.testing.assert_array_equal(state.root_ang_vel, np.zeros((1, 3)))
