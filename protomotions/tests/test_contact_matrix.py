# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partner identity, force direction, current-only lifecycle, and reward routing."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from protomotions.components.pose_lib import build_body_ids_tensor
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.meta import RewardMeta
from protomotions.envs.rewards.regularization import (
    compute_terrain_foot_contact_match_rew,
)
from protomotions.envs.terminations.base import named_contact_fall_termination
from protomotions.simulator.base_simulator.contacts import ContactLayout


def _terrain_foot_reward(body_names, reference_body_names, **overrides):
    """The recipe's binding of the terrain foot-contact reward: the selection is
    all a recipe writes; every reading is the kernel's default."""
    return MdpComponent.from_kernel(
        compute_terrain_foot_contact_match_rew,
        meta=RewardMeta(weight=-0.1, zero_during_grace_period=True),
        required_keys={
            "contact_body_index": body_names,
            "contact_partner_index": ["terrain"],
        },
        contact_body_ids=build_body_ids_tensor(
            reference_body_names, body_names, device=torch.device("cpu")
        ),
        body_names=body_names,
        **overrides,
    )


def _fall_termination(body_names, all_body_names, partner_names, **overrides):
    """The recipe's binding of the named-contact fall termination."""
    return MdpComponent.from_kernel(
        named_contact_fall_termination,
        required_keys={
            "contact_body_index": body_names,
            "contact_partner_index": partner_names,
        },
        body_ids=build_body_ids_tensor(
            all_body_names, body_names, device=torch.device("cpu")
        ),
        body_names=body_names,
        partner_names=partner_names,
        **overrides,
    )


def test_layout_has_common_body_order_and_explicit_partner_names():
    layout = ContactLayout.build(["right", "torso", "left"], 2, 1)
    assert layout.body_index == {"right": 0, "torso": 1, "left": 2}
    assert layout.partner_index == {
        "terrain": 0,
        "robot/right": 1,
        "robot/torso": 2,
        "robot/left": 3,
        "object/0": 4,
        "object/1": 5,
        "projectile/0": 6,
    }
    assert layout.common_body_ids == (0, 1, 2)
    assert layout.is_complete


def test_selected_rows_and_partners_are_compact_and_canonical():
    layout = ContactLayout.build(
        ["right", "torso", "left"],
        100,
        2,
        contact_bodies=["left", "right"],
        contact_partners=["object/73", "robot/torso"],
    )
    assert layout.body_index == {"right": 0, "left": 1}
    assert layout.common_body_ids == (0, 2)
    assert layout.partner_index == {"robot/torso": 0, "object/73": 1}
    assert not layout.is_complete


def test_partner_globs_expand_once_and_deduplicate():
    layout = ContactLayout.build(
        ["foot", "hand"],
        2,
        2,
        contact_partners=[
            "terrain",
            "robot/*",
            "robot/foot",
            "object/*",
            "projectile/*",
        ],
    )
    assert layout.is_complete
    assert len(layout.partner_index) == 7


def test_robot_default_partners_do_not_expand_to_all_objects():
    from protomotions.robot_configs.base import RobotConfig

    assert RobotConfig.__dataclass_fields__["contact_partners"].default_factory() == [
        "terrain"
    ]


def test_contact_requirements_survive_pickle_and_readable_config():
    import pickle

    original = _terrain_foot_reward(["foot"], ["foot", "torso"])
    restored = pickle.loads(pickle.dumps(original))
    assert restored.required_keys == original.required_keys
    assert restored.to_dict()["required_keys"] == original.required_keys


@pytest.mark.parametrize("selection", [[], "object/2", ["robot/typo"], "missing/*"])
def test_invalid_partner_selection_fails_during_layout_construction(selection):
    with pytest.raises(ValueError, match="Contact partner"):
        ContactLayout.build(["foot"], 2, 0, contact_partners=selection)


def test_selected_contact_termination_uses_common_body_ids():
    bodies = [f"body_{i}" for i in range(25)]
    component = _fall_termination(["body_24"], bodies, ["terrain"])
    forces = torch.zeros(2, 2, 1, 3)
    forces[0, 0, 0, 2] = 10
    forces[1, 1, 0, 2] = 10
    ctx = SimpleNamespace(
        current=SimpleNamespace(rigid_body_pos=torch.zeros(2, 25, 3)),
        contact_forces=forces,
        contact_body_index={"body_7": 0, "body_24": 1},
        contact_partner_index={"terrain": 0},
        ground_heights=torch.zeros(2),
        progress_buf=torch.tensor([5, 5]),
    )
    torch.testing.assert_close(
        component.compute(ctx),
        torch.tensor([False, True]),
    )


@pytest.mark.parametrize("source", ["clean", "noisy"])
@pytest.mark.parametrize("historical", [False, True])
def test_privileged_contacts_cannot_be_requested_as_sensor_estimates(
    source, historical
):
    from protomotions.envs.context_sources import sensed
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.obs.humanoid import (
        compute_humanoid_max_coords_contact_observations,
    )
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_max_coords_from_state,
    )

    # Contact names are produced by the simulator alone, so sensed() refuses
    # to move a component that binds one to a sensor namespace.
    if historical:
        component = MdpComponent.from_kernel(
            compute_historical_max_coords_from_state,
            observe_contacts=True,
            historical_body_contacts=EnvContext.historical.body_contacts,
            history_steps=1,
        )
    else:
        component = MdpComponent.from_kernel(
            compute_humanoid_max_coords_contact_observations
        )
    with pytest.raises(ValueError, match="only the simulator produces"):
        sensed({"contacts": component}, source)


def test_terrain_matching_with_subset_rows_and_reference_labels_compiles():
    reward = _terrain_foot_reward(["left", "right"], ["torso", "right", "hand", "left"])
    context = SimpleNamespace(
        contact_forces=torch.tensor([[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0]]]]),
        contact_body_index={"right": 0, "left": 1},
        contact_partner_index={"terrain": 0},
        mimic=SimpleNamespace(
            ref_state=SimpleNamespace(
                rigid_body_contacts=torch.tensor([[0.0, 1.0, 0.0, 0.25]])
            )
        ),
    )
    torch.testing.assert_close(reward.compute(context), torch.tensor([0.25]))
    compiled = torch.compile(reward.compute_func, backend="eager", fullgraph=True)
    result = compiled(
        contact_body_ids=reward.static_params["contact_body_ids"],
        body_names=["left", "right"],
        contact_forces=context.contact_forces,
        ref_contacts=context.mimic.ref_state.rigid_body_contacts,
        contact_body_index=context.contact_body_index,
        contact_partner_index=context.contact_partner_index,
    )
    torch.testing.assert_close(result, torch.tensor([0.25]))
    context.contact_partner_index = {"object/0": 0}
    with pytest.raises(KeyError, match="terrain"):
        reward.compute(context)


@pytest.mark.parametrize("missing", ["foot", "terrain"])
@pytest.mark.parametrize("kind", ["reward", "termination"])
def test_named_contact_requirements_fail_during_setup(missing, kind):
    from protomotions.envs.component_manager import ComponentManager

    component = (
        _terrain_foot_reward(["foot"], ["foot", "torso"])
        if kind == "reward"
        else _fall_termination(["foot"], ["foot", "torso"], ["terrain"])
    )
    ctx = SimpleNamespace(
        current=SimpleNamespace(rigid_body_pos=torch.zeros(1, 2, 3)),
        contact_forces=torch.zeros(1, 1, 1, 3),
        contact_body_index={"foot": 0},
        contact_partner_index={"terrain": 0},
        ground_heights=torch.zeros(1),
        progress_buf=torch.ones(1),
        mimic=SimpleNamespace(
            ref_state=SimpleNamespace(rigid_body_contacts=torch.zeros(1, 2))
        ),
    )
    manager = ComponentManager(torch.device("cpu"))
    manager.validate_bindings({"named_contact": component}, ctx, group=kind)
    if missing == "foot":
        ctx.contact_body_index = {"torso": 0}
    else:
        ctx.contact_partner_index = {"object/0": 0}
    with pytest.raises(
        ValueError, match=f"{kind} component 'named_contact'.*{missing}.*not registered"
    ):
        manager.validate_bindings({"named_contact": component}, ctx, group=kind)


def test_named_fall_termination_does_not_confuse_rows_reference_ids_or_partners():
    component = _fall_termination(
        ["torso"], ["foot", "head", "torso"], ["terrain"], termination_height=0.15
    )
    positions = torch.ones(4, 3, 3)
    positions[:, 2, 2] = 0.1
    forces = torch.zeros(4, 2, 2, 3)
    forces[0, 0, 1, 2] = 50  # torso -> terrain: fall
    forces[1, 1, 1, 2] = 50  # foot -> terrain: allowed
    forces[2, 0, 0, 2] = 50  # torso -> object: not this termination's partner
    forces[3, 0, 1, 2] = 50  # torso -> terrain, but reset grace
    ctx = SimpleNamespace(
        current=SimpleNamespace(rigid_body_pos=positions),
        contact_forces=forces,
        contact_body_index={"torso": 0, "foot": 1},
        contact_partner_index={"object/0": 0, "terrain": 1},
        ground_heights=torch.zeros(4),
        progress_buf=torch.tensor([5, 5, 5, 1]),
    )
    torch.testing.assert_close(
        component.compute(ctx), torch.tensor([True, False, False, False])
    )
    resolved, params = component.resolve_args(ctx)
    compiled = torch.compile(component.compute_func, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(**resolved, **params), component.compute(ctx))
    # Contact alone is not a fall when all forbidden bodies are high enough.
    positions[:, 2, 2] = 0.5
    assert not component.compute(ctx).any()


def test_named_contact_requirement_rejects_non_mapping_binding():
    from protomotions.envs.component_manager import ComponentManager

    component = _terrain_foot_reward(["foot"], ["foot"])
    ctx = SimpleNamespace(
        contact_forces=torch.zeros(1, 1, 1, 3),
        contact_body_index=torch.tensor([0]),
        contact_partner_index={"terrain": 0},
        mimic=SimpleNamespace(
            ref_state=SimpleNamespace(rigid_body_contacts=torch.zeros(1, 1))
        ),
    )
    with pytest.raises(
        ValueError, match="reward component 'feet'.*contact_body_index.*mapping"
    ):
        ComponentManager(torch.device("cpu")).validate_bindings(
            {"feet": component}, ctx, group="reward"
        )


def test_required_contact_pattern_needs_at_least_one_matching_partner():
    from protomotions.envs.component_manager import ComponentManager
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.meta import RewardMeta

    def read_partners(partners):
        return partners

    component = MdpComponent(
        compute_func=read_partners,
        dynamic_vars={"partners": EnvContext.contact_partner_index},
        required_keys={"partners": ["robot/*"]},
        meta=RewardMeta(weight=1.0),
    )
    ctx = SimpleNamespace(contact_partner_index={"terrain": 0, "object/0": 1})
    manager = ComponentManager(torch.device("cpu"))
    with pytest.raises(
        ValueError, match="reward component 'self'.*robot/.*not registered"
    ):
        manager.validate_bindings({"self": component}, ctx, group="reward")
    ctx.contact_partner_index["robot/hand"] = 2
    manager.validate_bindings({"self": component}, ctx, group="reward")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compact_terrain_reward_compiles_on_cuda_with_config_reference_ids():
    forces = torch.tensor([[[[0.0, 0.0, 0.0]], [[0.0, 0.0, 2.0]]]], device="cuda")
    labels = torch.tensor([[1.0, 0.25, 0.0]], device="cuda")
    reference_ids = torch.tensor([1, 2])  # Config tensors originate on CPU.
    args = dict(
        contact_body_ids=reference_ids,
        body_names=["left", "right"],
        contact_forces=forces,
        ref_contacts=labels,
        contact_body_index={"left": 0, "right": 1},
        contact_partner_index={"terrain": 0},
    )
    result = torch.compile(compute_terrain_foot_contact_match_rew, fullgraph=True)(
        **args
    )
    torch.testing.assert_close(result, torch.tensor([1.25], device="cuda"))


def test_mdp_reductions_do_not_cancel_opposing_partner_forces():
    from protomotions.envs.contact_features import contact_flags, contact_magnitudes
    from protomotions.envs.rewards.regularization import (
        compute_contact_force_change_rew,
    )

    forces = torch.zeros(2, 3, 5, 3)
    forces[0, 1, 0, 2] = 40.0
    forces[0, 1, 4, 2] = -40.0
    forces[1, 2, 4, 0] = 60.0  # Object-only contact in the second env.
    assert forces[0, 1].sum(dim=0).norm() == 0
    torch.testing.assert_close(
        contact_magnitudes(forces), torch.tensor([[0.0, 40.0, 0.0], [0.0, 0.0, 60.0]])
    )
    torch.testing.assert_close(
        contact_flags(forces),
        torch.tensor([[False, True, False], [False, False, True]]),
    )
    torch.testing.assert_close(
        contact_flags(forces, torch.tensor([2, 1]), torch.tensor([0])),
        torch.tensor([[False, True], [False, False]]),
    )
    torch.testing.assert_close(
        _fall_termination(
            ["a", "b"], ["root", "a", "b"], ["terrain", "object/0"]
        ).compute(
            SimpleNamespace(
                current=SimpleNamespace(rigid_body_pos=torch.zeros(2, 3, 3)),
                contact_forces=forces,
                contact_body_index={"root": 0, "a": 1, "b": 2},
                contact_partner_index={"terrain": 0, "object/0": 4},
                ground_heights=torch.zeros(2),
                progress_buf=torch.tensor([5, 5]),
            )
        ),
        torch.tensor([True, True]),
    )
    torch.testing.assert_close(
        compute_contact_force_change_rew(forces, torch.zeros(2, 3), threshold=30.0),
        torch.tensor([10.0, 30.0]),
    )


def test_eval_runtime_snapshot_preserves_current_contacts_but_reset_clears_them():
    from protomotions.tests.test_base_simulator_lifecycle import _sim

    sim = _sim(contacts=True)
    measured = torch.ones(2, 2, 5, 3)
    sim._get_simulator_contact_forces = lambda: measured.clone()
    sim._initialize_with_markers({})
    sim.step(torch.zeros(2, 2))
    snapshot = sim.save_state()
    sim.reset_envs(sim.get_default_robot_reset_state(), env_ids=torch.arange(2))
    assert torch.count_nonzero(sim.get_contact_forces()) == 0
    sim.restore_state(snapshot)
    torch.testing.assert_close(sim.get_contact_forces(), measured)


@pytest.mark.parametrize("bodies", [["foot"], "all"])
def test_isaacgym_rejects_contacts_before_setup(bodies):
    constructor = _isaacgym_constructor()
    with pytest.raises(
        AssertionError, match="IsaacGym contact reporting is not supported"
    ):
        constructor(
            config=SimpleNamespace(),
            robot_config=SimpleNamespace(contact_bodies=bodies),
            terrain=None,
            device=torch.device("cpu"),
            scene_lib=None,
        )


@pytest.mark.parametrize("bodies", [None, []])
def test_isaacgym_without_contacts_proceeds_to_setup(bodies):
    constructor = _isaacgym_constructor()
    with pytest.raises(_SetupReached):
        constructor(
            config=SimpleNamespace(),
            robot_config=SimpleNamespace(contact_bodies=bodies),
            terrain=None,
            device=torch.device("cpu"),
            scene_lib=None,
        )


class _SetupReached(Exception):
    pass


def _isaacgym_constructor():
    """Run the actual constructor guard without loading the proprietary SDK."""
    path = Path(__file__).parents[1] / "simulator/isaacgym/simulator.py"
    cls = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
    )
    constructor = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )

    class SetupBoundary:
        def __init__(self, **kwargs):
            raise _SetupReached

    definition = ast.ClassDef(
        name="Backend",
        bases=[ast.Name(id="SetupBoundary", ctx=ast.Load())],
        keywords=[],
        body=[constructor],
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            definition,
        ],
        type_ignores=[],
    )
    namespace = {"SetupBoundary": SetupBoundary}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Backend"]


def test_mujoco_actual_terrain_self_projectile_and_fixed_obstacle_contacts():
    mujoco = pytest.importorskip("mujoco")
    from protomotions.simulator.mujoco.simulator import MujocoSimulator

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="floor" type="plane" size="3 3 .1"/>
      <geom name="obstacle" type="sphere" size=".1" pos="3 0 1"/>
      <body name="left" pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
      <body name="right" pos="1 0 .09"><freejoint/><geom type="sphere" size=".1" mass="2"/></body>
      <body name="torso" pos="4 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
      <body name="projectile_0" pos="2 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    sim = object.__new__(MujocoSimulator)
    sim.model, sim.data = model, data
    sim.config = SimpleNamespace(projectile=SimpleNamespace(num_projectiles=1))
    # A cyclic permutation detects accidentally using simulator body order.
    layout = sim.contact_layout = ContactLayout.build(["right", "torso", "left"], 0, 1)
    left, right = layout.body_index["left"], layout.body_index["right"]
    floor, cube = layout.partner_index["terrain"], layout.partner_index["projectile/0"]

    def read():
        mujoco.mj_forward(model, data)
        return sim._get_simulator_contact_forces()[0]

    forces = read()
    assert (forces[[left, right], floor, 2] > 1).all()
    torch.testing.assert_close(forces[:, floor, :2], torch.zeros(3, 2))
    torch.testing.assert_close(forces[:, 1:], torch.zeros(3, 4, 3))

    # Robot-robot contact: equal/opposite forces, neither counted as terrain.
    data.qpos[:3], data.qpos[7:10] = [0, 0, 1], [0.15, 0, 1]
    forces = read()
    a = forces[left, layout.partner_index["robot/right"]]
    b = forces[right, layout.partner_index["robot/left"]]
    assert a[0] < -1
    torch.testing.assert_close(a, -b)
    torch.testing.assert_close(forces[:, floor], torch.zeros(3, 3))
    for name, row in layout.body_index.items():
        torch.testing.assert_close(
            forces[row, layout.partner_index[f"robot/{name}"]], torch.zeros(3)
        )

    # Dynamic projectile vs foot, then simultaneous terrain + projectile.
    data.qpos[7:10], data.qpos[21:24] = [1, 0, 1], [0.15, 0, 1]
    forces = read()
    assert forces[left, cube, 0] < -1
    torch.testing.assert_close(forces[:, floor], torch.zeros(3, 3))
    data.qpos[:3], data.qpos[21:24] = [0, 0, 0.09], [0.15, 0, 0.09]
    forces = read()
    assert forces[left, floor, 2] > 1
    assert forces[left, cube, 0] < -1
    torch.testing.assert_close(forces[left, floor, :2], torch.zeros(2))

    # World-attached obstacle is not terrain, even though body ID is zero.
    data.qpos[:3], data.qpos[21:24] = [2.85, 0, 1], [2, 0, 1]
    forces = read()
    assert data.ncon > 0
    torch.testing.assert_close(forces[:, floor], torch.zeros(3, 3))


def test_mujoco_wall_plane_is_not_floor():
    mujoco = pytest.importorskip("mujoco")
    from protomotions.simulator.mujoco.simulator import MujocoSimulator

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="wall" type="plane" size="3 3 .1" euler="0 90 0"/>
      <body name="foot" pos=".09 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")
    sim = object.__new__(MujocoSimulator)
    sim.model, sim.data = model, mujoco.MjData(model)
    sim.contact_layout = ContactLayout.build(["foot"], 0, 0)
    mujoco.mj_forward(model, sim.data)
    assert sim.data.ncon > 0
    with pytest.raises(ValueError, match="horizontal world ground plane"):
        sim._get_simulator_contact_forces()


def test_mujoco_subset_rows_and_robot_only_partner_need_no_ground_plane():
    mujoco = pytest.importorskip("mujoco")
    from protomotions.simulator.mujoco.simulator import MujocoSimulator

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="torso" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
      <body name="hand" pos=".15 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sim = object.__new__(MujocoSimulator)
    sim.model, sim.data = model, data
    sim.contact_layout = ContactLayout.build(
        ["torso", "hand"],
        0,
        0,
        contact_bodies=["hand"],
        contact_partners=["robot/torso"],
    )
    forces = sim._get_simulator_contact_forces()
    assert forces.shape == (1, 1, 1, 3)
    assert forces[0, 0, 0, 0] > 1
    torch.testing.assert_close(forces[0, 0, 0, 1:], torch.zeros(2))


def test_complete_motion_labels_require_all_rows_and_partners():
    from protomotions.tests.test_base_simulator_lifecycle import _sim

    sim = _sim(contacts=True)
    assert sim.has_complete_contact_labels
    sim.contact_layout = ContactLayout.build(
        ["root", "foot"], 0, 2, contact_bodies=["foot"]
    )
    assert not sim.has_complete_contact_labels
    sim.contact_layout = ContactLayout.build(
        ["root", "foot"], 0, 2, contact_partners=["terrain"]
    )
    assert not sim.has_complete_contact_labels
    sim.contacts_enabled = False
    assert not sim.has_complete_contact_labels


def _isaaclab_method(name):
    """Execute a production reader without importing the GPU SDK."""
    path = Path(__file__).parents[1] / "simulator/isaaclab/simulator.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "IsaacLabSimulator"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace = {"torch": torch, "wp": SimpleNamespace(to_torch=lambda x: x)}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


def test_isaaclab_preserves_every_partner_and_common_body_order():
    matrix = torch.arange(3 * 3 * 6 * 3).reshape(9, 6, 3).float()
    read_dt = []

    def read(dt):
        read_dt.append(dt)
        return matrix

    forces = _isaaclab_method("_get_simulator_contact_forces")(
        SimpleNamespace(
            num_envs=3,
            contact_layout=ContactLayout.build(["c", "a", "b"], 1, 1),
            _sim=SimpleNamespace(get_physics_dt=lambda: 0.002),
            _contact_matrix_view=SimpleNamespace(get_contact_force_matrix=read),
        )
    )
    assert forces.shape == (3, 3, 6, 3)
    assert read_dt == [0.002]
    torch.testing.assert_close(forces, matrix.reshape(3, 3, 6, 3))
    matrix.zero_()
    assert forces.abs().sum() > 0  # independent of the mutable PhysX view


def test_isaaclab_patterns_preserve_env_order_and_isolate_partners():
    from protomotions.simulator.isaaclab.utils.contact_matrix import (
        contact_view_patterns,
    )

    layout = ContactLayout.build(["b", "a"], 2, 1)
    envs = ["/World/envs/env_12", "/World/envs/env_2"]
    sources, filters = contact_view_patterns(layout, {"b": "b", "a": "b/a"}, envs)
    assert sources == [f"{env}/Robot/{link}" for env in envs for link in ("b", "b/a")]
    for env_id, env in enumerate(envs):
        expected = [
            "/World/ground/terrain/mesh",
            f"{env}/Robot/b",
            f"{env}/Robot/b/a",
            f"{env}/Object_0",
            f"{env}/Object_1",
            f"{env}/Projectile_0",
        ]
        assert filters[2 * env_id] == filters[2 * env_id + 1] == expected
    with pytest.raises(KeyError, match="a"):
        contact_view_patterns(layout, {"b": "b"}, envs)


def test_isaaclab_allocates_only_selected_sources_and_counterparts():
    from protomotions.simulator.isaaclab.utils.contact_matrix import (
        contact_view_patterns,
    )

    layout = ContactLayout.build(
        ["torso", "foot"],
        100,
        1,
        contact_bodies=["foot"],
        contact_partners=["robot/torso", "object/73"],
    )
    sources, filters = contact_view_patterns(
        layout, {"foot": "torso/foot", "torso": "torso"}, ["/World/envs/env_0"]
    )
    assert sources == ["/World/envs/env_0/Robot/torso/foot"]
    assert filters == [["/World/envs/env_0/Robot/torso", "/World/envs/env_0/Object_73"]]


def test_contact_matrix_lifecycle_and_subset_ownership():
    from protomotions.tests.test_base_simulator_lifecycle import _sim

    sim = _sim(contacts=True)
    measured = torch.arange(2 * 2 * 5 * 3).reshape(2, 2, 5, 3).float()
    calls = []

    def read():
        calls.append(sim.physics_steps)
        return measured.clone()

    sim._get_simulator_contact_forces = read
    sim._initialize_with_markers({})
    torch.testing.assert_close(sim.get_contact_forces(), torch.zeros_like(measured))
    sim.step(torch.zeros(2, 2))
    assert calls == [0, 1]  # one validation read, then once per control step
    snapshot = sim.get_contact_forces()
    subset = sim.get_contact_forces(torch.tensor([1, 0]))
    torch.testing.assert_close(snapshot, measured)
    torch.testing.assert_close(subset, measured.flip(0))
    state = sim.get_default_robot_reset_state()
    sim.reset_envs(state, env_ids=torch.tensor([0]))
    torch.testing.assert_close(
        sim.get_contact_forces()[0], torch.zeros_like(measured[0])
    )
    torch.testing.assert_close(sim.get_contact_forces()[1], measured[1])
    torch.testing.assert_close(snapshot, measured)
    torch.testing.assert_close(subset, measured.flip(0))
    sim.step(torch.zeros(2, 2))
    torch.testing.assert_close(sim.get_contact_forces(), measured)


def test_unsupported_backend_fails_at_build_and_disabled_api_fails_explicitly():
    from protomotions.tests.test_base_simulator_lifecycle import _sim

    with pytest.raises(NotImplementedError, match="partner-resolved"):
        _sim(contacts=True)._initialize_with_markers({})
    with pytest.raises(ValueError, match="robot.contact_bodies"):
        _sim().get_contact_forces()


def test_wrong_backend_axis_shape_fails_at_build():
    from protomotions.tests.test_base_simulator_lifecycle import _sim

    sim = _sim(contacts=True)
    sim._get_simulator_contact_forces = lambda: torch.zeros(2, 2, 4, 3)
    with pytest.raises(ValueError, match="Contact matrix shape"):
        sim._initialize_with_markers({})


@pytest.mark.parametrize("enabled", [False, True])
def test_environment_matrix_is_current_only_and_keeps_subset_order(enabled):
    from protomotions.envs.base_env.build import declare_owners
    from protomotions.envs.component_manager import ComponentManager
    from protomotions.tests.test_base_env_helpers import _make_env

    env = _make_env(history=True)
    env.simulator.contacts_enabled = enabled
    env.simulator.contact_layout = ContactLayout.build(["root", "foot"], 0, 0)
    full = torch.arange(3 * 2 * 3 * 3).reshape(3, 2, 3, 3).float()
    calls = []

    def forces(env_ids):
        calls.append(env_ids)
        return full[env_ids]

    env.simulator.get_contact_forces = forces
    # The contact publishers join the publish list at bind time, so rebind
    # after switching the simulator's contact reporting.
    declare_owners(env)
    subset = torch.tensor([2, 0])
    context = dict(env._publish(subset))
    context["mimic.ref_state.rigid_body_contacts"] = torch.zeros(2, 2)
    components = {"foot_floor": _terrain_foot_reward(["foot"], ["root", "foot"])}
    if enabled:
        torch.testing.assert_close(calls[0], subset)
        torch.testing.assert_close(context["contact_forces"], full[subset])
        assert context["contact_partner_index"] == {
            "terrain": 0,
            "robot/root": 1,
            "robot/foot": 2,
        }
        ComponentManager.validate_bindings(None, components, context, group="reward")
    else:
        assert calls == []
        assert "prev_contact_force_magnitudes" not in context
        with pytest.raises(ValueError, match="foot_floor.*contact_forces"):
            ComponentManager.validate_bindings(
                None, components, context, group="reward"
            )
    assert "current.contact_forces" not in context
    assert "historical.contact_forces" not in context


def test_terrain_reward_selects_partner_and_feet_with_soft_reference_labels():
    from protomotions.envs.rewards.regularization import compute_contact_match_rew

    matrix = torch.full((1, 3, 5, 3), 1000.0)
    # Reorder feet. Non-terrain columns deliberately have huge forces.
    matrix[:, :, 0] = 0
    matrix[0, 2, 0, 2], matrix[0, 0, 0, 2] = 0.02, 0.01
    context = SimpleNamespace(
        contact_forces=matrix,
        contact_partner_index={"terrain": 0},
        contact_body_index={"right": 0, "hand": 1, "left": 2},
        current=SimpleNamespace(rigid_body_contacts=torch.ones(1, 3)),
        mimic=SimpleNamespace(
            ref_state=SimpleNamespace(
                rigid_body_contacts=torch.tensor([[0.4, 1.0, 1.0]])
            )
        ),
        contact_body_ids=torch.tensor([2, 0]),
    )
    reward = _terrain_foot_reward(["left", "right"], ["right", "hand", "left"])
    torch.testing.assert_close(
        reward.static_params["contact_body_ids"], torch.tensor([2, 0])
    )
    assert reward.get_bindings_dict() == {
        "contact_forces": "contact_forces",
        "ref_contacts": "mimic.ref_state.rigid_body_contacts",
        "contact_body_index": "contact_body_index",
        "contact_partner_index": "contact_partner_index",
    }
    torch.testing.assert_close(reward.compute(context), torch.tensor([0.4]))
    general_context = SimpleNamespace(**vars(context))
    general_context.contact_forces = matrix[:, [2, 0]]
    general = MdpComponent.from_kernel(
        compute_contact_match_rew, meta=RewardMeta(weight=-0.1)
    )
    torch.testing.assert_close(general.compute(general_context), torch.tensor([0.6]))
    assert reward.meta.zero_during_grace_period is True
    assert reward.meta.weight == -0.1

    # The reward resolves its own reference and matrix indices by name rather
    # than depending on another consumer's reference-index selection.
    context.contact_body_ids = torch.tensor([1])
    torch.testing.assert_close(reward.compute(context), torch.tensor([0.4]))
    del context.contact_body_ids
    torch.testing.assert_close(reward.compute(context), torch.tensor([0.4]))

    # Labels come from the live kinematic stream in the same common body order,
    # not from the simulated flags, future targets, or a cached reference frame.
    context.mimic.ref_state.rigid_body_contacts = torch.tensor([[1.0, 0.0, 1.0]])
    torch.testing.assert_close(reward.compute(context), torch.tensor([1.0]))


@pytest.mark.parametrize(
    ("names", "message"),
    [(["left", "left"], "a list of distinct"), (["missing"], "missing")],
)
def test_terrain_foot_matching_rejects_invalid_body_selection(names, message):
    # A repeated selection is refused by MdpComponent's required-key rule; a
    # name the robot lacks, by build_body_ids_tensor's lookup.
    with pytest.raises(ValueError, match=message):
        _terrain_foot_reward(names, ["right", "left"])


@pytest.mark.parametrize("threshold", [-0.5, float("inf"), float("nan")])
def test_terrain_foot_matching_refuses_a_threshold_no_force_compares_with(threshold):
    """The finiteness check the retired factory made is the kernel's: a
    negative, infinite or NaN threshold is refused at the first compute."""
    context = SimpleNamespace(
        contact_forces=torch.zeros(1, 3, 1, 3),
        contact_partner_index={"terrain": 0},
        contact_body_index={"right": 0, "hand": 1, "left": 2},
        mimic=SimpleNamespace(
            ref_state=SimpleNamespace(rigid_body_contacts=torch.zeros(1, 3))
        ),
    )
    reward = _terrain_foot_reward(
        ["left", "right"], ["right", "hand", "left"], force_threshold=threshold
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        reward.compute(context)


def _fall_context():
    return SimpleNamespace(
        current=SimpleNamespace(rigid_body_pos=torch.zeros(1, 2, 3)),
        contact_forces=torch.full((1, 2, 2, 3), 100.0),
        contact_body_index={"torso": 0, "foot": 1},
        contact_partner_index={"object/0": 0, "terrain": 1},
        ground_heights=torch.zeros(1),
        progress_buf=torch.tensor([5]),
    )


def test_named_fall_termination_refuses_an_empty_selection_at_compute():
    """``body_names=[]`` builds (the required-keys check requires nothing of an
    empty list) and would never fire; ``partner_names=[]`` would die indexing
    the matrix. Both are refused by name at the first compute, as the retired
    factories refused them at construction."""
    no_bodies = _fall_termination([], ["foot", "torso"], ["terrain"])
    with pytest.raises(ValueError, match="body_names must name at least one contact"):
        no_bodies.compute(_fall_context())
    no_partners = _fall_termination(["torso"], ["foot", "torso"], [])
    with pytest.raises(
        ValueError, match="partner_names must name at least one contact"
    ):
        no_partners.compute(_fall_context())


def test_contact_rows_refuses_empty_and_repeated_names():
    from protomotions.envs.contact_features import contact_rows

    index = {"torso": 0, "foot": 1}
    assert contact_rows(index, ["foot", "torso"], "body_names") == [1, 0]
    with pytest.raises(ValueError, match="body_names must name at least one contact"):
        contact_rows(index, [], "body_names")
    with pytest.raises(
        ValueError, match=r"partner_names must not repeat a name: \['foot'\]"
    ):
        contact_rows(index, ["foot", "torso", "foot"], "partner_names")


def test_terrain_reward_refuses_an_empty_foot_selection_at_compute():
    """A terrain-foot match over no feet returned zero mismatch for ever; it is
    refused by name instead."""
    reward = _terrain_foot_reward([], ["right", "hand", "left"])
    context = SimpleNamespace(
        contact_forces=torch.zeros(1, 3, 1, 3),
        contact_partner_index={"terrain": 0},
        contact_body_index={"right": 0, "hand": 1, "left": 2},
        mimic=SimpleNamespace(
            ref_state=SimpleNamespace(rigid_body_contacts=torch.ones(1, 3))
        ),
        contact_body_ids=torch.tensor([], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="body_names must name at least one contact"):
        reward.compute(context)


def test_named_fall_termination_refuses_a_partner_pattern_at_compute():
    """A glob in ``partner_names`` passes the build's required-keys check (it
    accepts patterns) but addresses no column; the kernel refuses it by name
    instead of failing as a KeyError."""
    component = _fall_termination(["torso"], ["foot", "torso"], ["ter*"])
    ctx = SimpleNamespace(
        current=SimpleNamespace(rigid_body_pos=torch.ones(1, 2, 3)),
        contact_forces=torch.zeros(1, 2, 2, 3),
        contact_body_index={"torso": 0, "foot": 1},
        contact_partner_index={"object/0": 0, "terrain": 1},
        ground_heights=torch.zeros(1),
        progress_buf=torch.tensor([5]),
    )
    with pytest.raises(ValueError, match="partner_names must be exact contact names"):
        component.compute(ctx)
