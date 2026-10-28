# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contact costs select their own bodies and partners from shared measurements."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.meta import RewardMeta
from protomotions.envs.rewards.regularization import compute_self_contacts
from protomotions.simulator.base_simulator.contacts import ContactLayout


def _self_contacts(body_names, partner_names, **overrides):
    """The recipe's binding of the self-contact cost: the two selections name
    the rows and columns it reads; the readings are the kernel's defaults."""
    return MdpComponent.from_kernel(
        compute_self_contacts,
        meta=RewardMeta(weight=-0.1),
        required_keys={
            "contact_body_index": body_names,
            "contact_partner_index": partner_names,
        },
        body_names=body_names,
        partner_names=partner_names,
        **overrides,
    )


def test_explicit_selection_ignores_other_measured_bodies_and_partners():
    layout = ContactLayout.build(["hand", "torso", "other"], 1, 1)
    forces = torch.zeros(1, 3, len(layout.partner_index), 3)
    ctx = SimpleNamespace(
        contact_forces=forces,
        contact_body_index=layout.body_index,
        contact_partner_index=layout.partner_index,
    )
    reward = _self_contacts(["torso"], ["robot/hand"])
    assert reward.get_bindings_dict() == {
        "contact_forces": "contact_forces",
        "contact_body_index": "contact_body_index",
        "contact_partner_index": "contact_partner_index",
    }
    assert reward.meta.weight == -0.1
    ComponentManager(torch.device("cpu")).validate_bindings(
        {"self_contact": reward}, ctx, group="reward"
    )

    torso = layout.body_index["torso"]
    for partner in ("terrain", "object/0", "projectile/0", "robot/other"):
        forces[0, torso, layout.partner_index[partner], 2] = 1000.0
    # Reporting more rows must not silently expand the reward.
    forces[0, layout.body_index["other"], :, 0] = 1000.0
    forces[0, layout.body_index["hand"], layout.partner_index["robot/torso"], 0] = 5
    assert reward.compute(ctx).item() == 0
    forces[0, torso, layout.partner_index["robot/hand"], 0] = -5
    assert reward.compute(ctx).item() == 1


def test_each_selected_body_counts_once_not_each_pair():
    ctx = SimpleNamespace(
        contact_forces=torch.tensor(
            [[[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], [[-5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
        ),
        contact_body_index={"hand": 0, "torso": 1},
        contact_partner_index={"robot/hand": 0, "robot/torso": 1},
    )
    reward = _self_contacts(["hand", "torso"], ["robot/hand", "robot/torso"])
    assert reward.compute(ctx).item() == 2


@pytest.mark.parametrize(
    ("bodies", "partners", "argument"),
    [
        (["torso", "torso"], ["robot/hand"], "contact_body_index"),
        (["torso"], ["robot/hand", "robot/hand"], "contact_partner_index"),
        ("torso", ["robot/hand"], "contact_body_index"),
    ],
)
def test_repeated_or_unlisted_selections_rejected(bodies, partners, argument):
    """A repeated name, or a bare string where a list of names belongs, is
    refused where the required keys are declared, naming the argument; a
    missing name is the build check's."""
    with pytest.raises(
        ValueError,
        match=f"required keys for '{argument}' must be a list of distinct",
    ):
        _self_contacts(bodies, partners)


@pytest.mark.parametrize(
    ("bodies", "partners", "overrides", "message"),
    [
        # A glob passes the build's required-keys check (which accepts
        # patterns) but addresses no row of the matrix.
        (["tor*"], ["robot/hand"], {}, "body_names must be exact contact names"),
        (["torso"], ["robot/*"], {}, "partner_names must be exact contact names"),
        # An empty selection would count zero forever instead of refusing.
        ([], ["robot/hand"], {}, "body_names must name at least one contact"),
        (["torso"], [], {}, "partner_names must name at least one contact"),
        # A partner outside the robot would count terrain contacts as self
        # contacts.
        (["torso"], ["terrain"], {}, "partner_names are robot/<link> names"),
        (["torso"], ["robot/"], {}, "partner_names are robot/<link> names"),
        (["torso"], ["robot/hand"], {"force_threshold": -1.0}, "finite and nonneg"),
        (
            ["torso"],
            ["robot/hand"],
            {"force_threshold": float("inf")},
            "finite and nonnegative",
        ),
    ],
)
def test_patterns_foreign_partners_and_bad_thresholds_refused_at_compute(
    bodies, partners, overrides, message
):
    """The checks the retired factory made at construction are the kernel's:
    a pattern where exact names belong, a partner that is not a robot link, a
    threshold no force can be compared with. Each is refused at the first
    compute with the argument named, never a KeyError or a silent count."""
    layout = ContactLayout.build(["hand", "torso", "other"], 1, 1)
    ctx = SimpleNamespace(
        contact_forces=torch.zeros(1, 3, len(layout.partner_index), 3),
        contact_body_index=layout.body_index,
        contact_partner_index=layout.partner_index,
    )
    reward = _self_contacts(bodies, partners, **overrides)
    with pytest.raises(ValueError, match=message):
        reward.compute(ctx)


def test_threshold_multiple_partners_and_live_update():
    forces = torch.zeros(3, 1, 2, 3)
    forces[:, 0, 0, 0] = torch.tensor([1.0, 1.01, 10.0])
    forces[:, 0, 1, 0] = -forces[:, 0, 0, 0]
    ctx = SimpleNamespace(
        contact_forces=forces,
        contact_body_index={"torso": 0},
        contact_partner_index={"robot/left": 0, "robot/right": 1},
    )
    reward = _self_contacts(["torso"], ["robot/left", "robot/right"])
    # Opposing forces from different partners must not cancel; equality is zero.
    torch.testing.assert_close(reward.compute(ctx), torch.tensor([0.0, 1.0, 1.0]))
    forces.zero_()
    torch.testing.assert_close(reward.compute(ctx), torch.zeros(3))


def test_disabled_reporting_is_not_a_fake_zero_reward():
    reward = _self_contacts(["torso"], ["robot/hand"])
    with pytest.raises(AttributeError, match="contact_forces"):
        reward.compute(SimpleNamespace())


@pytest.mark.parametrize("missing", ["body", "partner"])
def test_missing_requested_measurements_fail_at_setup(missing):
    reward = _self_contacts(["torso"], ["robot/hand"])
    ctx = SimpleNamespace(
        contact_forces=torch.zeros(1, 1, 1, 3),
        contact_body_index={"torso": 0},
        contact_partner_index={"robot/hand": 0},
    )
    assert reward.required_keys == {
        "contact_body_index": ["torso"],
        "contact_partner_index": ["robot/hand"],
    }
    manager = ComponentManager(torch.device("cpu"))
    manager.validate_bindings({"self_contact": reward}, ctx, group="reward")
    if missing == "body":
        ctx.contact_body_index = {"other": 0}
    else:
        # Having some robot partner is not sufficient: it must be the requested one.
        ctx.contact_partner_index = {"robot/other": 0}
    with pytest.raises(
        ValueError, match="reward component 'self_contact'.*not registered"
    ):
        manager.validate_bindings({"self_contact": reward}, ctx, group="reward")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compact_layout_compiles(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    reward = _self_contacts(["hand"], ["robot/torso"])
    ctx = SimpleNamespace(
        contact_forces=torch.tensor(
            [[[[0.0, 0.0, 1000.0], [0.0, 0.0, 10.0]]]], device=device
        ),
        contact_body_index={"hand": 0},
        contact_partner_index={"object/7": 0, "robot/torso": 1},
    )
    resolved, params = reward.resolve_args(ctx)
    compiled = torch.compile(
        reward.compute_func,
        backend="inductor" if device == "cuda" else "eager",
        fullgraph=True,
    )
    torch.testing.assert_close(
        compiled(**resolved, **params), torch.ones(1, device=device)
    )
    torch.testing.assert_close(compiled(**resolved, **params), reward.compute(ctx))


def test_real_mujoco_matrix_cost_ignores_ground_and_projectile():
    mujoco = pytest.importorskip("mujoco")
    from protomotions.simulator.mujoco.simulator import MujocoSimulator

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="floor" type="plane" size="3 3 .1"/>
      <body name="left" pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
      <body name="right" pos="1 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
      <body name="torso" pos="4 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
      <body name="projectile_0" pos="2 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")
    names = ["right", "torso", "left"]  # Cyclic common order, not simulator order.
    sim = object.__new__(MujocoSimulator)
    sim.model, sim.data = model, mujoco.MjData(model)
    sim.config = SimpleNamespace(projectile=SimpleNamespace(num_projectiles=1))
    sim.contact_layout = ContactLayout.build(names, 0, 1)
    reward = _self_contacts(["right", "torso"], ["robot/left"])

    def cost():
        mujoco.mj_forward(model, sim.data)
        return reward.compute(
            SimpleNamespace(
                contact_forces=sim._get_simulator_contact_forces(),
                contact_body_index=sim.contact_layout.body_index,
                contact_partner_index=sim.contact_layout.partner_index,
            )
        ).item()

    assert cost() == 0  # Real floor contacts only.
    sim.data.qpos[7] = 0.15
    assert cost() == 1  # Left/right collision: only the right body is scored.
    sim.data.qpos[7], sim.data.qpos[21:24] = 1, [0.15, 0, 0.09]
    assert cost() == 0  # Projectile is not a selected partner.
