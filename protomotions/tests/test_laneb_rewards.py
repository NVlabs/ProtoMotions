# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Lane-B additions: foot-force penalty, fall penalty, delay DR config."""

import torch

from protomotions.envs.rewards import (
    compute_foot_contact_force_penalty,
    compute_fall_penalty,
    compute_reference_contact_liftoff_penalty,
)
from protomotions.envs.mdp_component import MdpComponent, _METADATA_KEYS
from protomotions.envs.component_factories import (
    foot_contact_force_penalty_factory,
    fall_penalty_factory,
    reference_contact_liftoff_penalty_factory,
)
from protomotions.simulator.base_simulator.config import (
    DelayDomainRandomizationConfig,
    DomainRandomizationConfig,
)


def test_foot_contact_force_penalty_kernel():
    # 3 envs, 5 bodies; feet are bodies 1 and 3.
    forces = torch.tensor(
        [
            [10.0, 100.0, 10.0, 200.0, 10.0],   # both feet below threshold -> 0
            [10.0, 500.0, 10.0, 450.0, 10.0],   # excess: (500-400)+(450-400)=150
            [10.0, 900.0, 10.0, 10.0, 10.0],    # excess: (900-400)=500
        ]
    )
    contact_ids = torch.tensor([1, 3])
    out = compute_foot_contact_force_penalty(forces, contact_ids, force_threshold=400.0)
    assert torch.allclose(out, torch.tensor([0.0, 150.0, 500.0])), out
    # Non-foot body (0) with huge force must NOT contribute.
    forces2 = forces.clone()
    forces2[0, 0] = 10000.0
    out2 = compute_foot_contact_force_penalty(forces2, contact_ids, force_threshold=400.0)
    assert torch.allclose(out2, torch.tensor([0.0, 150.0, 500.0])), out2


def test_fall_penalty_kernel():
    # anchor at z; ref anchor z at 1.0; anchor_idx=0. height error = |z - ref_z|.
    current_anchor_pos = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.6], [0.0, 0.0, 0.8]])
    ref = torch.zeros(3, 2, 3)
    ref[:, 0, 2] = 1.0  # ref anchor (body 0) height = 1.0
    out = compute_fall_penalty(current_anchor_pos, ref, anchor_idx=0, height_threshold=0.25)
    # errors: 0.0, 0.4(>0.25 -> fall), 0.2(<=0.25 -> ok)
    assert torch.allclose(out, torch.tensor([0.0, 1.0, 0.0])), out


def test_reference_contact_liftoff_penalty_kernel():
    # Feet are bodies 1 and 3. Only a simulated contact->no-contact transition
    # during reference stance is penalized.
    sim_contacts = torch.tensor(
        [
            [False, False, False, True],
            [False, False, False, False],
            [False, False, False, False],
        ]
    )
    ref_contacts = torch.tensor(
        [
            [False, True, False, True],
            [False, False, False, True],
            [False, True, False, True],
        ]
    )
    historical_contacts = torch.tensor(
        [
            [[True, True]],    # left foot lifts unnecessarily, right stays planted.
            [[True, True]],    # left foot lifts during ref swing, right lifts in stance.
            [[False, True]],   # left was already airborne, right lifts in stance.
        ]
    )
    out = compute_reference_contact_liftoff_penalty(
        sim_contacts,
        ref_contacts,
        contact_body_ids=torch.tensor([1, 3]),
        historical_body_contacts=historical_contacts,
    )
    assert torch.allclose(out, torch.tensor([1.0, 1.0, 1.0])), out

    try:
        compute_reference_contact_liftoff_penalty(
            sim_contacts,
            None,
            contact_body_ids=torch.tensor([1, 3]),
            historical_body_contacts=historical_contacts,
        )
        assert False, "missing reference contacts must fail loudly"
    except ValueError:
        pass


def test_factories_metadata_split():
    ff = foot_contact_force_penalty_factory(weight=-2e-5, min_value=-0.3, force_threshold=350.0)
    assert isinstance(ff, MdpComponent)
    params = ff.get_params()
    assert params["weight"] == -2e-5
    assert params["min_value"] == -0.3
    assert params["force_threshold"] == 350.0
    # func_params (what the kernel actually receives) must strip metadata but keep force_threshold.
    func_params = {k: v for k, v in ff.static_params.items() if k not in _METADATA_KEYS}
    assert "weight" not in func_params and "min_value" not in func_params
    assert "force_threshold" in func_params

    fp = fall_penalty_factory(weight=-3.0, height_threshold=0.3)
    assert isinstance(fp, MdpComponent)
    fp_func = {k: v for k, v in fp.static_params.items() if k not in _METADATA_KEYS}
    assert "weight" not in fp_func
    assert "height_threshold" in fp_func

    rc = reference_contact_liftoff_penalty_factory(
        weight=-0.07,
        min_value=-0.25,
        ref_contact_threshold=0.6,
    )
    assert isinstance(rc, MdpComponent)
    assert rc.dynamic_vars["historical_body_contacts"].path == "historical.body_contacts"
    rc_func = {k: v for k, v in rc.static_params.items() if k not in _METADATA_KEYS}
    assert "weight" not in rc_func and "min_value" not in rc_func
    assert rc_func["ref_contact_threshold"] == 0.6


def test_delay_config_and_backcompat():
    d = DelayDomainRandomizationConfig(
        action_delay_steps=(0, 2), observation_delay_steps=(1, 3)
    )
    assert d.has_delay() and d.has_action_delay() and d.has_observation_delay()
    assert d.max_action_delay() == 2 and d.max_observation_delay() == 3

    off = DelayDomainRandomizationConfig()
    assert not off.has_delay()

    # Back-compat: default DomainRandomizationConfig has no delay.
    dr = DomainRandomizationConfig()
    assert getattr(dr, "delay", None) is None

    dr2 = DomainRandomizationConfig(delay=d)
    assert getattr(dr2, "delay", None) is d

    # Validation.
    for bad in [(-1, 0), (3, 1)]:
        try:
            DelayDomainRandomizationConfig(action_delay_steps=bad)
            assert False, "should have raised"
        except ValueError:
            pass


if __name__ == "__main__":
    test_foot_contact_force_penalty_kernel()
    test_fall_penalty_kernel()
    test_reference_contact_liftoff_penalty_kernel()
    test_factories_metadata_split()
    test_delay_config_and_backcompat()
    print("ALL LANEB TESTS PASSED")
