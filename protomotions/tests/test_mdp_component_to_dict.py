# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``MdpComponent.to_dict`` serializes the saved config form deterministically.

Every rank of a multi-GPU run, and every restart, is its own interpreter. A
serialized config that embeds a memory address differs between them, so any
comparison or hash of saved configs across processes fails. ``to_dict`` names a
kernel by ``__name__`` (a callable object by its class), emits reward metadata
and required keys, and carries no compile flag.
"""

import os
import subprocess
import sys
from pathlib import Path

import torch

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.meta import RewardMeta
from protomotions.envs.terminations.tracking import max_body_pos_error

REPO_ROOT = Path(__file__).resolve().parents[2]

_TO_DICT_SCRIPT = """
import json
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.terminations.tracking import max_body_pos_error

component = MdpComponent(
    compute_func=max_body_pos_error,
    dynamic_vars={},
    static_params={"scale": 0.6},
)
print(json.dumps(component.to_dict(), sort_keys=True))
"""


def _name(func) -> str:
    return MdpComponent(compute_func=func, dynamic_vars={}).to_dict()["compute_func"]


def test_plain_function_name_is_unchanged():
    """A kernel is named by its ``__name__``."""
    assert _name(max_body_pos_error) == "max_body_pos_error"


def test_callable_object_without_a_name_is_named_by_its_class():
    """A callable object (no ``__name__``) is named by its class, no address."""

    class Kernel:
        def __call__(self, x):
            return x

    name = _name(Kernel())
    assert name.endswith("Kernel")
    assert "0x" not in name


def test_to_dict_has_no_compile_flag():
    """``to_dict`` carries no compile key; eagerness is a marker on the kernel."""
    component = MdpComponent(compute_func=max_body_pos_error, dynamic_vars={})
    assert "compile" not in component.to_dict()


def test_to_dict_emits_reward_meta_as_a_dict():
    """A reward component's ``RewardMeta`` is serialized as a plain dict."""
    component = MdpComponent(
        compute_func=max_body_pos_error,
        dynamic_vars={},
        meta=RewardMeta(weight=-0.5, min_value=-1.0),
    )
    d = component.to_dict()
    assert d["meta"] == {
        "weight": -0.5,
        "multiplicative": False,
        "min_value": -1.0,
        "max_value": None,
        "zero_during_grace_period": False,
    }


def test_to_dict_omits_meta_when_absent():
    """A non-reward component serializes no ``meta`` key."""
    component = MdpComponent(compute_func=max_body_pos_error, dynamic_vars={})
    assert "meta" not in component.to_dict()


def test_to_dict_keeps_required_keys_when_present():
    """``required_keys`` survive serialization when non-empty."""

    def kernel(contacts=EnvContext.contact_forces):
        return contacts

    component = MdpComponent(
        compute_func=kernel,
        dynamic_vars={"contacts": EnvContext.contact_forces},
        required_keys={"contacts": ["left_foot"]},
    )
    assert component.to_dict()["required_keys"] == {"contacts": ["left_foot"]}


def test_to_dict_flattens_tensor_static_params_to_lists():
    """Fixed tensors in ``static_params`` serialize as lists."""
    component = MdpComponent(
        compute_func=max_body_pos_error,
        dynamic_vars={},
        static_params={"limit": torch.tensor([1.0, 2.0])},
    )
    assert component.to_dict()["limit"] == [1.0, 2.0]


def test_to_dict_matches_across_processes():
    """The saved form is byte-identical in two separate interpreters."""
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(REPO_ROOT)]
            + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        ),
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", _TO_DICT_SCRIPT],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )
        .stdout.strip()
        .splitlines()[-1]
        for _ in range(2)
    }
    assert len(outputs) == 1
