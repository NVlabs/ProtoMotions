# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every example recipe's components serialize to a committed snapshot.

Each example recipe binds observation, reward and termination kernels. This
pins ``{recipe: {group: {name: to_dict()}}}`` for every recipe that builds
offline against a committed JSON baseline, so a later mechanical sweep that
moves a binding from a factory into a kernel default, or a metadata key from
``static_params`` into a ``RewardMeta``, is caught as a diff rather than
slipping through. Re-baseline deliberately (``PROTOMOTIONS_REBASELINE=1``) when
the change is intended; the stacks that only shuffle bindings assert equality.
"""

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.tests.support.recipe_stages import build_experiment_configs
from protomotions.tests.test_recipe_stages import (
    CONFIGS_NEED_CHECKPOINT,
    _experiment_names,
    recipe_robot,
)

BASELINE = Path(__file__).with_name("recipe_component_snapshot.json")

# The env's three MdpComponent groups, serialized in binding order.
GROUPS = ("observation_components", "reward_components", "termination_components")


def _serialize_entry(entry):
    """A component's ``to_dict``; a termination rule's cut around its component;
    anything else keeps its type name."""
    if isinstance(entry, TerminationRule):
        return {
            "component": entry.component.to_dict(),
            "threshold": entry.threshold,
            "fail_above": entry.fail_above,
            "truncation": entry.truncation,
        }
    if isinstance(entry, MdpComponent):
        return entry.to_dict()
    return type(entry).__name__


def _serialize_group(group) -> dict:
    """Serialize one component group in binding order."""
    return {name: _serialize_entry(entry) for name, entry in (group or {}).items()}


def build_snapshot() -> dict:
    """Build every offline-buildable example recipe, on the robot it requires,
    and serialize its components."""
    snapshot = {}
    for name in _experiment_names():
        # A recipe that reads another run's checkpoint while its configs build
        # cannot be snapshotted offline (build_configs also runs the config-only
        # binding check last and refuses a binding this configuration can never
        # satisfy, exactly as training would).
        if name in CONFIGS_NEED_CHECKPOINT:
            continue
        with redirect_stdout(io.StringIO()):  # the robot factories print
            configs = build_experiment_configs(
                name, "examples.experiments", recipe_robot(name)
            )
        env = configs.env
        snapshot[name] = {
            group.replace("_components", ""): _serialize_group(getattr(env, group, None))
            for group in GROUPS
        }
    return snapshot


def test_recipe_component_snapshot_matches_baseline():
    """Every recipe's bound components serialize exactly as the committed baseline."""
    snapshot = build_snapshot()
    if os.environ.get("PROTOMOTIONS_REBASELINE"):
        BASELINE.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        pytest.skip("re-baselined recipe_component_snapshot.json")
    baseline = json.loads(BASELINE.read_text())
    # Compared as JSON: a step selection is a tuple in the component and a
    # list in the file.
    snapshot = json.loads(json.dumps(snapshot, sort_keys=True))
    assert snapshot == baseline, (
        "recipe component bindings drifted from the committed snapshot; if the "
        "change is intended, re-run with PROTOMOTIONS_REBASELINE=1 and commit "
        "recipe_component_snapshot.json"
    )
