# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sensed`` repoints an actor's privileged state bindings at a sensor source.

A ``current``/``historical`` binding is rewritten to the source's namespace;
control namespaces, ``dt`` and the rest pass unchanged; a binding to a name only
the simulator produces (contacts) is refused.
"""

import pytest

from protomotions.envs.context_sources import PRIVILEGED_ONLY_PATHS, sensed
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent


def _obs(
    body_pos=EnvContext.current.rigid_body_pos,
    past=EnvContext.historical.rigid_body_pos,
    ref=EnvContext.mimic.ref_state.rigid_body_pos,
    dt=EnvContext.dt,
):
    return body_pos


def test_sensed_rewrites_current_and_historical_to_the_source():
    """current/historical bindings move to the source's namespaces."""
    out = sensed({"o": MdpComponent.from_kernel(_obs)}, "noisy")
    assert out["o"].get_bindings_dict() == {
        "body_pos": "noisy.rigid_body_pos",
        "past": "noisy_historical.rigid_body_pos",
        "ref": "mimic.ref_state.rigid_body_pos",
        "dt": "dt",
    }


def test_sensed_privileged_is_identity():
    """sensed(..., 'privileged') returns the components unchanged."""
    component = MdpComponent.from_kernel(_obs)
    out = sensed({"o": component}, "privileged")
    assert out["o"] is component


def _contact_obs(contacts=EnvContext.current.rigid_body_contacts):
    return contacts


def test_sensed_refuses_a_simulator_only_binding_that_would_move():
    """A current/historical contact binding is refused for a sensor source."""
    with pytest.raises(ValueError, match="only the simulator produces"):
        sensed({"c": MdpComponent.from_kernel(_contact_obs)}, "noisy")


def _force_obs(forces=EnvContext.contact_forces):
    return forces


def test_sensed_refuses_a_simulator_only_binding_that_would_stay_put():
    """A top-level contact binding (``contact_forces``) never moves, but a
    sensor-sourced component may not read it either: it would sit beside the
    sensor estimates as the one privileged input (the guard the max-coords
    contact factory used to apply)."""
    with pytest.raises(ValueError, match="only the simulator produces"):
        sensed({"c": MdpComponent.from_kernel(_force_obs)}, "clean")
    component = MdpComponent.from_kernel(_force_obs)
    assert sensed({"c": component}, "privileged")["c"] is component


def test_sensed_privileged_is_identity_even_for_a_contact_binding():
    """sensed(..., 'privileged') never refuses: nothing moves, so it is the
    identity even when a component binds a simulator-only contact name."""
    component = MdpComponent.from_kernel(_contact_obs)
    out = sensed({"c": component}, "privileged")
    assert out["c"] is component


def test_sensed_binds_the_declared_sensor_name():
    """Repointing a binding returns the sensor name's own declaration, so its
    requirement and its deploy slot are the declared ones: ``noisy.root_pos``
    needs localization although ``current.root_pos`` needs nothing, and the
    config-only check can refuse it on a suite without a localization body."""
    component = MdpComponent(
        lambda root_pos: root_pos, {"root_pos": EnvContext.current.root_pos}
    )
    moved = sensed({"o": component}, "noisy")["o"].dynamic_vars["root_pos"]
    assert moved is EnvContext.noisy.root_pos
    assert moved.path == "noisy.root_pos"
    assert moved.requires == ("localization",)
    assert EnvContext.current.root_pos.requires == ()
    assert moved.deploy is EnvContext.noisy.root_pos.deploy


def test_privileged_only_paths_include_the_contact_names():
    """The refused set names the contact rows, columns and forces."""
    assert {"contact_forces", "contact_body_ids", "current.rigid_body_contacts"} <= set(
        PRIVILEGED_ONLY_PATHS
    )


def test_sensed_leaves_non_state_namespaces_untouched():
    """A component with only control and top-level bindings is returned as-is."""

    def task_obs(speed=EnvContext.steering.tar_speed, dt=EnvContext.dt):
        return speed

    component = MdpComponent.from_kernel(task_obs)
    out = sensed({"t": component}, "noisy")
    assert out["t"] is component


def test_sensed_refuses_an_undeclared_sensor_twin_as_a_value_error():
    """A current name with no declared twin under the sensor namespace is a
    recipe mistake like the others here, so it is refused with the same kind
    of error, carrying the vocabulary's own sentence."""
    component = MdpComponent(lambda f: f, {"f": EnvContext.current.dof_forces})
    with pytest.raises(ValueError, match="context has no 'noisy.dof_forces'"):
        sensed({"forces": component}, "noisy")
