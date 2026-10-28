# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step selections as typed kernel parameters.

A kernel that reads a window declares the slots it takes with a parameter
annotated ``StepSelection``; the recipe's value (an int N or a sequence) is
normalised to a tuple once, at construction, and the manager checks at build
that it never reaches past the slots the window publishes. A ``FieldPath``
default on a kernel is bound by a direct construction as well, so a kernel
never sees the path object.
"""

from typing import Optional

import pytest
import torch
from torch import Tensor

from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.context import FrozenContext
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.steps import (
    StepSelection,
    is_window_path,
    normalize_steps,
    required_history_depth,
    select_steps,
    step_ages,
)


def reads_history(window: Tensor, history_steps: StepSelection) -> Tensor:
    return select_steps(window, history_steps).flatten(1)


def reads_whole_window_by_default(
    window: Tensor, future_steps: Optional[StepSelection] = None
) -> Tensor:
    return select_steps(window, future_steps).flatten(1)


def reads_the_first_slot_by_default(
    window: Tensor, future_steps: StepSelection = (1,)
) -> Tensor:
    return select_steps(window, future_steps).flatten(1)


def reads_current_state(dof_pos: Tensor) -> Tensor:
    return dof_pos


def reads_one_slot(previous_action: Tensor) -> Tensor:
    return previous_action


def anchored(
    body_pos: Tensor, anchor_body_index: int = EnvContext.current.anchor_idx
) -> Tensor:
    return body_pos[:, anchor_body_index]


HISTORY = {"window": EnvContext.historical.dof_pos}


# --------------------------------------------------------------------------
# normalize_steps / select_steps / step_ages / is_window_path
# --------------------------------------------------------------------------


def test_normalize_steps_expands_an_int_and_keeps_a_sequence_in_order():
    assert normalize_steps(3) == (1, 2, 3)
    assert normalize_steps(1) == (1,)
    assert normalize_steps([4, 1, 9]) == (4, 1, 9)
    assert normalize_steps((2,)) == (2,)
    assert normalize_steps(range(1, 4)) == (1, 2, 3)
    assert normalize_steps(torch.tensor([2, 5]).tolist()) == (2, 5)


@pytest.mark.parametrize(
    "value,error,words",
    [
        (None, TypeError, "cannot be None"),
        (True, TypeError, "not True"),
        ("3", TypeError, "not '3'"),
        (2.0, TypeError, "not 2.0"),
        (0, ValueError, "must be positive"),
        (-1, ValueError, "must be positive"),
        ([], ValueError, "cannot be empty"),
        ([1, 0], ValueError, "1-based and positive"),
        ([1, 1], ValueError, "names a step twice"),
        ([1, 2.0], TypeError, "every step is an int"),
        ([1, True], TypeError, "every step is an int"),
    ],
)
def test_normalize_steps_refuses_what_is_not_a_selection(value, error, words):
    with pytest.raises(error, match=words):
        normalize_steps(value)


def test_select_steps_views_a_leading_run_and_gathers_anything_else():
    window = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    assert select_steps(window, None) is window
    leading = select_steps(window, (1, 2, 3))
    assert torch.equal(leading, window[:, :3])
    assert leading.data_ptr() == window.data_ptr()  # a view, not a copy
    assert torch.equal(select_steps(window, (3, 1)), window[:, [2, 0]])
    assert torch.equal(select_steps(window, (5,)), window[:, 4:5])
    assert torch.equal(select_steps(window, (2,), dim=0), window[1:2])


def test_step_ages_are_float32_seconds_and_refuse_an_unknown_origin():
    ages = step_ages((1, 3, 8), 0.5, "current", torch.device("cpu"))
    assert ages.dtype == torch.float32
    assert torch.equal(ages, torch.tensor([0.5, 1.5, 4.0]))
    first_slot = step_ages((1, 3, 8), 0.5, "first_slot", torch.device("cpu"))
    assert torch.equal(first_slot, torch.tensor([0.0, 1.0, 3.5]))
    with pytest.raises(ValueError, match="time_origin must be one of"):
        step_ages((1,), 0.5, "previous", torch.device("cpu"))


@pytest.mark.parametrize(
    "path",
    [
        "historical.dof_pos",
        "noisy_historical.rigid_body_pos",
        "clean_historical.anchor_rot",
        "raw_action_history",
        "processed_action_history",
        "mimic.future_rigid_body_pos",
        "masked_mimic.ref_rigid_body_pos",
        "masked_mimic.local_ref_rigid_body_rot",
        "masked_mimic.time_offsets",
        "masked_mimic.target_poses_masks",
        "masked_mimic.anchor_translation_mask",
    ],
)
def test_window_paths_are_published_with_a_step_axis(path):
    assert is_window_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "current.dof_pos",
        "noisy.rigid_body_pos",
        "mimic.anchor_idx",
        "mimic.ref_rigid_body_pos",
        "masked_mimic.anchor_idx",
        "previous_action",
        "previous_processed_action",
        "historical_actions",
    ],
)
def test_single_slot_paths_are_not_windows(path):
    assert not is_window_path(path)


# --------------------------------------------------------------------------
# MdpComponent: normalisation at construction, refusal of a path, defaults
# --------------------------------------------------------------------------


def test_a_bound_step_selection_is_normalised_at_construction():
    from_int = MdpComponent(reads_history, HISTORY, {"history_steps": 3})
    assert from_int.static_params["history_steps"] == (1, 2, 3)
    assert from_int.step_selections() == {"history_steps": (1, 2, 3)}

    from_list = MdpComponent(reads_history, HISTORY, {"history_steps": [1, 3]})
    assert from_list.static_params["history_steps"] == (1, 3)
    assert from_list.get_params()["history_steps"] == (1, 3)


def test_a_step_selection_refuses_a_context_path():
    with pytest.raises(TypeError, match="StepSelection and must be a constant"):
        MdpComponent(
            reads_history,
            {**HISTORY, "history_steps": EnvContext.historical.dof_pos},
        )


def test_a_bad_step_selection_is_refused_naming_the_kernel_and_parameter():
    with pytest.raises(ValueError, match="reads_history, parameter 'history_steps'"):
        MdpComponent(reads_history, HISTORY, {"history_steps": 0})
    with pytest.raises(TypeError, match="reads_history, parameter 'history_steps'"):
        MdpComponent(reads_history, HISTORY, {"history_steps": "3"})


def test_step_selections_fall_back_to_the_kernel_default():
    whole = MdpComponent(reads_whole_window_by_default, HISTORY)
    assert whole.step_selections() == {"future_steps": None}
    assert "future_steps" not in whole.static_params

    explicit_none = MdpComponent(
        reads_whole_window_by_default, HISTORY, {"future_steps": None}
    )
    assert explicit_none.step_selections() == {"future_steps": None}
    assert explicit_none.static_params["future_steps"] is None

    first = MdpComponent(reads_the_first_slot_by_default, HISTORY)
    assert first.step_selections() == {"future_steps": (1,)}
    assert "future_steps" not in first.static_params

    overridden = MdpComponent(
        reads_the_first_slot_by_default, HISTORY, {"future_steps": 2}
    )
    assert overridden.step_selections() == {"future_steps": (1, 2)}

    assert MdpComponent(reads_current_state, {}).step_selections() == {}


def test_a_required_step_selection_left_unbound_is_refused_at_construction():
    """A forgotten ``history_steps`` fails when the component is built (so a
    saved configuration that lacks it fails at load), not at the first call."""
    with pytest.raises(
        TypeError,
        match=r"reads_history: 'history_steps' is a StepSelection with no default",
    ):
        MdpComponent(reads_history, HISTORY)
    # A kernel whose selection has a default (a tuple or None) needs no binding.
    MdpComponent(reads_whole_window_by_default, HISTORY)
    MdpComponent(reads_the_first_slot_by_default, HISTORY)


def test_a_field_path_default_is_bound_by_a_direct_construction():
    dynamic_vars = {"body_pos": EnvContext.current.rigid_body_pos}
    component = MdpComponent(anchored, dynamic_vars)
    assert component.dynamic_vars["anchor_body_index"] is EnvContext.current.anchor_idx
    assert component.get_bindings_dict() == {
        "body_pos": "current.rigid_body_pos",
        "anchor_body_index": "current.anchor_idx",
    }
    assert "anchor_body_index" not in dynamic_vars  # the caller's dict is copied

    body_pos = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3)
    ctx = FrozenContext(
        {"current.rigid_body_pos": body_pos, "current.anchor_idx": 1},
        env_ids=None,
        dt=0.1,
    )
    assert torch.equal(component.compute(ctx), body_pos[:, 1])


def test_an_explicit_anchor_binding_wins_over_the_default():
    constant = MdpComponent(
        anchored,
        {"body_pos": EnvContext.current.rigid_body_pos},
        {"anchor_body_index": 2},
    )
    assert "anchor_body_index" not in constant.dynamic_vars
    assert constant.static_params["anchor_body_index"] == 2

    rebound = MdpComponent(
        anchored,
        {
            "body_pos": EnvContext.current.rigid_body_pos,
            "anchor_body_index": EnvContext.mimic.anchor_idx,
        },
    )
    assert rebound.get_bindings_dict()["anchor_body_index"] == "mimic.anchor_idx"


# --------------------------------------------------------------------------
# ComponentManager: the window check at build
# --------------------------------------------------------------------------


def _ctx(**published):
    return FrozenContext(published, env_ids=None, dt=0.1)


def test_manager_refuses_a_selection_past_the_published_slots():
    manager = ComponentManager(torch.device("cpu"))
    four_slots = _ctx(**{"historical.dof_pos": torch.zeros(2, 4, 3)})

    fits = {"obs": MdpComponent(reads_history, HISTORY, {"history_steps": [1, 4]})}
    manager.validate_bindings(fits, four_slots, group="observation")

    reaches = {"obs": MdpComponent(reads_history, HISTORY, {"history_steps": [1, 5]})}
    with pytest.raises(
        ValueError,
        match=r"step selection \(1, 5\) reaches slot 5 but 'historical.dof_pos' "
        r"publishes 4 slots",
    ):
        manager.validate_bindings(reaches, four_slots, group="observation")


def test_manager_checks_the_kernel_default_and_leaves_non_windows_alone():
    manager = ComponentManager(torch.device("cpu"))
    # The default selection (1,) fits any window with a slot.
    one_slot = _ctx(**{"historical.dof_pos": torch.zeros(2, 1, 3)})
    manager.validate_bindings(
        {"obs": MdpComponent(reads_the_first_slot_by_default, HISTORY)},
        one_slot,
        group="observation",
    )
    # None reads the whole window, whatever its size.
    manager.validate_bindings(
        {"obs": MdpComponent(reads_whole_window_by_default, HISTORY)},
        one_slot,
        group="observation",
    )
    # A selection bound next to a name that is not a window is not compared
    # against that name's second axis.
    not_a_window = _ctx(**{"current.dof_pos": torch.zeros(2, 4, 3)})
    manager.validate_bindings(
        {
            "obs": MdpComponent(
                reads_history,
                {"window": EnvContext.current.dof_pos},
                {"history_steps": [9]},
            )
        },
        not_a_window,
        group="observation",
    )


# --------------------------------------------------------------------------
# required_history_depth
# --------------------------------------------------------------------------


def _first_lag(root):
    return {"historical": 1, "noisy_historical": 1, "previous_action": 1}.get(root)


def test_required_history_depth_is_the_deepest_slot_any_component_reads():
    components = {
        "max_coords": MdpComponent(reads_history, HISTORY, {"history_steps": [1, 3]}),
        "noisy": MdpComponent(
            reads_history,
            {"window": EnvContext.noisy_historical.dof_pos},
            {"history_steps": 2},
        ),
        "current": MdpComponent(
            reads_current_state, {"dof_pos": EnvContext.current.dof_pos}
        ),
    }
    assert required_history_depth(components, _first_lag) == 3
    assert required_history_depth({"current": components["current"]}, _first_lag) == 0
    assert required_history_depth({}, _first_lag) == 0


def test_required_history_depth_counts_a_single_slot_name_as_its_lag():
    one_slot = {
        "previous": MdpComponent(
            reads_one_slot, {"previous_action": EnvContext.previous_action}
        )
    }
    assert (
        required_history_depth(
            one_slot, _first_lag, single_slot_roots=("previous_action",)
        )
        == 1
    )
    with pytest.raises(ValueError, match="previous, argument 'previous_action'"):
        required_history_depth(one_slot, _first_lag)


def test_required_history_depth_refuses_a_history_read_without_a_selection():
    components = {
        "reads_ring": MdpComponent(
            reads_current_state, {"dof_pos": EnvContext.historical.dof_pos}
        )
    }
    with pytest.raises(ValueError, match="declares no step selection"):
        required_history_depth(components, _first_lag)


def test_the_component_owns_its_dicts_and_leaves_the_recipe_literals_alone():
    static_params = {"history_steps": 3}
    component = MdpComponent(reads_history, HISTORY, static_params)
    assert component.static_params == {"history_steps": (1, 2, 3)}
    assert static_params == {"history_steps": 3}
    assert component.static_params is not static_params
    assert component.dynamic_vars is not HISTORY
