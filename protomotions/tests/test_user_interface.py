# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from protomotions.simulator.base_simulator.user_interface import (
    KeyBindingScope,
    UserInterface,
    KeyBinding,
)


def test_user_interface_rejects_duplicate_keys_with_existing_use():
    ui = UserInterface()
    ui.register_key("r", owner="env", description="Reset all environments")

    with pytest.raises(ValueError, match="R.*env.*Reset all environments"):
        ui.register_key("R", owner="target", description="Move target forward")


def test_user_interface_rejects_empty_descriptions():
    ui = UserInterface()

    with pytest.raises(ValueError, match="description"):
        ui.register_key("R", owner="env", description="")


def test_user_interface_only_exposes_state_through_registered_handles():
    ui = UserInterface()
    reset_key = ui.register_key("R", owner="env", description="Reset all envs")
    target_key = ui.register_key("W", owner="target", description="Move target")

    ui.begin_step()
    ui.handle_key_event("R", pressed=True)

    assert ui.was_pressed(reset_key)
    assert ui.is_down(reset_key)
    assert not ui.was_pressed(target_key)

    assert ui.consume_key_press(reset_key)
    assert not ui.was_pressed(reset_key)


def test_user_interface_invokes_press_callback_once_per_press_event():
    ui = UserInterface()
    calls = []
    key = ui.register_key(
        "L",
        owner="simulator",
        description="Toggle recording",
        on_press=lambda: calls.append("record"),
    )

    ui.begin_step()
    ui.handle_key_event("L", pressed=True)
    ui.handle_key_event("L", pressed=False)

    assert calls == ["record"]
    assert ui.was_pressed(key)
    assert not ui.is_down(key)


def test_user_interface_hold_is_down_but_not_repeated_press():
    ui = UserInterface()
    calls = []
    key = ui.register_key(
        "L",
        owner="simulator",
        description="Toggle recording",
        on_press=lambda: calls.append("record"),
    )

    ui.begin_step()
    ui.handle_key_event("L", pressed=True)
    ui.handle_key_event("L", pressed=True)

    assert calls == ["record"]
    assert key.pressed()
    assert key.down()

    ui.begin_step()
    ui.handle_key_event("L", pressed=True)

    assert calls == ["record"]
    assert not key.pressed()
    assert key.down()

    ui.handle_key_event("L", pressed=False)
    assert not key.down()


def test_user_interface_notifies_registration_callbacks_for_existing_and_new_keys():
    ui = UserInterface()
    ui.register_key("L", owner="simulator", description="Toggle recording")
    registrations = []

    ui.add_registration_callback(
        lambda handle: registrations.append((handle.key, handle.owner)),
        replay_existing=True,
    )
    ui.register_key("R", owner="env", description="Reset all environments")

    assert registrations == [("L", "simulator"), ("R", "env")]


# ---------------------------------------------------------------------------
# Handle convenience methods (pressed / consume / down)
# ---------------------------------------------------------------------------


def test_user_interface_handle_convenience_methods_match_ui_methods():
    ui = UserInterface()
    handle = ui.register_key("R", owner="env", description="Reset")

    ui.begin_step()
    ui.handle_key_event("R", pressed=True)

    # All three handle methods agree with the underlying UI methods.
    assert handle.pressed() is ui.was_pressed(handle)
    assert handle.down() is ui.is_down(handle)
    assert handle.pressed() is True
    assert handle.down() is True

    assert handle.consume() is True
    # consume() flipped state; the underlying method now agrees.
    assert ui.was_pressed(handle) is False
    assert handle.pressed() is False

    # Second consume on the same press returns False.
    assert handle.consume() is False


def test_user_interface_handle_methods_after_release_raise():
    ui = UserInterface()
    handle = ui.register_key("R", owner="env", description="Reset")
    ui.unregister_key(handle)

    with pytest.raises(ValueError, match="not registered"):
        handle.pressed()
    with pytest.raises(ValueError, match="not registered"):
        handle.down()
    with pytest.raises(ValueError, match="not registered"):
        handle.consume()


# ---------------------------------------------------------------------------
# Unregister
# ---------------------------------------------------------------------------


def test_user_interface_unregister_releases_handle_and_allows_re_registration():
    ui = UserInterface()
    handle = ui.register_key("R", owner="env", description="Reset all envs")
    ui.unregister_key(handle)

    # The slot is free; a second owner can take it.
    new_handle = ui.register_key("R", owner="other", description="Other reset")
    assert new_handle.owner == "other"

    # The old handle is now invalid even though its field values match.
    with pytest.raises(ValueError, match="not registered|token mismatch"):
        ui.consume_key_press(handle)


def test_user_interface_unregister_twice_raises():
    ui = UserInterface()
    handle = ui.register_key("R", owner="env", description="Reset")
    ui.unregister_key(handle)
    with pytest.raises(ValueError, match="not registered"):
        ui.unregister_key(handle)


# ---------------------------------------------------------------------------
# Forged handle rejection
# ---------------------------------------------------------------------------


def test_user_interface_rejects_externally_constructed_handles():
    ui = UserInterface()
    real_handle = ui.register_key("R", owner="env", description="Reset")

    # An externally constructed handle with matching field values still
    # carries a different per-registration token and must be rejected.
    forged = KeyBinding(
        key=real_handle.key,
        owner=real_handle.owner,
        description=real_handle.description,
        _token=object(),
        _ui=ui,
    )

    ui.begin_step()
    ui.handle_key_event("R", pressed=True)

    with pytest.raises(ValueError, match="token mismatch|not registered"):
        ui.consume_key_press(forged)
    # The real handle still works.
    assert ui.consume_key_press(real_handle) is True


def test_user_interface_rejects_non_handle_types():
    ui = UserInterface()
    with pytest.raises(TypeError, match="KeyBinding"):
        ui.consume_key_press("R")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# KeyBindingScope
# ---------------------------------------------------------------------------


def test_owner_scope_register_and_attribute_access():
    ui = UserInterface()
    env_ui = ui.scope("env")
    env_ui.register("R", "reset", "Reset all environments")

    ui.begin_step()
    ui.handle_key_event("R", pressed=True)

    # Attribute access returns the handle; consume reads the state.
    assert isinstance(env_ui.reset, KeyBinding)
    assert env_ui.reset.pressed() is True
    assert env_ui.reset.consume() is True
    assert env_ui.reset.consume() is False  # already consumed this step


def test_owner_scope_unknown_action_raises_attribute_error_with_listing():
    ui = UserInterface()
    scope = ui.scope("env")
    scope.register("R", "reset", "Reset")

    with pytest.raises(AttributeError, match="rset.*env.*reset"):
        _ = scope.rset  # typo


def test_owner_scope_action_name_must_be_valid_identifier():
    ui = UserInterface()
    scope = ui.scope("env")

    with pytest.raises(ValueError, match="identifier"):
        scope.register("R", "reset-now", "Bad action name")
    with pytest.raises(ValueError, match="underscore"):
        scope.register("R", "_reset", "Bad action name")
    with pytest.raises(ValueError, match="keyword"):
        scope.register("R", "class", "Bad action name")


def test_owner_scope_duplicate_action_name_raises_within_scope():
    ui = UserInterface()
    scope = ui.scope("env")
    scope.register("R", "reset", "Reset all environments")

    with pytest.raises(ValueError, match="already registered in scope 'env'"):
        scope.register("Q", "reset", "Different key, same action name")


def test_owner_scope_can_reuse_action_name_across_different_scopes():
    ui = UserInterface()
    env_scope = ui.scope("env")
    target_scope = ui.scope("target")

    env_scope.register("R", "reset", "Reset all environments")
    target_scope.register("T", "reset", "Reset target position")

    # Both scopes have a 'reset' action mapped to their own keys.
    assert env_scope.reset.key == "R"
    assert target_scope.reset.key == "T"


def test_owner_scope_global_key_conflict_propagates():
    ui = UserInterface()
    env_scope = ui.scope("env")
    target_scope = ui.scope("target")
    env_scope.register("R", "reset", "Reset all environments")

    # Different scopes, same key — global uniqueness check fires.
    with pytest.raises(ValueError, match="already registered by 'env'"):
        target_scope.register("R", "reset", "Reset target position")


def test_owner_scope_unregister_all_releases_handles_and_clears_state():
    ui = UserInterface()
    scope = ui.scope("target")
    scope.register("W", "forward", "Move target forward")
    scope.register("S", "backward", "Move target backward")
    assert set(scope.actions) == {"forward", "backward"}

    scope.unregister_all()

    assert scope.actions == {}
    # Keys are free; a different scope can re-register them.
    other_scope = ui.scope("other")
    other_scope.register("W", "forward", "Other forward")
    # Calling unregister_all again is a no-op.
    scope.unregister_all()


def test_owner_scope_owner_must_be_non_empty():
    ui = UserInterface()
    with pytest.raises(ValueError, match="non-empty"):
        ui.scope("   ")


def test_owner_scope_contains_reports_registration():
    ui = UserInterface()
    scope = ui.scope("env")
    scope.register("R", "reset", "Reset")
    assert "reset" in scope
    assert "quit" not in scope


# ---------------------------------------------------------------------------
# help_text
# ---------------------------------------------------------------------------


def test_user_interface_help_text_groups_by_owner_and_sorts_keys():
    ui = UserInterface()
    ui.register_key("Q", owner="simulator", description="Quit")
    ui.register_key("M", owner="simulator", description="Toggle markers")
    ui.register_key("R", owner="env", description="Reset all environments")

    text = ui.help_text()
    lines = text.splitlines()

    # Owners sorted alphabetically: 'env' before 'simulator'.
    assert lines[0] == "[env]"
    assert "[simulator]" in lines
    env_idx = lines.index("[env]")
    sim_idx = lines.index("[simulator]")
    assert env_idx < sim_idx

    # Keys within an owner sorted alphabetically.
    sim_section = lines[sim_idx + 1 :]
    sim_keys = [line.split()[0] for line in sim_section if line.startswith("  ")]
    assert sim_keys == sorted(sim_keys)


def test_user_interface_help_text_is_empty_when_nothing_registered():
    assert UserInterface().help_text() == ""


# ---------------------------------------------------------------------------
# Backward compatibility — existing register_key + consume_key_press still work
# ---------------------------------------------------------------------------


def test_user_interface_low_level_api_still_works_for_existing_callers():
    """Existing call sites that use register_key + consume_key_press(handle)
    must continue to work unchanged after the convenience layer is added."""
    ui = UserInterface()
    handle = ui.register_key("R", owner="env", description="Reset all environments")

    ui.begin_step()
    ui.handle_key_event("R", pressed=True)

    assert ui.was_pressed(handle) is True
    assert ui.is_down(handle) is True
    assert ui.consume_key_press(handle) is True
    assert ui.was_pressed(handle) is False
