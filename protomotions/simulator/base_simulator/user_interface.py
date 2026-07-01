# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Viewer/user input registry shared by simulator backends.

The simulator owns raw key capture because key events come from the viewer
backend. Consumers own semantics by registering the exact keys they use and
receiving opaque handles back. That keeps task code from polling arbitrary
global key state and makes key conflicts fail during setup.

Two API levels:

- :meth:`UserInterface.register_key` is the low-level primitive. It returns
  a :class:`KeyBinding` handle. The handle exposes
  :meth:`KeyBinding.pressed` / :meth:`KeyBinding.consume` /
  :meth:`KeyBinding.down` so callers do not need to pass the handle
  back to the UI on every read.
- :meth:`UserInterface.scope` returns a :class:`KeyBindingScope` that lets
  a single owner declare named actions and access them by attribute, e.g.
  ``self._key_bindings.reset.consume()``. The scope owns its handles;
  teardown via :meth:`KeyBindingScope.unregister_all` releases them in
  one call.

Owner-scoping comes from API shape, not runtime caller introspection: each
scope object only contains its owner's handles, so an owner literally
cannot reach another owner's actions.
"""

from __future__ import annotations

import keyword
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class KeyBinding:
    """Handle returned to the code that registered a key.

    Callers should use the convenience methods (:meth:`pressed`,
    :meth:`consume`, :meth:`down`) rather than calling the registry
    directly. The handle is genuinely opaque: forging one externally
    (constructing a matching dataclass instance from outside) will not
    pass the registry's identity check, because ``_token`` is a unique
    object created per registration inside ``UserInterface.register_key``.
    """

    key: str
    owner: str
    description: str
    _token: object = field(repr=False, compare=True)
    _ui: "UserInterface" = field(repr=False, compare=False)

    def pressed(self) -> bool:
        """Was this key pressed during the current step (without consuming)."""
        return self._ui.was_pressed(self)

    def consume(self) -> bool:
        """Was this key pressed this step; mark it consumed so a second
        reader sees False. Use for one-shot actions like reset."""
        return self._ui.consume_key_press(self)

    def down(self) -> bool:
        """Is this key currently held down (level-triggered)."""
        return self._ui.is_down(self)


@dataclass
class _KeyState:
    handle: KeyBinding
    on_press: Optional[Callable[[], None]]
    is_down: bool = False
    pressed: bool = False
    consumed: bool = False


def _validate_action_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError(
            f"Action name must be a string, got {type(name).__name__}"
        )
    if not name.isidentifier():
        raise ValueError(
            f"Action name '{name}' is not a valid Python identifier"
        )
    if name.startswith("_"):
        raise ValueError(
            f"Action name '{name}' must not start with underscore "
            "(would shadow scope internals)"
        )
    if keyword.iskeyword(name):
        raise ValueError(f"Action name '{name}' is a Python keyword")


class KeyBindingScope:
    """Per-owner attribute-style access to registered keys.

    The scope is sugar over :class:`UserInterface`. It lets one owner
    declare named actions and access them via attribute. Action names must
    be valid non-private Python identifiers because they become attribute
    names; underscore-prefixed names would shadow the scope's own
    internals and are rejected at registration time.
    """

    def __init__(self, ui: "UserInterface", owner: str):
        self._ui = ui
        self._owner = owner
        self._actions: Dict[str, KeyBinding] = {}

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def actions(self) -> Dict[str, KeyBinding]:
        """Read-only view of action_name -> handle for inspection."""
        return dict(self._actions)

    def register(
        self,
        key: str,
        action: str,
        description: str,
        *,
        on_press: Optional[Callable[[], None]] = None,
    ) -> KeyBinding:
        """Register a key as a named action in this scope.

        Action names are scope-local: two different scopes may both have an
        action named ``"reset"``, but the underlying keys must differ
        (global key uniqueness is still enforced by :class:`UserInterface`).
        """
        _validate_action_name(action)
        if action in self._actions:
            raise ValueError(
                f"Action '{action}' is already registered in scope "
                f"'{self._owner}'"
            )
        handle = self._ui.register_key(
            key,
            owner=self._owner,
            description=description,
            on_press=on_press,
        )
        self._actions[action] = handle
        return handle

    def unregister_all(self) -> None:
        """Release every handle this scope created. Safe to call once
        per scope; calling twice is a no-op."""
        for handle in list(self._actions.values()):
            self._ui.unregister_key(handle)
        self._actions.clear()

    def __getattr__(self, action: str) -> KeyBinding:
        # __getattr__ is only invoked if normal lookup fails, so the
        # scope's own _ui / _owner / _actions resolve via __dict__ first.
        actions = self.__dict__.get("_actions")
        if actions is None or action not in actions:
            registered = sorted(actions or ())
            raise AttributeError(
                f"No action '{action}' registered in scope "
                f"'{self.__dict__.get('_owner', '?')}'. "
                f"Registered actions: {registered}"
            )
        return actions[action]

    def __contains__(self, action: str) -> bool:
        return action in self._actions


class UserInterface:
    """Registry and per-step state for viewer/user input keys."""

    def __init__(self) -> None:
        self._keys: Dict[str, _KeyState] = {}
        self._registration_callbacks: List[Callable[[KeyBinding], None]] = []
        self.active_env_id: int = 0

    @property
    def registered_keys(self) -> Dict[str, KeyBinding]:
        return {key: state.handle for key, state in self._keys.items()}

    def registered_key_names(self) -> Iterable[str]:
        return self._keys.keys()

    def register_key(
        self,
        key: str,
        *,
        owner: str,
        description: str,
        on_press: Optional[Callable[[], None]] = None,
    ) -> KeyBinding:
        normalized = self.normalize_key(key)
        owner = owner.strip()
        description = description.strip()
        if not owner:
            raise ValueError("User-interface key owner must be non-empty")
        if not description:
            raise ValueError("User-interface key description must be non-empty")
        if normalized in self._keys:
            existing = self._keys[normalized].handle
            raise ValueError(
                f"User-interface key '{normalized}' is already registered by "
                f"'{existing.owner}' for: {existing.description}. "
                f"Cannot also register it for '{owner}' ({description})."
            )

        handle = KeyBinding(
            key=normalized,
            owner=owner,
            description=description,
            _token=object(),
            _ui=self,
        )
        self._keys[normalized] = _KeyState(handle=handle, on_press=on_press)
        for callback in self._registration_callbacks:
            callback(handle)
        return handle

    def unregister_key(self, handle: KeyBinding) -> None:
        """Release a previously-registered key.

        Raises ``ValueError`` if the handle was not issued by this UI or if
        it has already been released. Callers should release exactly once
        per registration.
        """
        state = self._state_for_handle(handle)
        del self._keys[state.handle.key]

    def scope(self, owner: str) -> KeyBindingScope:
        """Return a per-owner attribute-style facade over :meth:`register_key`.

        Multiple scopes for the same owner string are allowed but each must
        register different keys (global key uniqueness still applies via
        :meth:`register_key`).
        """
        owner = owner.strip()
        if not owner:
            raise ValueError("KeyBindingScope owner must be non-empty")
        return KeyBindingScope(self, owner)

    def add_registration_callback(
        self,
        callback: Callable[[KeyBinding], None],
        *,
        replay_existing: bool = False,
    ) -> None:
        """Notify a backend adapter whenever a key is registered.

        Simulator backends that must explicitly subscribe viewer keys can use
        this to handle keys registered after simulator construction, such as
        env reset and interactive task-control bindings.
        """
        self._registration_callbacks.append(callback)
        if replay_existing:
            for state in self._keys.values():
                callback(state.handle)

    def begin_step(self) -> None:
        for state in self._keys.values():
            state.pressed = False
            state.consumed = False

    def handle_key_event(self, key: str, *, pressed: bool = True) -> bool:
        """Record a raw backend key transition.

        ``pressed`` is level state from the simulator backend. ``KeyBinding.pressed``
        is an edge signal and is raised only on a false->true transition, while
        ``KeyBinding.down`` mirrors the latest level state. Backends may therefore
        call this every frame with their current key-down value without turning a
        held key into repeated one-shot presses.
        """
        normalized = self.normalize_key(key)
        state = self._keys.get(normalized)
        if state is None:
            return False

        was_down = state.is_down
        state.is_down = pressed
        if pressed and not was_down:
            state.pressed = True
            state.consumed = False
            if state.on_press is not None:
                state.on_press()
        return True

    def was_pressed(self, handle: KeyBinding) -> bool:
        state = self._state_for_handle(handle)
        return state.pressed and not state.consumed

    def consume_key_press(self, handle: KeyBinding) -> bool:
        state = self._state_for_handle(handle)
        if state.pressed and not state.consumed:
            state.consumed = True
            return True
        return False

    def is_down(self, handle: KeyBinding) -> bool:
        return self._state_for_handle(handle).is_down

    def help_text(self) -> str:
        """Formatted listing of registered keys, grouped by owner.

        Use to surface bindings on viewer startup. Returns an empty string
        if no keys are registered.
        """
        if not self._keys:
            return ""
        by_owner: Dict[str, List[KeyBinding]] = {}
        for state in self._keys.values():
            by_owner.setdefault(state.handle.owner, []).append(state.handle)
        lines: List[str] = []
        for owner in sorted(by_owner):
            lines.append(f"[{owner}]")
            for handle in sorted(by_owner[owner], key=lambda h: h.key):
                lines.append(f"  {handle.key:>4}  {handle.description}")
        return "\n".join(lines)

    def _state_for_handle(self, handle: KeyBinding) -> _KeyState:
        if not isinstance(handle, KeyBinding):
            raise TypeError(
                f"Expected KeyBinding, got {type(handle).__name__}"
            )
        normalized = self.normalize_key(handle.key)
        state = self._keys.get(normalized)
        if state is None:
            raise ValueError(
                f"Key handle '{handle.key}' is not registered with this "
                "user interface"
            )
        # Identity check on the per-registration token rejects forged
        # handles (constructed externally with matching field values) and
        # stale handles (held across an unregister + re-register cycle).
        if state.handle._token is not handle._token:
            raise ValueError(
                f"Key handle '{handle.key}' was not issued by this user "
                "interface (token mismatch — handle was forged or released)"
            )
        return state

    @staticmethod
    def normalize_key(key: str) -> str:
        if not isinstance(key, str) or len(key) == 0:
            raise ValueError("User-interface keys must be non-empty strings")
        key = key.strip()
        if len(key) == 0:
            raise ValueError("User-interface keys must be non-empty strings")
        if len(key) == 1:
            return key.upper()
        return key
