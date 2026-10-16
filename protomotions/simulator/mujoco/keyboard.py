# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task keys in the MuJoCo viewer, without a second window."""

import logging
from queue import Empty, SimpleQueue
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)


class MujocoViewerKeyboard:
    """Queue viewer-thread presses; dispatch them on the simulation thread.

    MuJoCo also runs its own shortcuts before notifying our callback. Task
    mode restores visualization settings so W/R/L and digits do not leave
    wireframe, reflections or geom groups toggled. Period switches to native
    mode for intentional viewer adjustments, without issuing task commands.
    """

    def __init__(self) -> None:
        self._events = SimpleQueue()
        self.native = False
        self._baseline = None

    def on_key(self, keycode: int) -> None:
        # The passive viewer supplies discrete presses, not key releases.
        # Never mutate physics or invoke application callbacks on its thread.
        if 33 <= keycode <= 126:
            self._events.put(chr(keycode))

    def drain_presses(self) -> Iterator[str]:
        while True:
            try:
                yield self._events.get_nowait()
            except Empty:
                return

    def toggle_native(self) -> None:
        self.native = not self.native
        log.info(
            "MuJoCo keyboard: %s",
            "native viewer shortcuts" if self.native else "task controls",
        )

    def capture(self, viewer) -> None:
        with viewer.lock():
            self._baseline = (
                viewer.opt.flags.copy(),
                viewer.opt.geomgroup.copy(),
                viewer.user_scn.flags.copy(),
            )

    def sync(self, viewer) -> None:
        if self.native:
            viewer.sync()
            return
        if self._baseline is None:
            self.capture(viewer)
        flags, groups, render_flags = self._baseline
        with viewer.lock():
            viewer.opt.flags[:] = flags
            viewer.opt.geomgroup[:] = groups
        viewer.sync()
        # sync first copies native render flags into user_scn. Only a *change*
        # to user_scn is propagated back, so setting it before that first sync
        # cannot undo a native shortcut. A second sync is needed only on change.
        with viewer.lock():
            changed = not np.array_equal(viewer.user_scn.flags, render_flags)
            if changed:
                viewer.user_scn.flags[:] = render_flags
        if changed:
            viewer.sync()
