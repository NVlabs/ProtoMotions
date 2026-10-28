# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-safe helpers shared by the train and inference entry shims.

Nothing here imports torch or a simulator backend, so a shim can call
:func:`simulator_from_argv` to pick the backend *before* torch is imported.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Callable, Sequence


def simulator_from_argv(argv: Sequence[str]) -> str:
    """Pre-parse ``--simulator`` so the backend is imported before torch."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--simulator", required=True)
    args, _ = parser.parse_known_args(list(argv))
    return args.simulator


def run_reporting_failures(main: Callable[[], None]) -> None:
    """Run ``main()``, forcing a nonzero exit on failure.

    Isaac Sim and IsaacLab each register an atexit hook (isaacsim
    simulation_app.py, isaaclab app_launcher.py) that calls app.close() with the
    default exit_code=0. Under Kit's fastShutdown that reaches
    quickReleaseFrameworkAndTerminate, documented as calling _exit() on Linux --
    so it replaces CPython's pending status 1 with 0 during interpreter
    finalization. NVIDIA documents this: close(exit_code=...) exists precisely to
    "preserve nonzero script/test failure status when fast shutdown is enabled".

    The effect is that a crashed training run reports SUCCESS. Observed on a real
    4-rank job where every rank raised torch.OutOfMemoryError and the retry loop
    still logged "attempt 3 exited 0 -- done" and the cluster marked the job
    Completed. Every retry loop, give-up sentinel and job state keyed on this exit
    code was reading a value that could not report failure.

    os._exit skips atexit entirely, so we set the status before Kit can overwrite
    it. Only the failure path does this: on success we return normally and let the
    hooks run, which is the clean shutdown they are there for.
    """
    try:
        main()
    except SystemExit as exit_request:
        # Preserve an intentional exit code (argparse inside main(), for example)
        # rather than turning a deliberate exit 0 into a failure.
        code = exit_request.code
        if code is None:
            code = 0
        elif not isinstance(code, int):
            code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
