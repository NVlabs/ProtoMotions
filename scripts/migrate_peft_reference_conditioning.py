#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Give an older PEFT checkpoint's reference policy its own conditioning network.

RLFT files saved before the reference policy kept a frozen copy of its
conditioning network store the reference without ``reference_peft_model.*``
keys. That reference read the student's live conditioning network, so the
student network saved in the same file (``actor_peft_model.*``) is the one it
read when the file was saved. The loaders refuse such a file and name this
script, which copies that student network under the reference prefix and
writes the result as a new file; the input is left as it is:

    python scripts/migrate_peft_reference_conditioning.py <checkpoint> <out>

Full (``last.ckpt``) and adapter (``inference_last.ckpt``) files both work. A
file whose reference already has its copy, or that stores no reference or no
student conditioning network, needs no migration and is refused.
"""

import argparse
import sys
from pathlib import Path

import torch

from protomotions.agents.peft.utils.adapter_state import (
    REFERENCE_PEFT_MODEL_PREFIX,
    STUDENT_PEFT_MODEL_PREFIX,
    reference_lacks_conditioning_copy,
    strip_actor_prefix,
)


def migrated_model_state(model_state: dict) -> dict:
    """``model_state`` with the student conditioning network copied under the
    reference prefix (a full file's ``_actor.`` prefix is kept on the copy)."""
    if not reference_lacks_conditioning_copy(model_state):
        raise ValueError(
            "this checkpoint needs no migration: its reference policy has its own "
            "conditioning network, or the file stores no reference or no student "
            "conditioning network"
        )
    migrated = dict(model_state)
    for key, value in model_state.items():
        local_key = strip_actor_prefix(key)
        if local_key.startswith(STUDENT_PEFT_MODEL_PREFIX):
            actor_prefix = key[: len(key) - len(local_key)]
            network_key = local_key[len(STUDENT_PEFT_MODEL_PREFIX) :]
            migrated[actor_prefix + REFERENCE_PEFT_MODEL_PREFIX + network_key] = (
                value.clone()
            )
    return migrated


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "checkpoint", type=Path, help="the older full or adapter checkpoint"
    )
    parser.add_argument(
        "out", type=Path, help="where to write the migrated file (must not exist)"
    )
    args = parser.parse_args(argv)
    if args.out.exists():
        sys.exit(f"{args.out} already exists; nothing is overwritten")
    # The same loader the agent uses on this file, and the two shapes its adapter
    # loader accepts: the agent's checkpoint, or its "model" entry on its own.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    wrapped = isinstance(state, dict) and "model" in state
    try:
        migrated = migrated_model_state(state["model"] if wrapped else state)
    except ValueError as error:
        sys.exit(f"{args.checkpoint}: {error}")
    if wrapped:
        state["model"] = migrated
    else:
        state = migrated
    torch.save(state, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
