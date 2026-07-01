# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers for command-line argument parsing."""

import argparse


def parse_bool(value):
    """Parse flexible CLI boolean values for argparse ``type=`` hooks."""
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")
