# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import has_magic
from typing import List


def get_contact_sensor_body_patterns(body_name: str) -> List[str]:
    """Build Newton body-label patterns for a configured contact body.

    ProtoMotions contact bodies use short MJCF body names like
    ``left_ankle_roll_link``. Newton 1.0 stores ``model.body_label`` as full
    paths like ``g1_29dof/worldbody/.../left_ankle_roll_link`` and
    ``SensorContact`` matches against those labels via ``fnmatch``.

    Match both the original short name and any full-path label that ends in
    that body name, while preserving explicit glob patterns or full paths
    unchanged.
    """
    if "/" in body_name or has_magic(body_name):
        return [body_name]

    return [body_name, f"*/{body_name}"]


def validate_contact_sensor_match(
    body_name: str, sensor_body_patterns: List[str], matched_body_count: int
) -> None:
    """Fail fast when a configured contact body matches no Newton labels."""
    if matched_body_count == 0:
        raise ValueError(
            "Newton contact sensor for "
            f"'{body_name}' matched no body labels using "
            f"{sensor_body_patterns}. "
            "This usually means the configured ProtoMotions body name does not "
            "match Newton's body_label format."
        )
