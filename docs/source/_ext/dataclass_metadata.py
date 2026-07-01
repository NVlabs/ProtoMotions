# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sphinx extension to extract help text from dataclass field metadata.

This extension processes dataclass fields and adds their metadata["help"]
as documentation, similar to how docstrings work for class attributes.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sphinx.application import Sphinx


def process_docstring(app, what, name, obj, options, lines):
    """Process docstrings to add dataclass field metadata."""
    if what != "class" or not is_dataclass(obj):
        return

    # Check if there's already an Attributes section
    has_attributes = any("Attributes:" in line for line in lines)
    
    # Collect field documentation from metadata
    field_docs = []
    for f in fields(obj):
        metadata = getattr(f, "metadata", None) or {}
        help_text = metadata.get("help", "")
        if help_text:
            field_docs.append(f"    {f.name}: {help_text}")
    
    if field_docs and not has_attributes:
        # Add Attributes section if we have documented fields
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("Attributes:")
        lines.extend(field_docs)


def setup(app: Sphinx):
    """Setup the extension."""
    app.connect("autodoc-process-docstring", process_docstring)
    
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

