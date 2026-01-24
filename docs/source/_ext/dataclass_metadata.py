# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

