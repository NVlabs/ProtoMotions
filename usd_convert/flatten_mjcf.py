# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
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
#
"""
Flatten a MuJoCo MJCF XML file by resolving all default-class inheritance.

Performs the following transformations:
  1. Resolves ``<default class="...">`` inheritance — inlines all inherited
     attributes onto each element that references a class, then removes the
     ``<default>`` section entirely.
  2. Converts ``<freejoint>`` elements to ``<joint type="free">``.
  3. Auto-names unnamed mesh geoms (e.g. ``mesh="X"`` gets ``name="X_visual"``
     or ``name="X_geom"``).
  4. Adds ``limited="true"`` to joints that have a ``range`` attribute
     (required by IsaacGym/PhysX; MuJoCo infers it implicitly).

After flattening, both the original and flattened files are loaded in MuJoCo
to verify the compiled models are physically identical (same bodies, joints,
geoms, tendons, actuators, etc.).  This guarantees flattening is purely
syntactic — no physics or visual properties change.

The output preserves ``<contact>``, ``<sensor>``, ``<tendon>``, and
``<material>`` sections exactly as they are.

Usage:
    # Default output: <stem>_flat.xml in the same directory
    python usd_convert/flatten_mjcf.py protomotions/data/assets/mjcf/g1_holo_compat.xml

    # Explicit output path:
    python usd_convert/flatten_mjcf.py input.xml --output flat_output.xml

    # Skip MuJoCo verification (e.g. if mujoco is not installed):
    python usd_convert/flatten_mjcf.py input.xml --no-verify
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET


def resolve_mjcf_defaults(tree: ET.ElementTree) -> int:
    """Resolve MuJoCo ``<default class="...">`` inheritance into explicit attributes.

    Handles both the **top-level (unnamed) default** that applies to all elements,
    and **named class defaults** (``<default class="X">``) with full parent
    inheritance.  Inlines resolved attributes onto each element, removes
    ``class``/``childclass`` attributes, and deletes the ``<default>`` section.

    Material references (``material="..."``) are preserved — they are semantic
    properties that MuJoCo resolves at compile time, not syntactic defaults.

    Returns the number of elements that had defaults inlined.
    """
    root = tree.getroot()
    top_default = root.find("default")
    if top_default is None:
        return 0

    # Build class_name -> {tag -> {attr: value}} with inheritance.
    # global_defaults holds the top-level (unnamed) defaults that apply to all
    # elements not covered by a named class.
    class_defaults: dict[str, dict[str, dict[str, str]]] = {}
    global_defaults: dict[str, dict[str, str]] = {}

    def collect_defaults(default_elem, parent_tag_defaults):
        class_name = default_elem.get("class")
        own_tag_defaults: dict[str, dict[str, str]] = {}
        for child in default_elem:
            if child.tag != "default":
                own_tag_defaults.setdefault(child.tag, {}).update(child.attrib)
        merged: dict[str, dict[str, str]] = {}
        for tag in set(parent_tag_defaults) | set(own_tag_defaults):
            merged[tag] = dict(parent_tag_defaults.get(tag, {}))
            if tag in own_tag_defaults:
                merged[tag].update(own_tag_defaults[tag])
        if class_name is not None:
            class_defaults[class_name] = merged
        else:
            # Top-level (unnamed) default
            global_defaults.update(merged)
        for child in default_elem:
            if child.tag == "default":
                collect_defaults(child, merged)

    collect_defaults(top_default, {})

    applicable_tags = {"geom", "joint", "site"}
    inlined_count = 0

    def apply_defaults(elem, active_class=None):
        nonlocal inlined_count
        if elem.tag == "body":
            if "childclass" in elem.attrib:
                active_class = elem.attrib.pop("childclass")
        if elem.tag in applicable_tags:
            elem_class = elem.attrib.pop("class", None) or active_class
            if elem_class and elem_class in class_defaults:
                # Named class: apply class-specific defaults (includes inherited)
                tag_defaults = class_defaults[elem_class].get(elem.tag, {})
            else:
                # No class: apply top-level (global) defaults
                tag_defaults = global_defaults.get(elem.tag, {})
            if tag_defaults:
                for attr, value in tag_defaults.items():
                    if attr not in elem.attrib:
                        elem.set(attr, value)
                inlined_count += 1
        for child in list(elem):
            apply_defaults(child, active_class)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        apply_defaults(worldbody)
    actuator = root.find("actuator")
    if actuator is not None:
        apply_defaults(actuator)

    root.remove(top_default)
    return inlined_count


def normalize_mjcf_structure(tree: ET.ElementTree) -> dict:
    """Normalize MJCF structural patterns the Isaac Sim converter doesn't handle.

    Fixes:
    1. **Unnamed mesh geoms** — adds ``name`` attributes (e.g. ``mesh="X"`` gets
       ``name="X_visual"``).
    2. **<freejoint>** to ``<joint type="free">`` conversion.
    3. **Joint limits** — adds ``limited="true"`` to joints that have a ``range``
       attribute.  IsaacGym/PhysX requires the explicit flag; MuJoCo infers it.

    Returns a dict with counts of each fix applied.
    """
    root = tree.getroot()
    fixes = {"named_geoms": 0, "freejoints": 0, "joint_limits": 0}

    # Fix 1: Auto-name unnamed mesh geoms (with uniqueness tracking)
    used_names: set[str] = {
        geom.get("name") for geom in root.iter("geom") if geom.get("name")
    }
    for body in list(root.iter("body")):
        for geom in body.findall("geom"):
            mesh_name = geom.get("mesh")
            if mesh_name and not geom.get("name"):
                is_visual = geom.get("group") == "1" or (
                    geom.get("contype") == "0" and geom.get("conaffinity") == "0"
                )
                suffix = "visual" if is_visual else "geom"
                candidate = f"{mesh_name}_{suffix}"
                if candidate in used_names:
                    idx = 2
                    while f"{candidate}_{idx}" in used_names:
                        idx += 1
                    candidate = f"{candidate}_{idx}"
                geom.set("name", candidate)
                used_names.add(candidate)
                fixes["named_geoms"] += 1

    # Fix 2: Convert <freejoint> to <joint type="free">
    for body in root.iter("body"):
        for fj in body.findall("freejoint"):
            joint = ET.SubElement(body, "joint")
            joint.set("type", "free")
            if fj.get("name"):
                joint.set("name", fj.get("name"))
            joint.set("limited", "false")
            joint.set("actuatorfrclimited", "false")
            body.remove(fj)
            fixes["freejoints"] += 1

    # Fix 3: Add limited="true" to joints with a range attribute.
    # IsaacGym (PhysX) requires the explicit limited flag to enforce joint
    # limits; MuJoCo implicitly enables limits when range is present.
    for joint in root.iter("joint"):
        if joint.get("range") is not None and joint.get("limited") is None:
            joint.set("limited", "true")
            fixes["joint_limits"] += 1

    return fixes


def verify_models_match(input_path: str, output_path: str) -> list[str]:
    """Load both MJCFs in MuJoCo and compare compiled models.

    Strips ``<contact>`` and ``<sensor>`` before loading because they may
    reference external scene elements (e.g. a floor geom).  These sections
    are not modified by flattening, so excluding them is safe.

    Returns a list of difference descriptions.  Empty list means models match.
    """
    try:
        import mujoco
        import numpy as np
    except ImportError:
        return ["mujoco not installed — skipping verification"]

    input_dir = os.path.dirname(os.path.abspath(input_path))

    def load_stripped(path: str) -> "mujoco.MjModel":
        tree = ET.parse(path)
        root = tree.getroot()
        for tag in ["contact", "sensor"]:
            for elem in root.findall(tag):
                root.remove(elem)
        # Write temp alongside original so relative mesh paths resolve
        tmp = os.path.join(input_dir, "_flatten_verify_tmp.xml")
        tree.write(tmp, xml_declaration=False, encoding="unicode")
        try:
            return mujoco.MjModel.from_xml_path(tmp)
        finally:
            os.unlink(tmp)

    orig = load_stripped(input_path)
    flat = load_stripped(output_path)

    # Collect all numpy array fields from the model
    fields = sorted(
        {
            attr
            for attr in dir(orig)
            if isinstance(getattr(orig, attr, None), np.ndarray)
        }
    )

    # Fields that are expected to differ:
    # - name_*adr: name buffer offsets shift when geom names are added
    # - names*: name hash maps and string buffers
    # - _sizes: internal allocation sizes (includes nnames, total model size)
    skip_prefixes = ("name_", "names", "_sizes")
    diffs = []
    for f in fields:
        if any(f.startswith(p) for p in skip_prefixes):
            continue
        a = getattr(orig, f)
        b = getattr(flat, f)
        if a.dtype.kind in ("U", "S"):
            continue  # string arrays — names may change
        if a.shape != b.shape:
            diffs.append(f"{f}: shape {a.shape} vs {b.shape}")
        elif not np.array_equal(a, b):
            n_diff = int(np.sum(a != b))
            if np.issubdtype(a.dtype, np.floating):
                max_err = float(np.max(np.abs(a - b)))
                diffs.append(f"{f}: {n_diff}/{a.size} diffs, max_err={max_err:.2e}")
            else:
                diffs.append(f"{f}: {n_diff}/{a.size} diffs (dtype={a.dtype})")

    return diffs


def flatten_mjcf(input_path: str, output_path: str, verify: bool = True) -> None:
    """Flatten an MJCF file: resolve defaults, normalize structure, write output."""
    tree = ET.parse(input_path)

    n_inlined = resolve_mjcf_defaults(tree)
    if n_inlined:
        print(f"  Resolved defaults: inlined attributes on {n_inlined} elements")
    else:
        print("  No <default> section found (already flat or none defined)")

    fixes = normalize_mjcf_structure(tree)
    if fixes["named_geoms"]:
        print(f"  Named {fixes['named_geoms']} unnamed mesh geoms")
    if fixes["freejoints"]:
        print(f"  Converted {fixes['freejoints']} <freejoint> to <joint type=\"free\">")
    if fixes["joint_limits"]:
        print(f"  Added limited=\"true\" to {fixes['joint_limits']} joints with range")

    tree.write(output_path, xml_declaration=False, encoding="unicode")
    print(f"\nFlattened MJCF written to: {output_path}")

    if verify:
        print("\nVerifying with MuJoCo...")
        diffs = verify_models_match(input_path, output_path)
        if not diffs:
            print("  Models match — flattening is lossless.")
        else:
            print("  WARNING: Model differences detected:", file=sys.stderr)
            for d in diffs:
                print(f"    - {d}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Flatten a MuJoCo MJCF file by resolving default-class inheritance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="Path to the input MJCF XML file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path for the flattened output XML. "
            "Defaults to <stem>_flat.xml in the same directory as the input."
        ),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip MuJoCo verification (e.g. if mujoco is not installed).",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        stem = os.path.splitext(input_path)[0]
        output_path = f"{stem}_flat.xml"

    print(f"Flattening: {input_path}")
    flatten_mjcf(input_path, output_path, verify=not args.no_verify)


if __name__ == "__main__":
    main()
