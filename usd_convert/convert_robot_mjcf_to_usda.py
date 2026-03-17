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
End-to-end MJCF-to-USDA converter for ProtoMotions robot assets.

Wraps the low-level Isaac Lab MJCF importer (convert_mjcf_to_usd.py) with
automatic preprocessing and postprocessing so that the generated USDA works
out-of-the-box with the ProtoMotions IsaacLab simulator backend.

**Important:** The input MJCF must be *flattened* (no ``<default>`` section,
no ``class=``/``childclass=`` attributes, no ``<freejoint>`` elements, all
mesh geoms named).  Use ``flatten_mjcf.py`` to produce a flat MJCF first.

The full pipeline is:
  1. Verify the input MJCF is flat (no unresolved defaults or structural issues).
  2. Strip MuJoCo-specific elements that break the Isaac Sim MJCF importer
     (<contact>, <sensor>, <tendon>) into a temporary cleaned XML.
  3. Invoke the Isaac Lab converter (convert_mjcf_to_usd.py) as a subprocess
     with --make-instanceable --headless and the required kit extension flag.
  4. Patch missing visual meshes into the base USD.
  5. Patch the generated .usda text wrapper:
       a. Remove the "_cleaned" suffix leaked from the temp filename.
       b. Add ``over "worldBody" (active = false)`` to deactivate the extra
          articulation root that the MJCF importer creates from <worldbody>.
  6. Delete the temporary cleaned XML.

Prerequisites:
  - Isaac Lab environment must be active (``source env_isaaclab/bin/activate``).
  - Run from the repository root.

Usage:
    source env_isaaclab/bin/activate

    # First, flatten your MJCF:
    python usd_convert/flatten_mjcf.py \\
        protomotions/data/assets/mjcf/g1_holo_compat.xml

    # Then convert the flattened version:
    python usd_convert/convert_robot_mjcf_to_usda.py \\
        protomotions/data/assets/mjcf/g1_holo_compat_flat.xml

    # Explicit output directory:
    python usd_convert/convert_robot_mjcf_to_usda.py \\
        protomotions/data/assets/mjcf/g1_holo_compat_flat.xml \\
        --output-dir protomotions/data/assets/usd/g1_holo_compat

Limitations:
  - **Tendons are silently dropped.** PhysX has no equivalent to MuJoCo fixed
    tendons (coupled ankle joints, etc.). The USDA will not encode them;
    IsaacLab gets ankle behavior from the robot config PD parameters instead.
  - **Contact pairs with custom solref/friction are dropped.** PhysX uses
    material-based friction, not per-pair. Friction must be set via
    IsaacLab/PhysX material properties at runtime.
  - **Joint solimplimit is dropped.** MuJoCo-specific solver parameter with no
    PhysX equivalent.
  - **Actuator ctrlrange is dropped.** IsaacLab overrides actuator properties
    from the robot config (e.g. g1.py) at runtime.
  - **Sensor elements are dropped.** The Isaac Sim MJCF importer may not handle
    <sensor> sections that reference sites; they are stripped to avoid import
    errors. Sensors are configured at the IsaacLab level instead.
  - **Marker bodies** (e.g. head, left_rubber_hand) with zero-density geoms will
    produce PhysX warnings about invalid inertia tensors. These are cosmetic
    and can be ignored.
  - **worldBody articulation** — The MJCF importer creates an articulation root
    for <worldbody> in addition to the robot root body. This script patches the
    USDA to deactivate it, but if you manually edit the USDA later, ensure the
    ``over "worldBody" (active = false)`` block is preserved.
  - **Only tested with free-floating robots** (freejoint on root body). Fixed-base
    robots may need --fix-base passed to the underlying converter.
"""

import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


CONVERTER_SCRIPT = os.path.join(os.path.dirname(__file__), "convert_mjcf_to_usd.py")
PATCH_SCRIPT = os.path.join(os.path.dirname(__file__), "patch_usd_visual_meshes.py")

ELEMENTS_TO_STRIP = ["contact", "sensor", "tendon"]


def verify_mjcf_is_flat(input_path: str) -> list[str]:
    """Check that an MJCF file has been flattened (no unresolved defaults).

    Returns a list of issue descriptions.  An empty list means the file is flat.
    """
    tree = ET.parse(input_path)
    root = tree.getroot()
    issues = []

    # Check 1: No <default> section
    if root.find("default") is not None:
        issues.append("Contains a <default> section (unresolved class inheritance)")

    # Check 2: No class= or childclass= attributes on any element
    for elem in root.iter():
        if "class" in elem.attrib and elem.tag != "default":
            issues.append(
                f'Element <{elem.tag}> has class="{elem.get("class")}" '
                f'(body: {elem.get("name", "unnamed")})'
            )
        if "childclass" in elem.attrib:
            issues.append(
                f'Element <{elem.tag}> has childclass="{elem.get("childclass")}" '
                f'(body: {elem.get("name", "unnamed")})'
            )

    # Check 3: No <freejoint> elements
    for fj in root.iter("freejoint"):
        parent = None
        for body in root.iter("body"):
            if fj in list(body):
                parent = body.get("name", "unnamed")
                break
        issues.append(
            f'Contains <freejoint> in body "{parent}" '
            '(should be <joint type="free">)'
        )

    # Check 4: No unnamed mesh geoms
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            if geom.get("mesh") and not geom.get("name"):
                issues.append(
                    f'Unnamed mesh geom (mesh="{geom.get("mesh")}") '
                    f'in body "{body.get("name", "unnamed")}"'
                )

    return issues


def inline_materials(tree: ET.ElementTree) -> int:
    """Replace ``material="..."`` references on geoms with inline ``rgba`` values.

    The Isaac Sim MJCF importer does not reliably handle MuJoCo ``<material>``
    definitions from ``<asset>``.  This inlines the material rgba directly onto
    each referencing geom so the converter sees the correct colors.

    Returns the number of geoms that had materials inlined.
    """
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        return 0

    materials: dict[str, str] = {}
    for mat in asset.findall("material"):
        name = mat.get("name")
        rgba = mat.get("rgba")
        if name and rgba:
            materials[name] = rgba

    if not materials:
        return 0

    count = 0
    for geom in root.iter("geom"):
        mat_name = geom.get("material")
        if mat_name and mat_name in materials:
            del geom.attrib["material"]
            if "rgba" not in geom.attrib:
                geom.set("rgba", materials[mat_name])
            count += 1
    return count


def strip_mjcf(input_path: str, output_path: str) -> list:
    """Strip converter-incompatible elements and inline materials.

    Removes ``<contact>``, ``<sensor>``, ``<tendon>`` and inlines
    ``material="..."`` references as ``rgba`` values.

    Returns list of element tags that were actually removed.
    """
    tree = ET.parse(input_path)
    root = tree.getroot()

    removed = []
    for tag in ELEMENTS_TO_STRIP:
        elements = root.findall(tag)
        for elem in elements:
            root.remove(elem)
            removed.append(tag)

    n_inlined = inline_materials(tree)
    if n_inlined:
        print(f"Inlined material references on {n_inlined} geoms")

    tree.write(output_path, xml_declaration=False, encoding="unicode")
    return removed


def patch_usda(usda_path: str, desired_prim_name: str) -> None:
    """Post-process the generated USDA to fix known issues.

    1. Replace any ``_cleaned`` suffix in prim names with the desired name.
    2. Insert ``over "worldBody" (active = false)`` if not already present.
    """
    with open(usda_path) as f:
        content = f.read()

    cleaned_name = desired_prim_name + "_cleaned"

    if cleaned_name in content:
        content = content.replace(cleaned_name, desired_prim_name)

    worldbody_override = (
        '    over "worldBody" (\n'
        "        active = false\n"
        "    )\n"
        "    {\n"
        "    }\n"
    )

    if 'over "worldBody"' not in content:
        pattern = r'(\)\n\{\n)(    variantSet "Physics")'
        replacement = r"\1" + worldbody_override + r"    \2"
        content, n = re.subn(pattern, replacement, content, count=1)
        if n == 0:
            print(
                f"WARNING: Could not inject worldBody override into {usda_path}. "
                "You may need to add it manually."
            )

    with open(usda_path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a ProtoMotions robot MJCF to USDA for IsaacLab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="Path to the input MJCF XML file (must be flattened first).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for the generated USDA and configuration/ files. "
            "Defaults to protomotions/data/assets/usd/<stem>/ where <stem> "
            "is the MJCF filename without extension."
        ),
    )
    parser.add_argument(
        "--fix-base",
        action="store_true",
        help="Pass --fix-base to the Isaac Lab converter (for fixed-base robots).",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    stem = os.path.splitext(os.path.basename(input_path))[0]

    # Step 1: Verify the MJCF is flat
    issues = verify_mjcf_is_flat(input_path)
    if issues:
        print("ERROR: Input MJCF is not flat. Issues found:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        print(
            f"\nFlatten it first:\n"
            f"  python usd_convert/flatten_mjcf.py {args.input}",
            file=sys.stderr,
        )
        sys.exit(1)
    print("MJCF verified: flat (no unresolved defaults or structural issues)")

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        assets_root = os.path.join(
            os.path.dirname(__file__),
            "..",
            "protomotions",
            "data",
            "assets",
            "usd",
            stem,
        )
        output_dir = os.path.abspath(assets_root)

    usda_name = f"{stem}.usda"
    usda_path = os.path.join(output_dir, usda_name)
    os.makedirs(output_dir, exist_ok=True)

    # Step 2: Strip <contact>, <sensor>, <tendon> into a temp file
    cleaned_path = os.path.join(os.path.dirname(input_path), f"{stem}_cleaned.xml")
    removed = strip_mjcf(input_path, cleaned_path)
    if removed:
        print(f"Stripped from MJCF: {', '.join(removed)}")
    else:
        print("No elements needed stripping.")
    converter_input = cleaned_path

    # Step 3: Run the Isaac Lab converter
    print(f"\nInput MJCF:  {converter_input}")
    print(f"Output USDA: {usda_path}\n")

    converter_cmd = [
        sys.executable,
        CONVERTER_SCRIPT,
        converter_input,
        usda_path,
        "--make-instanceable",
        "--headless",
        "--kit_args",
        "--enable isaacsim.asset.importer.mjcf",
    ]
    if args.fix_base:
        converter_cmd.append("--fix-base")

    print(f"Running converter: {' '.join(converter_cmd)}\n")
    result = subprocess.run(converter_cmd)

    if result.returncode != 0:
        print(
            f"\nERROR: Converter exited with code {result.returncode}.",
            file=sys.stderr,
        )
        if os.path.isfile(cleaned_path):
            os.remove(cleaned_path)
            print(f"Cleaned up temp file: {cleaned_path}")
        sys.exit(result.returncode)

    # Step 4: Patch missing visual meshes into the base USD
    base_usd_path = os.path.join(output_dir, "configuration", f"{stem}_base.usd")
    if os.path.isfile(base_usd_path) and os.path.isfile(PATCH_SCRIPT):
        print("\nPatching missing visual meshes...")
        patch_cmd = [
            sys.executable,
            PATCH_SCRIPT,
            "--mjcf",
            cleaned_path,
            "--usd",
            base_usd_path,
            "--headless",
        ]
        patch_result = subprocess.run(patch_cmd)
        if patch_result.returncode != 0:
            print(
                "WARNING: Visual mesh patching failed. Some meshes may be missing.",
                file=sys.stderr,
            )

    # Step 5: Patch the USDA
    if os.path.isfile(usda_path):
        patch_usda(usda_path, stem)
        print("\nPatched USDA: removed _cleaned suffix, added worldBody override.")
    else:
        print(f"\nWARNING: Expected USDA not found at {usda_path}", file=sys.stderr)

    # Step 6: Clean up temp file
    if os.path.isfile(cleaned_path):
        os.remove(cleaned_path)
        print(f"Cleaned up temp file: {cleaned_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Done! Generated files:")
    print(f"  {usda_path}")
    config_dir = os.path.join(output_dir, "configuration")
    if os.path.isdir(config_dir):
        for f in sorted(os.listdir(config_dir)):
            print(f"  {os.path.join(config_dir, f)}")
    print()
    rel_path = os.path.relpath(usda_path, os.path.join(output_dir, "..", ".."))
    print("Use in robot config or --overrides:")
    print(f"  robot.asset.usd_asset_file_name={rel_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
