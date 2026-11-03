# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate joint-group-specific SOMA assets from existing geometry templates."""

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from data.scripts.from_soma_to_anatomical_soma.anatomical_soma_joint_structure import (  # noqa: E402
    reorder_soma_tree,
)


def build(source, output, *, serial):
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = reorder_soma_tree(ET.parse(source, parser=parser), serial=serial)
    if not serial:
        custom = tree.getroot().find("custom")
        if custom is None:
            custom = ET.SubElement(tree.getroot(), "custom")
        if custom.find("text[@name='coordinate_system']") is None:
            ET.SubElement(
                custom,
                "text",
                name="coordinate_system",
                data="+X-left,-Y-forward,+Z-up",
            )
        if custom.find("text[@name='semantic_body_names']") is None:
            names = [b.get("name") for b in tree.find("worldbody").iter("body")]
            ET.SubElement(
                custom, "text", name="semantic_body_names", data=" ".join(names)
            )
    if not any(
        child.tag is ET.Comment and "SPDX-FileCopyrightText" in child.text
        for child in tree.getroot()
    ):
        tree.getroot().insert(
            0,
            ET.Comment(
                " SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. "
            ),
        )
        tree.getroot().insert(1, ET.Comment(" SPDX-License-Identifier: Apache-2.0 "))
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serial", action="store_true")
    args = parser.parse_args()
    build(args.source, args.output, serial=args.serial)
