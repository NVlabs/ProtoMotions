# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize mass and inertia of massless robot links before backend import."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

from dm_control import mjcf
import mujoco


MARKER_MASS = 0.001  # kg
MARKER_INERTIA = 4e-8  # kg m^2: a one-gram, one-centimetre-radius sphere


def _read_mjcf(path):
    """Expand includes and make external asset references location independent."""
    path = Path(path).resolve()
    root = ET.parse(path).getroot()

    def expand(parent, ancestors):
        for child in list(parent):
            if child.tag != "include":
                expand(child, ancestors)
                continue
            included_path = (path.parent / child.attrib["file"]).resolve()
            if included_path in ancestors:
                raise ValueError(f"Recursive MJCF include: {included_path}")
            included = ET.parse(included_path).getroot()
            expand(included, ancestors | {included_path})
            index = list(parent).index(child)
            parent.remove(child)
            for offset, element in enumerate(included):
                parent.insert(index + offset, element)

    expand(root, {path})
    compiler = root.find("compiler")
    options = compiler.attrib if compiler is not None else {}
    for element in root.findall("asset/*"):
        for attribute in (
            "file",
            "fileleft",
            "fileright",
            "filefront",
            "fileback",
            "fileup",
            "filedown",
        ):
            filename = element.get(attribute)
            if not filename:
                continue
            directory_key = {"mesh": "meshdir", "texture": "texturedir"}.get(
                element.tag, "assetdir"
            )
            directory = options.get(directory_key, options.get("assetdir", ""))
            if options.get("strippath") == "true":
                filename = Path(filename).name
            element.set(attribute, str((path.parent / directory / filename).resolve()))
    if compiler is not None:
        for attribute in ("meshdir", "texturedir", "assetdir", "strippath"):
            compiler.attrib.pop(attribute, None)
    return root


def prepare_robot_mjcf(mjcf_path, output_dir):
    """Return an importable MJCF with one-gram inertias for zero-mass bodies.

    The source is never edited. MuJoCo resolves defaults, geometry-derived mass
    and includes, rather than guessing from missing XML attributes. Only robot
    inertia is inspected; references to runtime sensors/ground are excluded from
    the probe, not from the resulting asset. Assets with no massless links use
    their original path. The caller owns the temporary output directory.
    """
    root = _read_mjcf(mjcf_path)
    probe = deepcopy(root)
    for tag in ("sensor", "contact", "actuator", "tendon", "equality", "keyframe"):
        for element in probe.findall(tag):
            probe.remove(element)
    compiler = probe.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(probe, "compiler")
    compiler.set("fusestatic", "false")
    source = mjcf.from_xml_string(ET.tostring(probe, encoding="unicode"))
    model = mujoco.MjModel.from_xml_string(source.to_xml_string(), source.get_assets())
    massless = {i for i in range(1, model.nbody) if model.body_mass[i] == 0}
    if not massless:
        return str(mjcf_path)

    compiler = root.find("compiler")
    infer_from_geometry = (
        compiler is not None and compiler.get("inertiafromgeom") == "true"
    )
    # With static fusion disabled, XML preorder is MuJoCo body order, including
    # unnamed bodies. Body names and ordering are never rewritten.
    for body_id, body in enumerate(root.findall(".//worldbody//body"), start=1):
        if body_id not in massless and not infer_from_geometry:
            continue
        properties = model.body(body_id)
        inertial = body.find("inertial")
        if inertial is not None:
            body.remove(inertial)
        if body_id in massless:
            attributes = dict(
                pos="0 0 0",
                mass=str(MARKER_MASS),
                diaginertia=" ".join([str(MARKER_INERTIA)] * 3),
            )
        else:
            # Preserve geometry-derived physical properties when switching the
            # compiler from forced inference to explicit/automatic inertia.
            attributes = dict(
                pos=" ".join(map(str, properties.ipos)),
                quat=" ".join(map(str, properties.iquat)),
                mass=str(properties.mass[0]),
                diaginertia=" ".join(map(str, properties.inertia)),
            )
        body.insert(0, ET.Element("inertial", attributes))
    if infer_from_geometry:
        compiler.set("inertiafromgeom", "auto")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(mjcf_path).name
    output_path.write_text(ET.tostring(root, encoding="unicode"))
    return str(output_path)
