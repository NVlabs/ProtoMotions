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

"""Add visual mesh geoms that the Isaac Sim MJCF converter dropped.

The MJCF converter only includes the first visual mesh geom per body.  When a
body has multiple visual meshes (e.g. the wrist body also carries a rubber-hand
mesh), the extras are silently dropped.

This script:
  1. Parses a *preprocessed* MJCF (defaults already resolved) to catalogue every
     visual mesh geom and its parent body.
  2. Opens the generated USD and checks which visual meshes are present.
  3. For each missing mesh, loads the STL file and creates a USD Mesh prim with
     the correct transform under the parent body's Xform prim.
  4. Saves the USD.

The script requires Isaac Sim (for pxr imports via AppLauncher).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Patch a generated USD by adding visual meshes the converter dropped."
)
parser.add_argument("--mjcf", required=True, help="Path to the preprocessed MJCF XML.")
parser.add_argument("--usd", required=True, help="Path to the USD file to patch.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

import trimesh  # noqa: E402
from pxr import Gf, Usd, UsdGeom, Vt  # noqa: E402


def parse_mjcf_visual_meshes(mjcf_path):
    """Return a list of visual mesh geom dicts from the preprocessed MJCF."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", ".") if compiler is not None else "."
    mjcf_dir = os.path.dirname(os.path.abspath(mjcf_path))
    mesh_dir = os.path.normpath(os.path.join(mjcf_dir, meshdir))

    asset = root.find("asset")
    mesh_files: dict[str, str] = {}
    if asset is not None:
        for m in asset.findall("mesh"):
            name = m.get("name")
            file_attr = m.get("file")
            if name and file_attr:
                mesh_files[name] = os.path.join(mesh_dir, file_attr)

    geoms = []

    def walk(elem, body_name=None):
        if elem.tag == "body":
            body_name = elem.get("name")

        if elem.tag == "geom":
            mesh_name = elem.get("mesh")
            if mesh_name and mesh_name in mesh_files and elem.get("type") == "mesh":
                is_visual = elem.get("group") == "1" or (
                    elem.get("contype") == "0" and elem.get("conaffinity") == "0"
                )
                if is_visual and body_name:
                    pos = [float(x) for x in elem.get("pos", "0 0 0").split()]
                    quat = [float(x) for x in elem.get("quat", "1 0 0 0").split()]
                    geoms.append(
                        {
                            "mesh_name": mesh_name,
                            "stl_path": mesh_files[mesh_name],
                            "parent_body": body_name,
                            "pos": pos,
                            "quat": quat,
                        }
                    )

        for child in elem:
            walk(child, body_name)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        walk(worldbody)

    return geoms


def find_visuals_prim(stage, body_name):
    """Find the visuals Xform prim for a body in the ``/visuals/`` tree.

    The MJCF converter places visual geometry under ``/visuals/<body_name>/``.
    """
    visuals_root = stage.GetPrimAtPath("/visuals")
    if not visuals_root:
        return None
    prim = visuals_root.GetPrimAtPath(body_name)
    return prim if prim and prim.IsValid() else None


def has_mesh_geometry(parent_prim, mesh_name):
    """Check if *parent_prim* already has a USD Mesh prim for *mesh_name*.

    Walks children and grandchildren.  Only counts actual ``UsdGeom.Mesh``
    prims — ``Sphere``, ``Capsule`` etc. are not meshes.
    """
    for child in parent_prim.GetAllChildren():
        if mesh_name in child.GetName() and child.IsA(UsdGeom.Mesh):
            return True
        for grandchild in child.GetAllChildren():
            if mesh_name in grandchild.GetName() and grandchild.IsA(UsdGeom.Mesh):
                return True
    return False


def add_mesh_prim(stage, parent_prim, mesh_name, stl_path, pos, quat):
    """Load an STL and create a USD Mesh prim under parent_prim."""
    mesh = trimesh.load(stl_path, force="mesh")

    parent_path = parent_prim.GetPath()
    prim_name = f"{mesh_name}_visual"
    xform_path = parent_path.AppendChild(prim_name)
    mesh_path = xform_path.AppendChild("mesh")

    xform = UsdGeom.Xform.Define(stage, xform_path)

    # Set transform (pos + quat)
    if any(v != 0.0 for v in pos):
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
    if quat != [1.0, 0.0, 0.0, 0.0]:
        # MuJoCo quat is wxyz, USD Quatd is also (w, x, y, z)
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(quat[0], Gf.Vec3d(quat[1], quat[2], quat[3]))
        )

    usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    vertices = mesh.vertices
    points = Vt.Vec3dArray(
        [Gf.Vec3d(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    )
    usd_mesh.GetPointsAttr().Set(points)

    faces = mesh.faces
    face_counts = Vt.IntArray([3] * len(faces))
    usd_mesh.GetFaceVertexCountsAttr().Set(face_counts)

    indices = Vt.IntArray(faces.flatten().tolist())
    usd_mesh.GetFaceVertexIndicesAttr().Set(indices)

    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
        normals = Vt.Vec3fArray(
            [
                Gf.Vec3f(float(n[0]), float(n[1]), float(n[2]))
                for n in mesh.vertex_normals
            ]
        )
        usd_mesh.GetNormalsAttr().Set(normals)
        usd_mesh.SetNormalsInterpolation("vertex")

    # Set the purpose to render (visual only)
    usd_mesh.GetPurposeAttr().Set(UsdGeom.Tokens.default_)

    print(
        f"  Added mesh '{prim_name}' under {parent_path} ({len(vertices)} verts)",
        flush=True,
    )
    return xform


def main():
    mjcf_geoms = parse_mjcf_visual_meshes(args.mjcf)
    print(f"Found {len(mjcf_geoms)} visual mesh geoms in MJCF", flush=True)

    stage = Usd.Stage.Open(args.usd)

    added = 0
    skipped = 0
    for geom_info in mjcf_geoms:
        mesh_name = geom_info["mesh_name"]
        parent_body = geom_info["parent_body"]

        # Look for the parent body in the /visuals/ tree
        visuals_prim = find_visuals_prim(stage, parent_body)
        if visuals_prim is None:
            print(
                f"  WARNING: '{parent_body}' not found in /visuals/, "
                f"skipping {mesh_name}",
                flush=True,
            )
            continue

        if has_mesh_geometry(visuals_prim, mesh_name):
            skipped += 1
            continue

        stl_path = geom_info["stl_path"]
        if not os.path.isfile(stl_path):
            print(
                f"  WARNING: STL not found: {stl_path}, skipping {mesh_name}",
                flush=True,
            )
            continue

        add_mesh_prim(
            stage,
            visuals_prim,
            mesh_name,
            stl_path,
            geom_info["pos"],
            geom_info["quat"],
        )
        added += 1

    if added:
        stage.GetRootLayer().Save()
        print(
            f"\nPatched {added} missing visual meshes into {args.usd}"
            f" ({skipped} already present)",
            flush=True,
        )
    else:
        print(
            f"\nNo missing visual meshes — USD is complete ({skipped} checked).",
            flush=True,
        )

    simulation_app.close()


if __name__ == "__main__":
    main()
