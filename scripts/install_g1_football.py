# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify and extract compiled G1 Football, without fetching policy weights."""

import argparse
import hashlib
import io
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

BUNDLE = Path(__file__).resolve().parents[1] / "demos/g1-football"
MODEL_LICENSE = (
    Path(__file__).resolve().parents[1] / "legal/NVIDIA-Open-Model-License.pdf"
)
WEIGHTS_URL = (
    "https://huggingface.co/nvidia/Metatron-v1/resolve/"
    "167355ae5630ebaece752545516da5217bf15d04/demo/weights.bin"
)
WEIGHTS_SHA256 = "b516946707127a66ee41c4dbc740ab3eb19504de3a8c93ebec16238e65ed9e05"
WEIGHTS_BYTES = 402946352
PERCEPTION = "perception/keeper-perception-cp313"
RUNTIME = {
    "pyodideVersion": "0.28.3",
    "pythonVersion": "3.13.2",
    "cacheTag": "cpython-313",
    "bytecodeMagicHex": "f30d0d0a",
}
REQUIRED = {
    "index.html",
    "isolation-worker.js",
    "world.xml",
    "world.json",
    "keeper-prediction.xml",
    "policy/manifest.json",
    "policy/model.onnx",
    "policy/network.onnx",
    "policy/observations.onnx",
    "policy/MODEL_CARD.md",
    "policy/NOTICE.txt",
    "policy/LICENSE.pdf",
    f"{PERCEPTION}.zip",
    f"{PERCEPTION}.manifest.json",
    "pyodide/pyodide.mjs",
    "pyodide/pyodide.asm.js",
    "pyodide/pyodide.asm.wasm",
    "pyodide/python_stdlib.zip",
    "pyodide/pyodide-lock.json",
    "NOTICE.txt",
    "licenses/Apache-2.0.txt",
    "licenses/robot-notices.txt",
    "licenses/pyodide-LICENSE.txt",
    "licenses/third-party/manifest.json",
}
ASSETS = {
    ".html",
    ".js",
    ".mjs",
    ".css",
    ".wasm",
    ".json",
    ".xml",
    ".stl",
    ".glb",
    ".png",
    ".jpg",
    ".webp",
    ".svg",
    ".ico",
}
MAX_BUNDLE_BYTES = 250_000_000
MAX_RUNTIME_BYTES = 200_000_000
MAX_PERCEPTION_BYTES = 10_000_000
MAX_GRAPH_BYTES = 5_000_000


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(payload):
    return json.loads(payload, object_pairs_hook=_unique_object)


def metadata(directory=BUNDLE):
    result = read_json((directory / "bundle.json").read_bytes())
    if result.get("release_status") not in {"integration-fixture", "public-release"}:
        raise ValueError("Unknown release status")
    return result


def safe_path(name):
    path = PurePosixPath(name)
    if (
        not name
        or not path.parts
        or path.is_absolute()
        or str(path) != name
        or "\\" in name
        or ":" in name
        or any(ord(c) < 32 for c in name)
        or any(p.startswith(".") or p.endswith((".", " ")) for p in path.parts)
    ):
        raise ValueError(f"Unsafe asset path: {name}")
    return path


def members(archive, limit, *, directories=False):
    """Bound expansion before reading payloads, including nested ZIPs/wheels."""
    entries = archive.infolist()
    names = set()
    total = 0
    if len(entries) > 20_000:
        raise ValueError("Archive member count exceeds limit")
    for entry in entries:
        name = entry.filename
        safe_path(name[:-1] if directories and entry.is_dir() else name)
        mode = stat.S_IFMT(entry.external_attr >> 16)
        if (
            name in names
            or entry.orig_filename != name
            or (entry.is_dir() and not directories)
            or mode not in {0, stat.S_IFREG, stat.S_IFDIR}
            or (mode == stat.S_IFDIR and not entry.is_dir())
            or entry.flag_bits & 1
            or entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
        ):
            raise ValueError(f"Unsafe or duplicate archive member: {name}")
        names.add(name)
        total += entry.file_size
        if (
            entry.file_size > 80_000_000
            or total > limit
            or entry.file_size > max(1_000_000, entry.compress_size * 1000)
        ):
            raise ValueError("Archive expansion exceeds size limit")
    return entries


def check_payload(payload, spec, label):
    if (
        type(spec.get("bytes")) is not int
        or spec["bytes"] != len(payload)
        or spec.get("sha256") != sha256(payload)
    ):
        raise ValueError(f"Size or checksum mismatch: {label}")


def check_policy(archive, info):
    manifest = read_json(archive.read("policy/manifest.json"))
    if manifest.get("release_status") != info["release_status"]:
        raise ValueError("Policy and bundle release statuses disagree")
    source = manifest.get("single_file_artifact_sha256", "")
    if (
        not re.fullmatch(r"[a-f0-9]{64}", source)
        or info.get("policy_source_sha256") != source
    ):
        raise ValueError("Bundle policy_source_sha256 disagrees with policy manifest")
    remote = manifest.get("external_data", {})
    download = remote.get("download", {})
    if (
        remote.get("path") != "weights.bin"
        or remote.get("sha256") != WEIGHTS_SHA256
        or remote.get("bytes") != WEIGHTS_BYTES
        or download.get("url") != WEIGHTS_URL
        or download.get("sha256") != WEIGHTS_SHA256
    ):
        raise ValueError("Invalid pinned policy download contract")
    graphs = {
        "model.onnx": {
            "bytes": manifest.get("artifact_bytes"),
            "sha256": manifest.get("artifact_sha256"),
        },
        "network.onnx": manifest.get("network_graph", {}),
        "observations.onnx": manifest.get("observations_graph", {}),
    }
    for name, spec in graphs.items():
        if name != "model.onnx" and spec.get("path") != name:
            raise ValueError("Unexpected policy graph path")
        payload = archive.read("policy/" + name)
        if not 0 < len(payload) <= MAX_GRAPH_BYTES:
            raise ValueError(
                "Policy graph exceeds size limit; do not bundle full models"
            )
        check_payload(payload, spec, name)
    if graphs["network.onnx"].get("source_sha256") != source:
        raise ValueError("Network graph source checksum disagrees")


def check_perception(archive):
    manifest = read_json(archive.read(f"{PERCEPTION}.manifest.json"))
    if (
        manifest.get("schema") != 1
        or manifest.get("format") != "sourceless-cpython-bytecode-zip"
        or manifest.get("runtime") != RUNTIME
        or manifest.get("entrypoint") != "keeper_perception"
        or manifest.get("extractDir") != "/football"
        or manifest.get("compile", {}).get("invalidationMode") != "UNCHECKED_HASH"
    ):
        raise ValueError("Unsupported perception format or Python/Pyodide ABI")
    payload = archive.read(f"{PERCEPTION}.zip")
    spec = manifest.get("archive", {})
    if spec.get("file") != "keeper-perception-cp313.zip":
        raise ValueError("Unexpected perception archive path")
    check_payload(payload, spec, "perception archive")
    inventory = manifest["inventory"]
    canonical = json.dumps(
        inventory, separators=(",", ":"), ensure_ascii=False
    ).encode()
    if sha256(canonical) != manifest.get("inventorySha256"):
        raise ValueError("Perception inventory checksum mismatch")
    indexed = {item["path"]: item for item in inventory}
    if len(indexed) != len(inventory) or "keeper_perception.pyc" not in indexed:
        raise ValueError("Invalid perception inventory")
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        entries = members(inner, MAX_PERCEPTION_BYTES)
        if {e.filename for e in entries} != set(indexed) | {"manifest.json"}:
            raise ValueError("Perception archive inventory mismatch")
        if read_json(inner.read("manifest.json")) != {
            key: value for key, value in manifest.items() if key != "archive"
        }:
            raise ValueError("Perception inner and outer manifests disagree")
        notice_paths = set()
        for name, item in indexed.items():
            suffix = safe_path(name).suffix
            data = inner.read(name)
            check_payload(data, item, name)
            if suffix == ".pyc" and item.get("kind") == "bytecode":
                if (
                    len(data) < 16
                    or data[:4].hex() != RUNTIME["bytecodeMagicHex"]
                    or data[4:8] != b"\x01\x00\x00\x00"
                ):
                    raise ValueError(f"Invalid CPython bytecode ABI/header: {name}")
            elif suffix == ".json" and item.get("kind") == "model-data":
                read_json(data)
            elif (
                name.startswith("notices/")
                and item.get("kind") == "notice"
                and (
                    suffix in {".txt", ".md", ".pdf"}
                    or PurePosixPath(name).name in {"LICENSE", "NOTICE"}
                )
            ):
                notice_paths.add(name)
            else:
                raise ValueError(f"Non-compiled perception member: {name}")
        if set(manifest.get("notices", [])) != notice_paths:
            raise ValueError("Perception notice inventory mismatch")
    return manifest["dependencies"]


def check_dependencies(archive, dependencies):
    lock_data = archive.read("pyodide/pyodide-lock.json")
    if sha256(lock_data) != dependencies.get("pyodideLockSha256"):
        raise ValueError("Pyodide lock checksum mismatch")
    lock = read_json(lock_data)
    if (
        lock.get("info", {}).get("python") != RUNTIME["pythonVersion"]
        or lock["info"].get("abi_version") != "2025_0"
        or lock["info"].get("arch") != "wasm32"
    ):
        raise ValueError("Pyodide dependency Python ABI mismatch")
    packages = dependencies["packages"]
    if not {"numpy", "opencv-python"} <= set(packages):
        raise ValueError("Missing perception dependencies")
    expected = {"pyodide/python_stdlib.zip"}
    for name, spec in packages.items():
        record = lock["packages"][name]
        filename = spec["file"]
        if (
            str(safe_path(filename)) != PurePosixPath(filename).name
            or not filename.endswith(".whl")
            or spec["version"] != record["version"]
            or filename != record["file_name"]
            or spec["sha256"] != record["sha256"]
        ):
            raise ValueError("Pyodide package contract mismatch")
        path = "pyodide/" + filename
        if sha256(archive.read(path)) != spec["sha256"]:
            raise ValueError(f"Pyodide dependency checksum mismatch: {name}")
        expected.add(path)
    actual = {
        n
        for n in archive.namelist()
        if n.endswith((".zip", ".whl")) and n != f"{PERCEPTION}.zip"
    }
    if actual != expected:
        raise ValueError("Unexpected nested runtime archive")
    total = 0
    for name in sorted(expected):
        with zipfile.ZipFile(io.BytesIO(archive.read(name))) as inner:
            entries = members(inner, MAX_RUNTIME_BYTES, directories=True)
            total += sum(e.file_size for e in entries)
            if total > MAX_RUNTIME_BYTES:
                raise ValueError("Nested runtime expansion exceeds size limit")
            for entry in entries:
                if entry.filename.endswith((".zip", ".whl")):
                    raise ValueError(
                        "Unexpected nested archive inside runtime dependency"
                    )
            if inner.testzip():
                raise ValueError("Corrupt runtime dependency")


def check_notices(archive):
    if archive.read("policy/LICENSE.pdf") != MODEL_LICENSE.read_bytes():
        raise ValueError("Model license differs from the repository license")
    if (
        b"Licensed by NVIDIA Corporation under the NVIDIA Open Model License"
        not in archive.read("policy/NOTICE.txt")
    ):
        raise ValueError("Missing NVIDIA model attribution")
    notices = read_json(archive.read("licenses/third-party/manifest.json"))["files"]
    prefix = "licenses/third-party/"
    actual = {
        n[len(prefix) :]
        for n in archive.namelist()
        if n.startswith(prefix) and n != prefix + "manifest.json"
    }
    if actual != set(notices):
        raise ValueError("Third-party notice inventory mismatch")
    for name, expected in notices.items():
        safe_path(name)
        if sha256(archive.read(prefix + name)) != expected:
            raise ValueError(f"Notice checksum mismatch: {name}")


def verify(directory=BUNDLE, *, allow_fixture=False):
    info = metadata(directory)
    if info["release_status"] != "public-release" and not allow_fixture:
        raise ValueError(
            "Bundle is not marked public-release; pass --allow-fixture for review"
        )
    path = directory / "game.zip"
    if path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("Bundle exceeds hosting budget")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != info.get("sha256"):
        raise ValueError("Bundle checksum mismatch; fetch its Git LFS payload")
    with zipfile.ZipFile(path) as archive:
        entries = members(archive, MAX_BUNDLE_BYTES)
        names = {e.filename for e in entries}
        if (
            names != set(info["files"])
            or not REQUIRED <= names
            or not any(n.startswith("assets/") and n.endswith(".js") for n in names)
            or not any(
                n.startswith("assets/") and n.endswith((".stl", ".glb")) for n in names
            )
        ):
            raise ValueError("Bundle inventory or required assets mismatch")
        for entry in entries:
            name = entry.filename
            asset = safe_path(name)
            notice = (
                name == "NOTICE.txt" or name.startswith(("licenses/", "policy/"))
            ) and (asset.suffix in {".txt", ".md", ".pdf"} or asset.name == "LICENSE")
            nested = name == f"{PERCEPTION}.zip" or (
                name.startswith("pyodide/") and asset.suffix in {".zip", ".whl"}
            )
            graph = name in {
                "policy/model.onnx",
                "policy/network.onnx",
                "policy/observations.onnx",
            }
            if (
                asset.name in {"package.json", "package-lock.json"}
                or any(
                    p in {"src", "scripts", "tests", "node_modules", "internal"}
                    for p in asset.parts
                )
                or not (
                    asset.suffix in ASSETS
                    or notice
                    or nested
                    or graph
                    or name == "world.mjb.lossless"
                )
            ):
                raise ValueError(f"Non-runtime file or bundled weights: {name}")
            if (
                type(info["files"][name]) is not int
                or entry.file_size != info["files"][name]
            ):
                raise ValueError(f"Incorrect asset size: {name}")
            if asset.suffix in {".js", ".mjs", ".css", ".wasm"}:
                data = archive.read(entry)
                if re.search(rb"sourceMappingURL\s*=|\.debug_info|\.debug_line", data):
                    raise ValueError(f"Source map or debug data: {name}")
                # Upstream Pyodide contains Python examples/docstrings by design.
                # This catches accidental app-module embedding, not arbitrary
                # obfuscation; it is not a general JavaScript source analyser.
                if (
                    asset.suffix in {".js", ".mjs"}
                    and not name.startswith("pyodide/")
                    and re.search(
                        rb"KEEPER_PERCEPTION_PY\s*=|(?:^|\n)\s*(?:def \w+\([^\n]*\):|from numpy |import numpy)",
                        data,
                    )
                ):
                    raise ValueError(f"Embedded application Python source: {name}")
        check_policy(archive, info)
        dependencies = check_perception(archive)
        check_dependencies(archive, dependencies)
        check_notices(archive)
        if archive.testzip():
            raise ValueError("Corrupt bundle")
    return info


def install(directory, destination, *, allow_fixture=False):
    verify(directory, allow_fixture=allow_fixture)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(directory / "game.zip") as archive:
        for entry in archive.infolist():
            target = destination / entry.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--out", type=Path, help="New static-site directory")
    parser.add_argument(
        "--allow-fixture",
        action="store_true",
        help="Allow review of an integration fixture",
    )
    args = parser.parse_args()
    if args.status:
        print(str(metadata()["release_status"] == "public-release").lower())
    elif args.check:
        verify(allow_fixture=args.allow_fixture)
        print("G1 Football bundle verified")
    else:
        install(BUNDLE, args.out, allow_fixture=args.allow_fixture)
        print(f"Installed G1 Football at {args.out}")


if __name__ == "__main__":
    main()
