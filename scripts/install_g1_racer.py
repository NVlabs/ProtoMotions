# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify and extract the compiled G1 Racer bundle; no application build needed."""

import argparse
import hashlib
import io
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

BUNDLE = Path(__file__).resolve().parents[1] / "demos/g1-racer"
REQUIRED = {
    "index.html",
    "policy/network.json",
    "observations.onnx",
    "observations.wasm",
    "observations.json",
    "manifest.json",
    "world.xml",
    "world.json",
    "gravel.bin",
    "game-runtime.zip",
    "g1_head_textured.glb",
    "parity.json",
    "NOTICE.txt",
    "licenses/Apache-2.0.txt",
    "licenses/robot-notices.txt",
    "licenses/third-party/manifest.json",
    "licenses/pyodide-LICENSE.txt",
    "policy/MODEL_CARD.md",
    "policy/NOTICE.txt",
    "policy/LICENSE.pdf",
    "motions/NOTICE.txt",
}
ASSETS = {
    ".html",
    ".js",
    ".mjs",
    ".css",
    ".wasm",
    ".json",
    ".onnx",
    ".xml",
    ".stl",
    ".png",
    ".glb",
    ".bin",
    ".svg",
}


def metadata(directory=BUNDLE):
    result = json.loads((directory / "bundle.json").read_text())
    if result["release_status"] not in {"integration-fixture", "public-release"}:
        raise ValueError("Unknown release status")
    return result


def safe_path(name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
        or str(path) != name
        or any(p.startswith(".") for p in path.parts)
    ):
        raise ValueError(f"Unsafe asset path: {name}")
    return path


def check_game(payload):
    """Application archive must contain bytecode/data, not Python sources."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        seen = set()
        for entry in archive.infolist():
            path = safe_path(entry.filename)
            if (
                entry.filename in seen
                or entry.is_dir()
                or stat.S_ISLNK(entry.external_attr >> 16)
                or path.suffix not in {".pyc", ".npz", ".json", ".pkl"}
            ):
                raise ValueError(f"Unexpected game member: {entry.filename}")
            seen.add(entry.filename)
        if "browser_game.pyc" not in seen:
            raise ValueError("Missing compiled game entry point")
        if sum(e.file_size for e in archive.infolist()) > 10_000_000:
            raise ValueError("Game archive exceeds size limit")
        if archive.testzip():
            raise ValueError("Corrupt game archive")


def check_network(info, source_sha256):
    """Pin remote weights without fetching them during install/CPU CI."""
    if (
        not re.fullmatch(
            r"https://huggingface\.co/nvidia/Metatron-v1/resolve/"
            r"[a-f0-9]{40}/racer/network\.onnx",
            str(info.get("url", "")),
        )
        or not re.fullmatch(r"[a-f0-9]{64}", str(info.get("sha256", "")))
        or not re.fullmatch(r"[a-f0-9]{64}", str(source_sha256))
        or info.get("source_sha256") != source_sha256
        or type(info.get("bytes")) is not int
        or not 0 < info["bytes"] <= 1024**3
    ):
        raise ValueError("Invalid pinned policy download contract")


def verify(directory=BUNDLE, *, allow_fixture=False):
    info = metadata(directory)
    if info["release_status"] != "public-release" and not allow_fixture:
        raise ValueError(
            "Bundle is not marked public-release; pass --allow-fixture to "
            "install it for review"
        )
    digest = hashlib.sha256()
    with (directory / "game.zip").open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != info["sha256"]:
        raise ValueError("Bundle checksum mismatch; fetch its Git LFS payload")
    with zipfile.ZipFile(directory / "game.zip") as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if (
            len(set(names)) != len(names)
            or set(names) != set(info["files"])
            or not REQUIRED <= set(names)
        ):
            raise ValueError("Bundle inventory or required assets mismatch")
        if sum(m.file_size for m in members) > 550_000_000:
            raise ValueError("Bundle exceeds hosting budget")
        for entry in members:
            name = entry.filename
            path = safe_path(name)
            notice = (
                name.startswith(("licenses/", "policy/", "motions/"))
                or name == "NOTICE.txt"
            ) and (path.suffix in {".txt", ".md", ".pdf"} or path.name == "LICENSE")
            runtime_archive = name == "game-runtime.zip" or (
                name.startswith("pyodide/") and path.suffix in {".zip", ".whl"}
            )
            if (
                entry.is_dir()
                or stat.S_ISLNK(entry.external_attr >> 16)
                or path.name in {"package.json", "package-lock.json"}
                or any(
                    p in {"src", "scripts", "node_modules", "tests"} for p in path.parts
                )
                or not (path.suffix in ASSETS or notice or runtime_archive)
            ):
                raise ValueError(f"Non-runtime file in bundle: {name}")
            if entry.file_size != info["files"][name]:
                raise ValueError(f"Incorrect asset size: {name}")
            if name == "observations.wasm" and any(
                section in archive.read(entry)
                for section in (b".debug_info", b".debug_line", b"sourceMappingURL")
            ):
                raise ValueError("Observation WASM must not contain debug/source maps")
            if path.suffix in {".js", ".mjs", ".css"} and re.search(
                rb"(?m)^\s*//[#@]\s*sourceMappingURL=|/\*[#@]\s*sourceMappingURL=",
                archive.read(entry),
            ):
                raise ValueError(f"Source map reference: {name}")
        check_game(archive.read("game-runtime.zip"))
        manifest = json.loads(archive.read("manifest.json"))
        remote = json.loads(archive.read("policy/network.json"))
        check_network(remote, manifest.get("artifact_sha256"))
        if info.get("policy_source_sha256") != manifest.get("artifact_sha256"):
            raise ValueError(
                "Bundle policy_source_sha256 disagrees with the policy manifest"
            )
        if "network.onnx" in names:
            raise ValueError("Hosted network must not also be bundled on Pages")
        if manifest["release_status"] != info["release_status"]:
            raise ValueError("Policy and bundle release statuses disagree")
        notices = json.loads(archive.read("licenses/third-party/manifest.json"))
        for name, expected in notices["files"].items():
            if (
                hashlib.sha256(archive.read("licenses/third-party/" + name)).hexdigest()
                != expected
            ):
                raise ValueError(f"Notice checksum mismatch: {name}")
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
        help="Install a bundle not marked public-release, for review",
    )
    args = parser.parse_args()
    if args.status:
        print(str(metadata()["release_status"] == "public-release").lower())
    elif args.check:
        verify(allow_fixture=args.allow_fixture)
        print("G1 Racer bundle verified")
    else:
        install(BUNDLE, args.out, allow_fixture=args.allow_fixture)
        print(f"Installed G1 Racer at {args.out}")


if __name__ == "__main__":
    main()
