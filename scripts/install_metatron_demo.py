# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify and install the prebuilt demo, without building application source."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

BUNDLE = Path(__file__).resolve().parents[1] / "demos/metatron"
REQUIRED_FILES = {
    "index.html",
    "threading-worker.js",
    "motion-notice.html",
    "assets/policy/manifest.json",
    "assets/policy/MODEL_CARD.md",
    "assets/policy/LICENSE.pdf",
    "assets/policy/NOTICE.txt",
    "licenses/Apache-2.0.txt",
    "licenses/robot-notices.txt",
    "licenses/third-party/manifest.json",
    "licenses/third-party/README.md",
    "assets/motions/NOTICE.txt",
    "assets/styles/NOTICE.txt",
    "assets/recovery/NOTICE.txt",
}
ASSET_SUFFIXES = {
    ".html",
    ".js",
    ".css",
    ".wasm",
    ".json",
    ".svg",
    ".mp4",
    ".xml",
    ".obj",
    ".stl",
    ".png",
    ".jpg",
    ".glb",
}


def read_metadata(directory: Path) -> dict:
    """Read release metadata even in a checkout without the LFS payload."""
    metadata = json.loads((directory / "bundle.json").read_text())
    if metadata["release_status"] not in {"integration-fixture", "public-release"}:
        raise ValueError("Unknown demo release status")
    return metadata


def validate_path(name: str) -> None:
    """Allow only relative runtime assets and notices, never development files."""
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
        or str(path) != name
    ):
        raise ValueError(f"Unsafe bundle path: {name}")
    if any(
        part.startswith(".") or part in {"src", "node_modules", "scripts", "tests"}
        for part in path.parts
    ):
        raise ValueError(f"Development file in bundle: {name}")
    notice = (name.startswith("licenses/") or name in REQUIRED_FILES) and (
        path.suffix in {".txt", ".md", ".pdf"} or path.name == "LICENSE"
    )
    if path.suffix not in ASSET_SUFFIXES and not notice:
        raise ValueError(f"Not a runtime asset or notice: {name}")
    if path.suffix in {".map", ".ts", ".tsx"} or path.name in {
        "package.json",
        "package-lock.json",
    }:
        raise ValueError(f"Source file in bundle: {name}")


def verify_bundle(directory: Path, *, allow_fixture: bool = False) -> dict:
    """Check the archive, its inventory, embedded release status and notices."""
    metadata = read_metadata(directory)
    if metadata["release_status"] != "public-release" and not allow_fixture:
        raise ValueError(
            "Demo is an integration fixture, not approved for public release"
        )
    archive_path = directory / "demo.zip"
    digest = hashlib.sha256()
    with archive_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != metadata["sha256"]:
        raise ValueError("Demo archive checksum mismatch; fetch the pinned LFS payload")
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [entry.filename for entry in members]
        if len(set(names)) != len(names) or set(names) != set(metadata["files"]):
            raise ValueError("Archive does not match its file inventory")
        if not REQUIRED_FILES <= set(names):
            raise ValueError(
                f"Missing runtime files/notices: {REQUIRED_FILES - set(names)}"
            )
        if sum(entry.file_size for entry in members) > 200_000_000:
            raise ValueError("Demo exceeds the Pages size budget")
        for entry in members:
            validate_path(entry.filename)
            if stat.S_ISLNK(entry.external_attr >> 16) or entry.is_dir():
                raise ValueError(f"Expected a regular asset: {entry.filename}")
            if entry.file_size != metadata["files"][entry.filename]:
                raise ValueError(f"Incorrect asset size: {entry.filename}")
            if entry.filename.endswith((".js", ".css")):
                if b"sourceMappingURL" in archive.read(entry):
                    raise ValueError(f"Source map reference: {entry.filename}")
        policy = json.loads(archive.read("assets/policy/manifest.json"))
        if policy["release_status"] != metadata["release_status"]:
            raise ValueError("Bundle and policy release statuses disagree")
        download = policy.get("model_download", {})
        if not isinstance(download, dict):
            raise ValueError("Invalid pinned policy download contract")
        if (
            not re.fullmatch(
                r"https://huggingface\.co/nvidia/Metatron-v1/resolve/"
                r"[a-f0-9]{40}/demo/model\.onnx",
                str(download.get("url", "")),
            )
            or not re.fullmatch(r"[a-f0-9]{64}", str(download.get("sha256", "")))
            or download.get("sha256") != policy.get("artifact_sha256")
            or policy.get("model_storage") is not None
        ):
            raise ValueError("Invalid pinned policy download contract")
        if any(name.startswith("assets/policy/model.onnx") for name in names):
            raise ValueError("Hosted weights must not also be bundled on Pages")
        licenses = json.loads(archive.read("licenses/third-party/manifest.json"))
        for name, expected in licenses["files"].items():
            member = f"licenses/third-party/{name}"
            try:
                data = archive.read(member)
            except KeyError:
                raise ValueError(f"Third-party manifest names a missing member: {name}")
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Third-party notice checksum mismatch: {name}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Corrupt asset: {bad_member}")
    return metadata


def install_bundle(
    directory: Path, destination: Path, *, allow_fixture: bool = False
) -> None:
    """Extract verified assets into a new directory; never overlay a stale build."""
    verify_bundle(directory, allow_fixture=allow_fixture)
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Destination already exists; pass --out at a new directory: {destination}"
        ) from exc
    with zipfile.ZipFile(directory / "demo.zip") as archive:
        for entry in archive.infolist():
            target = destination / entry.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--status",
        action="store_true",
        help="Print public-release readiness without loading assets",
    )
    mode.add_argument("--check", action="store_true", help="Verify without extracting")
    mode.add_argument("--out", type=Path, help="New directory for the static site")
    parser.add_argument(
        "--allow-fixture",
        action="store_true",
        help="Private preview only; never use for public Pages",
    )
    args = parser.parse_args()
    if args.status:
        print(str(read_metadata(BUNDLE)["release_status"] == "public-release").lower())
    elif args.check:
        verify_bundle(BUNDLE, allow_fixture=args.allow_fixture)
        print("Demo archive verified")
    else:
        install_bundle(BUNDLE, args.out, allow_fixture=args.allow_fixture)
        print(f"Demo installed at {args.out}")


if __name__ == "__main__":
    main()
