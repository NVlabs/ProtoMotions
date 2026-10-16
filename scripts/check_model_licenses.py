# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check published model licensing without importing ML packages or LFS data."""

import argparse
import hashlib
import json
from pathlib import Path
import re


DEFAULT_LICENSE = "NVIDIA Open Model License"
NOTICE = "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"
LICENSE_SHA256 = "4d2fb590aa9b30c47f2058bff17291df7fa2aa0c1bd775a20703da9bb267cfab"
MODEL_SUFFIXES = {".ckpt", ".onnx", ".safetensors", ".pth", ".pt", ".engine", ".plan"}


def model_directories(root):
    """Find cards AND uncarded payloads, including export-only model releases."""
    models = root / "data/pretrained_models"
    cards = {p.parent for p in models.rglob("MODEL_CARD.md")}
    directories = set(cards)
    for path in models.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MODEL_SUFFIXES:
            continue
        # Exported artifacts belong to their closest enclosing model card.
        owner = next((p for p in path.parents if p in cards), path.parent)
        directories.add(owner)
    return sorted(directories)


def check(root):
    """Return actionable errors; approved exceptions never disable card checks."""
    errors = []
    canonical = root / "legal/NVIDIA-Open-Model-License.pdf"
    if not canonical.is_file() or hashlib.sha256(canonical.read_bytes()).hexdigest() != LICENSE_SHA256:
        errors.append("Missing or altered canonical NVIDIA Open Model License PDF")
    manifest = root / "legal/model-license-exceptions.json"
    try:
        exceptions = json.loads(manifest.read_text())
    except (OSError, ValueError) as error:
        return errors + [f"{manifest}: {error}"]
    if not isinstance(exceptions, dict):
        return errors + ["Model license exceptions must be a JSON object"]
    directories = model_directories(root)
    if not directories:
        errors.append("No published model directories found")
    names = {p.relative_to(root).as_posix() for p in directories}
    for stale in exceptions.keys() - names:
        errors.append(f"Stale or unknown model license exception: {stale}")
    for directory in directories:
        name = directory.relative_to(root).as_posix()
        license_name, filename, reason = DEFAULT_LICENSE, "LICENSE.pdf", None
        if name in exceptions:
            entry = exceptions[name]
            if (not isinstance(entry, dict)
                    or set(entry) != {"license", "license_file", "reason"}
                    or not all(isinstance(v, str) and v.strip() for v in entry.values())):
                errors.append(f"{name}: exception requires license, license_file and reason")
                continue
            license_name, filename, reason = entry["license"], entry["license_file"], entry["reason"]
            if (license_name == DEFAULT_LICENSE or Path(filename).name != filename
                    or filename in {".", "..", "MODEL_CARD.md", "NOTICE.txt"}
                    or "\\" in filename):
                errors.append(f"{name}: invalid exception license or filename")
                continue
        card = directory / "MODEL_CARD.md"
        text = card.read_text() if card.is_file() else ""
        sections = re.findall(r"^## License\s*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
        section = sections[0] if len(sections) == 1 else ""
        if not re.search(r"^Model license: " + re.escape(license_name) + r"\s*$", section, re.M):
            errors.append(f"{name}: card must declare 'Model license: {license_name}' in ## License")
        if reason and f"License exception: {reason}" not in section:
            errors.append(f"{name}: document the exception rationale in ## License")
        for linked in (filename, "NOTICE.txt"):
            if f"]({linked})" not in section:
                errors.append(f"{name}: ## License must link {linked}")
        license_path = directory / filename
        if license_path.is_symlink() or not license_path.is_file() or not license_path.stat().st_size:
            errors.append(f"{name}: bundle a regular, nonempty {filename}")
        elif reason is None and hashlib.sha256(license_path.read_bytes()).hexdigest() != LICENSE_SHA256:
            errors.append(f"{name}: {filename} must match the canonical NVIDIA license")
        notice = directory / "NOTICE.txt"
        if notice.is_symlink() or not notice.is_file() or not notice.read_text().strip():
            errors.append(f"{name}: bundle a nonempty NOTICE.txt")
        elif reason is None and NOTICE not in notice.read_text():
            errors.append(f"{name}: NOTICE.txt is missing NVIDIA model attribution")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = check(args.root.resolve())
    for error in errors:
        print(error)
    if not errors:
        print("Model license checks passed.")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
