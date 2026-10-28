# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package an exported scene policy with its reviewed sensor-only recovery plan.

No weights, recovery recordings, credentials or hardware-specific paths belong
in git. This command creates a local artifact directory for the deployment CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from examples.metatron_v1_contract import validate_manifest
from examples.policy_artifact import resolve_policy_artifact


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare(
    model: Path, manifest_path: Path | None, recovery_plan: Path, output: Path
) -> None:
    model, manifest_path = resolve_policy_artifact(model, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    model_hash = digest(model)
    if model_hash != manifest["artifact_sha256"]:
        raise ValueError("ONNX checksum differs from the selected policy manifest")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(model, output / "model.onnx")
    shutil.copy2(recovery_plan, output / "recovery_plan.json")
    # The native package consumes ONNX/JSON, not YAML. Do not leave a dangling
    # pointer to the source bundle after materializing its unchanged model.
    manifest.pop("leapp_bundle", None)
    manifest["deployment_recovery_plan"] = {
        "file": "recovery_plan.json",
        "sha256": digest(recovery_plan),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(output),
                "model_sha256": model_hash,
                "recovery_plan_sha256": manifest["deployment_recovery_plan"]["sha256"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="ONNX file or LEAPP bundle YAML"
    )
    parser.add_argument(
        "--manifest", type=Path, help="defaults to the artifact's sidecar"
    )
    parser.add_argument("--recovery-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model, args.manifest, args.recovery_plan, args.output)


if __name__ == "__main__":
    main()
