# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a qualified Metatron v1 ONNX/manifest pair as a LEAPP bundle.

Packaging preserves the exact model bytes, action mapping and history contract.
It does not retrace a checkpoint or requalify the policy for hardware.
"""

import argparse
import json
from pathlib import Path

from examples.metatron_v1_contract import validate_manifest
from tools.leapp_export import export_onnx_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="new bundle directory"
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    print(export_onnx_bundle(args.model, manifest, args.output))


if __name__ == "__main__":
    main()
