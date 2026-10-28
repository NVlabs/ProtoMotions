# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export, build, qualify, and time all native GPC TensorRT sampling modes."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("prepare:", shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def capture(command: list[str]) -> str:
    print("prepare:", shlex.join(command), flush=True)
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    output = result.stdout + result.stderr
    print(output, end="", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("argmax", "gumbel", "topp"),
        default=["argmax", "gumbel", "topp"],
    )
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--builder-optimization-level", type=int, default=3)
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="opt into TF32 tactics (not recommended for discrete GPC parity)",
    )
    parser.add_argument("--no-tf32", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--fp16", action="store_true", help="build FP16 TensorRT tactics"
    )
    parser.add_argument(
        "--plugin-library",
        type=Path,
        help="TopPFilter shared library; required when topp is selected",
    )
    parser.add_argument(
        "--native-benchmark",
        type=Path,
        default=Path("build-clang/gpc_tensorrt_native_benchmark"),
    )
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    experiment_dir = args.experiment_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    plugin = args.plugin_library.expanduser().resolve() if args.plugin_library else None
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"missing experiment directory: {experiment_dir}")
    if "topp" in args.modes and (plugin is None or not plugin.is_file()):
        raise FileNotFoundError("--plugin-library is required for topp")
    if not args.native_benchmark.expanduser().is_file():
        raise FileNotFoundError(
            f"missing native benchmark: {args.native_benchmark}; build with "
            "-DPROTOMOTIONS_BUILD_GPC_TENSORRT=ON"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "experiment_dir": str(experiment_dir),
        "modes": {},
        "status": "ok",
    }
    for mode in args.modes:
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        export_command = [
            sys.executable,
            str(repo / "tools/export_gpc_policy.py"),
            "--experiment-dir",
            str(experiment_dir),
            "--output-dir",
            str(mode_dir),
            "--mode",
            mode,
            "--top-p",
            str(args.top_p),
            "--temperature",
            str(args.temperature),
        ]
        run(export_command)
        onnx = mode_dir / "unified_pipeline.onnx"
        manifest = mode_dir / "unified_pipeline.json"
        engine = mode_dir / "unified_pipeline.plan"
        benchmark_command = [
            sys.executable,
            str(repo / "bench/gpc_tensorrt_benchmark.py"),
            str(onnx),
            "--manifest",
            str(manifest),
            "--engine",
            str(engine),
            "--build",
            "--skip-benchmark",
            "--workspace-mib",
            str(args.workspace_mib),
            "--builder-optimization-level",
            str(args.builder_optimization_level),
        ]
        if args.tf32 and not args.no_tf32:
            benchmark_command.append("--tf32")
        if args.fp16:
            benchmark_command.append("--fp16")
        if mode == "topp":
            benchmark_command.extend(["--plugin-library", str(plugin)])
        run(benchmark_command)
        manifest_data = json.loads(manifest.read_text())
        reference = onnx
        if mode == "topp":
            reference_name = manifest_data.get("reference_onnx")
            if not reference_name:
                raise RuntimeError("top-p manifest does not name reference_onnx")
            reference = mode_dir / str(reference_name)
        parity_command = [
            sys.executable,
            str(repo / "tools/validate_gpc_tensorrt.py"),
            "--reference-onnx",
            str(reference),
            "--manifest",
            str(manifest),
            "--engine",
            str(engine),
            "--cases",
            str(args.cases),
        ]
        if mode == "topp":
            parity_command.extend(["--plugin-library", str(plugin)])
        parity = capture(parity_command)
        timing_command = [
            str(args.native_benchmark.expanduser().resolve()),
            str(engine),
            str(args.samples),
            "--mode",
            mode,
            "--cuda-graphs",
            "--warmup",
            str(args.warmup),
        ]
        if mode == "topp":
            timing_command.extend(["--plugin-library", str(plugin)])
        timing = capture(timing_command)
        (mode_dir / "native_timing.txt").write_text(timing)
        report["modes"][mode] = {
            "engine": str(engine),
            "parity": parity,
            "native_timing": timing,
        }
    (output_dir / "gpc_prepare_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"prepared GPC modes under {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
