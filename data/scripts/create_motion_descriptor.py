from __future__ import annotations

import os
from pathlib import Path
from typing import List

import yaml
import typer


def collect_motion_files(root: Path, patterns: List[str]) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Converted motion root does not exist: {root}")

    files: List[Path] = []
    if patterns:
        for pattern in patterns:
            files.extend(root.glob(pattern))
    else:
        files.extend(root.rglob("*.npy"))

    files = sorted({path.resolve() for path in files if path.is_file()})
    if not files:
        raise FileNotFoundError(f"No motion files found under {root} for patterns={patterns}")
    return files


def make_entry(idx: int, rel_path: str, fps: float) -> dict:
    return {
        "file": rel_path,
        "fps": float(fps),
        "idx": idx,
        "sub_motions": [
            {
                "idx": idx,
                "timings": {
                    "start": 0.0,
                    "end": -1,
                },
                "weight": 1.0,
            }
        ],
    }


def main(
    converted_root: Path = typer.Argument(..., help="Root directory containing Isaac/poselib motion npy files."),
    output_path: Path = typer.Argument(..., help="Destination YAML descriptor."),
    fps: float = typer.Option(30.0, help="Frame rate stored with each motion."),
    sequence_subdir: str = typer.Option(
        "",
        help="Optional sub-directory (relative to converted_root) that contains the motions for this descriptor.",
    ),
    file_glob: List[str] = typer.Option(
        ["**/*.npy"],
        "--file-glob",
        "-g",
        help="Glob pattern(s) (relative to the selected directory) used to find motion files.",
        show_default=True,
    ),
) -> None:
    """
    Generate a motion descriptor YAML that lists the converted motion files.

    The descriptor is compatible with MotionLib's packaging script.
    """
    patterns = file_glob or ["**/*.npy"]
    source_dir = converted_root / sequence_subdir if sequence_subdir else converted_root
    motion_files = collect_motion_files(source_dir, patterns)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_dir = output_path.parent.resolve()
    entries = []
    for idx, file_path in enumerate(motion_files):
        rel_path = os.path.relpath(file_path, yaml_dir)
        entries.append(make_entry(idx, rel_path, fps))

    descriptor = {"motions": entries}
    with output_path.open("w") as f:
        yaml.safe_dump(descriptor, f, sort_keys=False)

    typer.echo(f"[OK] Saved motion descriptor with {len(entries)} entries to {output_path}")


if __name__ == "__main__":
    typer.run(main)
