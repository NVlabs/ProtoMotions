"""
Utilities to convert custom SMPL parameter exports into AMASS-style motion clips.

The converter expects the preprocessing directory produced by the downstream
pipeline described in ``training.helpers.dataset.FullSceneDataset``. The
directory must contain::

    cameras_normalize.npz
    mean_shape.npy
    poses.npy
    normalize_trans.npy

Each SMPL track is converted into an AMASS-compatible ``.npz`` file with the
keys used by ``convert_amass_to_isaac.py``. The resulting clips can be referenced
from a YAML motion descriptor and packaged with ``package_motion_lib.py``.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import typer


def _load_scale(preprocess_dir: Path) -> float:
    """Extract the global scale used by the normalization step."""
    camera_dict = np.load(preprocess_dir / "cameras_normalize.npz")
    scale_key_candidates = [key for key in camera_dict.files if key.startswith("scale_mat_")]
    if not scale_key_candidates:
        raise KeyError(
            "Could not find keys that start with 'scale_mat_' in cameras_normalize.npz. "
            "This file is required to recover the global scale."
        )

    scale_mat = camera_dict[scale_key_candidates[0]]
    return 1.0 / float(scale_mat[0, 0])


def _sanitize_sequence_name(name: str) -> str:
    """Produce filesystem friendly sequence names."""
    replacements = [
        (" ", "_"),
        ("(", ""),
        (")", ""),
        ("[", ""),
        ("]", ""),
    ]
    for src, dst in replacements:
        name = name.replace(src, dst)
    return name


def convert_tracks_to_amass(
    preprocess_dir: Path,
    output_dir: Path,
    track_ids: Optional[List[int]],
    fps: float,
    gender: str,
    sequence_name: Optional[str],
) -> None:
    preprocess_dir = preprocess_dir.resolve()
    output_dir = output_dir.resolve()

    if not preprocess_dir.exists():
        raise FileNotFoundError(f"Preprocess directory not found: {preprocess_dir}")

    poses_path = preprocess_dir / "poses.npy"
    trans_path = preprocess_dir / "normalize_trans.npy"
    betas_path = preprocess_dir / "mean_shape.npy"

    for path in (poses_path, trans_path, betas_path):
        if not path.exists():
            raise FileNotFoundError(f"Expected file is missing: {path}")

    poses = np.load(poses_path)
    translations = np.load(trans_path)
    betas = np.load(betas_path)

    if poses.ndim == 2:
        poses = poses[:, np.newaxis, :]
    if translations.ndim == 2:
        translations = translations[:, np.newaxis, :]
    if betas.ndim == 1:
        betas = betas[np.newaxis, :]

    num_frames, num_tracks, pose_dim = poses.shape
    if pose_dim not in {72, 24 * 3}:
        raise ValueError(
            f"Expected 72 axis-angle parameters per frame, got pose dimension {pose_dim}."
        )

    if translations.shape[:2] != (num_frames, num_tracks) or translations.shape[2] != 3:
        raise ValueError(
            "normalize_trans.npy is expected to have shape [num_frames, num_tracks, 3]."
        )

    if betas.shape[0] != num_tracks:
        raise ValueError(
            f"mean_shape.npy first dimension {betas.shape[0]} must match number of tracks {num_tracks}."
        )

    if track_ids is None or len(track_ids) == 0:
        track_ids = list(range(num_tracks))

    missing_ids = [tid for tid in track_ids if tid < 0 or tid >= num_tracks]
    if missing_ids:
        raise ValueError(
            f"Requested track ids {missing_ids} are out of bounds for {num_tracks} tracks."
        )

    scale = _load_scale(preprocess_dir)
    sequence_name = _sanitize_sequence_name(sequence_name or preprocess_dir.name)
    sequence_dir = output_dir / sequence_name
    sequence_dir.mkdir(parents=True, exist_ok=True)

    gender = gender.lower()
    if gender not in {"neutral", "male", "female"}:
        raise ValueError("gender must be one of {'neutral', 'male', 'female'}.")

    for track_id in track_ids:
        pose_seq = poses[:, track_id].astype(np.float32)
        trans_seq = (translations[:, track_id] * scale).astype(np.float32)
        betas_vec = betas[track_id].astype(np.float32)

        out_file = sequence_dir / f"{sequence_name}_track{track_id:02d}.npz"
        np.savez(
            out_file,
            poses=pose_seq,
            trans=trans_seq,
            betas=betas_vec,
            gender=np.array(gender),
            mocap_framerate=np.array(fps, dtype=np.float32),
        )
        print(f"[OK] Saved AMASS-style motion: {out_file}")


def main(
    preprocess_dir: Path = typer.Argument(..., help="Directory with custom preprocessing outputs."),
    output_dir: Path = typer.Argument(..., help="Destination directory for AMASS-style clips."),
    track_ids: Optional[List[int]] = typer.Option(
        None,
        "--track-id",
        "-t",
        help="Track id(s) to export. If omitted, all tracks are converted.",
    ),
    fps: float = typer.Option(30.0, help="Frame rate of the source sequence."),
    gender: str = typer.Option("neutral", help="SMPL gender tag to store in the clip metadata."),
    sequence_name: Optional[str] = typer.Option(
        None,
        help="Optional name for the exported sequence. Defaults to the preprocessing directory name.",
    ),
) -> None:
    """
    Convert SMPL parameters produced by the custom pipeline into AMASS-style motion files.
    """

    convert_tracks_to_amass(
        preprocess_dir=preprocess_dir,
        output_dir=output_dir,
        track_ids=track_ids,
        fps=fps,
        gender=gender,
        sequence_name=sequence_name,
    )


if __name__ == "__main__":
    typer.run(main)
