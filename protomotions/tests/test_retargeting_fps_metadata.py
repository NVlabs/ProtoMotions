import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_converter(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "data" / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))

    pose_lib = types.ModuleType("protomotions.components.pose_lib")
    for name in [
        "extract_kinematic_info",
        "fk_from_transforms_with_velocities",
        "compute_cartesian_velocity",
        "extract_transforms_from_qpos",
        "extract_qpos_from_transforms",
    ]:
        setattr(pose_lib, name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "protomotions.components.pose_lib", pose_lib)

    factory = types.ModuleType("protomotions.robot_configs.factory")
    factory.robot_config = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "protomotions.robot_configs.factory", factory)

    motion_filter = types.ModuleType("motion_filter")
    motion_filter.passes_exclude_motion_filter = lambda *args, **kwargs: True
    monkeypatch.setitem(sys.modules, "motion_filter", motion_filter)

    module_path = scripts_dir / "convert_pyroki_retargeted_robot_motions_to_proto.py"
    spec = importlib.util.spec_from_file_location("retargeted_converter", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_retargeted_npz(path: Path, fps: float) -> None:
    np.savez_compressed(
        path,
        base_frame_pos=np.arange(18, dtype=np.float32).reshape(6, 3),
        base_frame_wxyz=np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (6, 1)
        ),
        joint_angles=np.arange(12, dtype=np.float32).reshape(6, 2),
        fps=np.array(fps, dtype=np.float32),
    )


def test_process_npz_file_uses_embedded_fps_by_default(tmp_path, monkeypatch):
    converter = _load_converter(monkeypatch)
    npz_path = tmp_path / "motion_retargeted.npz"
    _write_retargeted_npz(npz_path, fps=50.0)

    root_pos, root_rot_wxyz, joint_angles, motion_fps = converter.process_npz_file(
        npz_path,
        input_fps=30,
        output_fps=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert motion_fps == pytest.approx(50.0)
    assert root_pos.shape == (6, 3)
    assert root_rot_wxyz.shape == (6, 4)
    assert joint_angles.shape == (6, 2)


def test_process_npz_file_downsamples_from_embedded_fps(tmp_path, monkeypatch):
    converter = _load_converter(monkeypatch)
    npz_path = tmp_path / "motion_retargeted.npz"
    _write_retargeted_npz(npz_path, fps=50.0)

    root_pos, _, _, motion_fps = converter.process_npz_file(
        npz_path,
        input_fps=30,
        output_fps=25,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert motion_fps == pytest.approx(25.0)
    assert root_pos.shape == (3, 3)
    assert root_pos[:, 0].tolist() == [0.0, 6.0, 12.0]


def test_process_npz_file_rejects_inexact_fps_downsample(tmp_path, monkeypatch):
    converter = _load_converter(monkeypatch)
    npz_path = tmp_path / "motion_retargeted.npz"
    _write_retargeted_npz(npz_path, fps=50.0)

    with pytest.raises(ValueError, match="integer multiple"):
        converter.process_npz_file(
            npz_path,
            input_fps=30,
            output_fps=30,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
