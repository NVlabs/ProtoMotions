# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the simulator recording mixin."""

import sys
import types
from types import SimpleNamespace

import torch

from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)
from protomotions.simulator.base_simulator.record import RecordingMixin


def _identity_quat(*shape: int):
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


class _Recorder(RecordingMixin):
    def __init__(self):
        self.config = SimpleNamespace(experiment_name="unit", w_last=True)
        self.headless = False
        self.num_envs = 2
        self.scene_lib = SimpleNamespace(num_objects_per_scene=0)
        self._num_dof = 2
        self._proj_config = None
        self._original_marker_configs = {}
        self.updated_markers = []
        self.written_files = []

    def _update_simulator_markers(self, markers_state):
        self.updated_markers.append(markers_state)

    def _write_viewport_to_file(self, file_name):
        self.written_files.append(file_name)

    def get_robot_state(self):
        return SimpleNamespace(
            rigid_body_pos=torch.arange(2 * 2 * 3, dtype=torch.float).reshape(2, 2, 3),
            rigid_body_rot=_identity_quat(2, 2),
            rigid_body_vel=torch.ones(2, 2, 3),
            rigid_body_ang_vel=torch.ones(2, 2, 3) * 2.0,
            dof_pos=torch.ones(2, 2) * 3.0,
            dof_vel=torch.ones(2, 2) * 4.0,
            rigid_body_contacts=torch.tensor([[True, False], [False, True]]),
        )

    def get_object_root_state(self):
        return SimpleNamespace(
            root_pos=torch.ones(2, 1, 3),
            root_rot=_identity_quat(2, 1),
        )

    def _get_projectile_positions_rotations(self):
        return torch.ones(2, 1, 3), _identity_quat(2, 1)


def test_recording_state_toggles_camera_and_marker_visibility(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder = _Recorder()
    recorder.scene_lib = SimpleNamespace(num_objects_per_scene=1)

    recorder._init_recording_state()
    assert recorder._camera_target == {"env": 0, "element": 0}
    assert recorder._show_markers is True
    assert recorder._user_recording_video_path.startswith("output/renderings/unit-")

    recorder._toggle_video_record()
    assert recorder._user_is_recording is True
    assert recorder._user_recording_state_change is True
    recorder._cancel_video_record()
    assert recorder._user_is_recording is False
    assert recorder._delete_user_viewer_recordings is True

    recorder._toggle_camera_target()
    assert recorder._camera_target == {"env": 0, "element": 1}
    recorder._toggle_camera_target()
    assert recorder._camera_target == {"env": 1, "element": 0}

    recorder.config.w_last = False
    marker = {
        "target": MarkerState(
            translation=torch.ones(2, 3),
            orientation=torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        )
    }
    recorder._toggle_markers()
    recorder._update_markers(marker)

    updated = recorder.updated_markers[-1]["target"]
    assert torch.equal(updated.orientation[0], torch.tensor([4.0, 1.0, 2.0, 3.0]))
    assert torch.equal(updated.translation, torch.full((2, 3), -1000000.0))
    recorder._update_markers({})
    assert len(recorder.updated_markers) == 1


def test_recording_builds_marker_terrain_and_object_save_payloads(monkeypatch):
    recorder = _Recorder()
    recorder._recorded_markers = {
        "target": [
            (torch.tensor([1.0, 2.0, 3.0]), _identity_quat()),
            (torch.tensor([4.0, 5.0, 6.0]), _identity_quat()),
        ],
        "unconfigured": [(torch.zeros(3), _identity_quat())],
    }
    recorder._original_marker_configs = {
        "target": VisualizationMarkerConfig(
            type="arrow",
            color=(0.1, 0.2, 0.3),
            markers=[MarkerConfig(size="tiny"), MarkerConfig(size="small")],
        )
    }

    marker_payload = recorder._build_markers_save_data()
    assert marker_payload["markers"]["target"]["type"] == "arrow"
    assert marker_payload["markers"]["target"]["sizes"] == ["tiny", "small"]
    assert marker_payload["markers"]["unconfigured"]["type"] == "sphere"

    recorder.terrain = None
    assert recorder._build_terrain_save_data() is None
    recorder.terrain = SimpleNamespace(is_flat=lambda: True)
    assert recorder._build_terrain_save_data() is None
    recorder.terrain = SimpleNamespace(
        is_flat=lambda: False,
        height_field_raw=torch.ones(2, 2),
        horizontal_scale=0.1,
        vertical_scale=0.2,
    )
    assert recorder._build_terrain_save_data()["horizontal_scale"] == 0.1

    class _BoxSceneObject:
        width = 1.0
        depth = 2.0
        height = 3.0

    class _SphereSceneObject:
        radius = 0.5

    class _CylinderSceneObject:
        radius = 0.25
        height = 1.5

    class _MeshSceneObject:
        object_path = "mesh.obj"
        scale = (1.0, 2.0, 3.0)

    class _UnknownObject:
        object_dims = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0]

    class _UnknownNoDimsObject:
        object_dims = None

    scene_module = types.ModuleType("protomotions.components.scene_lib")
    scene_module.BoxSceneObject = _BoxSceneObject
    scene_module.SphereSceneObject = _SphereSceneObject
    scene_module.CylinderSceneObject = _CylinderSceneObject
    scene_module.MeshSceneObject = _MeshSceneObject
    monkeypatch.setitem(sys.modules, "protomotions.components.scene_lib", scene_module)

    recorder._recording_env_id = 0
    recorder._recorded_objects = [
        (
            torch.ones(6, 3),
            _identity_quat(6),
        )
    ]
    recorder.scene_lib = SimpleNamespace(
        _scene_to_original_scene_id=torch.tensor([0]),
        _original_scenes=[
            SimpleNamespace(
                objects=[
                    _BoxSceneObject(),
                    _SphereSceneObject(),
                    _CylinderSceneObject(),
                    _MeshSceneObject(),
                    _UnknownObject(),
                    _UnknownNoDimsObject(),
                ]
            )
        ],
    )
    recorder._proj_config = SimpleNamespace(
        get_sizes=lambda: [0.1, 0.2],
        hide_z=-2.0,
    )
    recorder._recorded_projectiles = [
        (
            torch.tensor([[0.0, 0.0, -3.0], [1.0, 2.0, 0.0]]),
            _identity_quat(2),
        )
    ]

    payload = recorder._build_objects_save_data()
    shapes = [obj["shape"] for obj in payload["objects"]]

    assert shapes == ["box", "sphere", "cylinder", "mesh", "box", "box", "box"]
    assert payload["objects"][3]["mesh_path"] == "mesh.obj"
    assert payload["objects"][4]["size"] == [1.0, 2.0, 3.0]
    assert payload["objects"][5]["size"] == [0.1, 0.1, 0.1]
    assert payload["objects"][6]["name"] == "projectile_1"


def test_render_start_capture_finalize_and_cleanup_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder = _Recorder()
    recorder._init_recording_state()
    recorder._camera_target = {"env": 1, "element": 0}
    recorder._last_markers_state = {
        "terrain_markers": MarkerState(
            translation=torch.ones(2, 3),
            orientation=_identity_quat(2),
        ),
        "target": MarkerState(
            translation=torch.ones(2, 3) * 2.0,
            orientation=_identity_quat(2),
        ),
    }
    recorder.scene_lib = SimpleNamespace(num_objects_per_scene=1)
    recorder._proj_config = SimpleNamespace(num_projectiles=1)
    recorder._user_is_recording = True
    recorder._user_recording_state_change = True

    recorder.render()

    assert recorder._recording_env_id == 1
    assert recorder.written_files[-1].endswith("0000.png")
    assert len(recorder._recorded_motion["gts"]) == 1
    assert "target" in recorder._recorded_markers
    assert "terrain_markers" not in recorder._recorded_markers
    assert len(recorder._recorded_objects) == 1
    assert len(recorder._recorded_projectiles) == 1

    class _Clip:
        def __init__(self, images, fps):
            self.images = images
            self.fps = fps

        def write_videofile(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    moviepy = types.ModuleType("moviepy")
    moviepy.ImageSequenceClip = _Clip
    monkeypatch.setitem(sys.modules, "moviepy", moviepy)
    finish_dir = tmp_path / "finish"
    finish_dir.mkdir()
    (finish_dir / "0000.png").write_bytes(b"fake")
    recorder._curr_user_recording_name = str(finish_dir)
    recorder._recorded_motion = {
        "gts": [torch.zeros(2, 3)],
        "grs": [_identity_quat(2)],
        "gvs": [],
        "gavs": [],
        "dps": [],
        "dvs": [],
        "contacts": [],
    }
    recorder._recorded_markers = {}
    recorder._recorded_objects = []
    recorder._recorded_projectiles = []
    recorder._user_is_recording = False
    recorder._user_recording_state_change = True
    recorder.terrain = None

    recorder.render()

    assert not finish_dir.exists()
    assert (tmp_path / "finish.motion").exists()
    assert recorder._recorded_motion is None
    assert recorder._delete_user_viewer_recordings is False


def test_render_finalize_saves_marker_object_and_terrain_sidecars(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    recorder = _Recorder()
    recorder._init_recording_state()

    class _Clip:
        def __init__(self, images, fps):
            self.images = images
            self.fps = fps

        def write_videofile(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    moviepy = types.ModuleType("moviepy")
    moviepy.ImageSequenceClip = _Clip
    monkeypatch.setitem(sys.modules, "moviepy", moviepy)

    class _BoxSceneObject:
        width = 1.0
        depth = 1.0
        height = 1.0

    scene_module = types.ModuleType("protomotions.components.scene_lib")
    scene_module.BoxSceneObject = _BoxSceneObject
    scene_module.SphereSceneObject = type("_SphereSceneObject", (), {})
    scene_module.CylinderSceneObject = type("_CylinderSceneObject", (), {})
    scene_module.MeshSceneObject = type("_MeshSceneObject", (), {})
    monkeypatch.setitem(sys.modules, "protomotions.components.scene_lib", scene_module)

    saved_paths = []
    monkeypatch.setattr(torch, "save", lambda data, path: saved_paths.append(path))

    finish_dir = tmp_path / "with_sidecars"
    finish_dir.mkdir()
    (finish_dir / "0000.png").write_bytes(b"fake")
    recorder._curr_user_recording_name = str(finish_dir)
    recorder._recording_env_id = 0
    recorder._recorded_motion = {
        "gts": [torch.zeros(2, 3)],
        "grs": [_identity_quat(2)],
        "gvs": [],
        "gavs": [],
        "dps": [],
        "dvs": [],
        "contacts": [],
    }
    recorder._recorded_markers = {
        "target": [(torch.ones(3), _identity_quat())],
    }
    recorder._recorded_objects = [
        (
            torch.ones(1, 3),
            _identity_quat(1),
        )
    ]
    recorder._recorded_projectiles = []
    recorder.scene_lib = SimpleNamespace(
        _scene_to_original_scene_id=torch.tensor([0]),
        _original_scenes=[SimpleNamespace(objects=[_BoxSceneObject()])],
    )
    recorder.terrain = SimpleNamespace(
        is_flat=lambda: False,
        height_field_raw=torch.ones(2, 2),
        horizontal_scale=0.1,
        vertical_scale=0.2,
    )
    recorder._user_is_recording = False
    recorder._user_recording_state_change = True

    recorder.render()

    suffixes = sorted(
        ".".join(path.split(".")[-2:]) if path.endswith(".pt") else path.split(".")[-1]
        for path in saved_paths
    )
    assert suffixes == ["markers.pt", "motion", "objects.pt", "terrain.pt"]
    assert recorder._recorded_markers is None
    assert recorder._recorded_objects is None
    assert recorder._recorded_projectiles is None


def test_render_finalize_reports_sidecar_save_failures(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    recorder = _Recorder()
    recorder._init_recording_state()

    class _Clip:
        def __init__(self, images, fps):
            self.images = images
            self.fps = fps

        def write_videofile(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    moviepy = types.ModuleType("moviepy")
    moviepy.ImageSequenceClip = _Clip
    monkeypatch.setitem(sys.modules, "moviepy", moviepy)

    finish_dir = tmp_path / "sidecar_error"
    finish_dir.mkdir()
    (finish_dir / "0000.png").write_bytes(b"fake")
    recorder._curr_user_recording_name = str(finish_dir)
    recorder._recorded_motion = {
        "gts": [torch.zeros(2, 3)],
        "grs": [_identity_quat(2)],
        "gvs": [],
        "gavs": [],
        "dps": [],
        "dvs": [],
        "contacts": [],
    }
    recorder._recorded_markers = {
        "target": [(torch.ones(3), _identity_quat())],
    }
    recorder._recorded_objects = []
    recorder._recorded_projectiles = []
    recorder.terrain = None
    recorder._user_is_recording = False
    recorder._user_recording_state_change = True

    def _raise_marker_payload():
        raise RuntimeError("marker payload failed")

    monkeypatch.setattr(recorder, "_build_markers_save_data", _raise_marker_payload)

    recorder.render()

    assert "Warning: failed to save markers/objects/terrain" in capsys.readouterr().out
