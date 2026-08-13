# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kit-free tests for IsaacLab MJCF→USD conversion helpers."""

from __future__ import annotations

import ast
import importlib.util
import multiprocessing
import time
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and earlier
    import tomli as tomllib
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch

import protomotions.simulator.isaaclab.utils.mjcf_to_usd as mjcf_to_usd
import protomotions.simulator.isaaclab.utils.mjcf_d6_workaround as mjcf_d6_workaround
import protomotions.simulator.isaaclab.utils.usd_body_paths as usd_body_paths
from protomotions.robot_configs.base import RobotAssetConfig
from protomotions.robot_configs.g1 import G1RobotConfig
from protomotions.robot_configs.smpl import SmplRobotConfig
from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig
from protomotions.simulator.isaaclab.utils.mjcf_to_usd import (
    build_mjcf_converter_cfg_kwargs,
    clear_mjcf_usd_conversion_cache,
    conversion_cache_key,
    convert_mjcf_to_usd,
    convert_robot_mjcf_to_usd,
    default_usd_cache_dir,
    dry_run_mjcf_converter_factory,
    predicted_converted_usd_path,
    resolve_robot_mjcf_path,
)
from protomotions.simulator.isaaclab.utils.usd_body_paths import (
    contact_sensor_prim_path,
    resolve_articulation_root_prim_path,
    resolve_articulation_root_prim_path_from_records,
    resolve_body_prim_paths,
    resolve_body_prim_paths_from_records,
)


def _recording_mjcf_converter_factory(**cfg_kwargs):
    call_record = Path(cfg_kwargs["usd_dir"]).parent / "factory-calls.txt"
    with call_record.open("a") as stream:
        stream.write("called\n")
    time.sleep(0.1)
    usd_path = Path(
        predicted_converted_usd_path(
            cfg_kwargs["asset_path"], cfg_kwargs["usd_dir"]
        )
    )
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    usd_path.write_text("#usda 1.0")
    return str(usd_path)


def _run_concurrent_mjcf_conversion(mjcf_path, usd_dir, result_queue):
    try:
        result = convert_mjcf_to_usd(
            mjcf_path,
            converter_factory=_recording_mjcf_converter_factory,
            cache={},
            usd_dir=usd_dir,
        )
    except Exception as exc:
        result_queue.put(("error", repr(exc)))
    else:
        result_queue.put(("ok", result))


def _load_offline_converter():
    path = (
        Path(__file__).parents[2]
        / "usd_convert"
        / "convert_robot_mjcf_to_usda.py"
    )
    spec = importlib.util.spec_from_file_location("offline_mjcf_converter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mjcf_d6_workaround_is_kit_lazy_and_idempotent(monkeypatch):
    original = lambda *args: False
    original_combine = lambda *args: 0
    fake_module = types.SimpleNamespace(
        _convert_overconstrained_group_to_d6=original,
        combine_overconstrained_joints_in_physx_layer=original_combine,
        convert_mjc_to_physx=lambda *args: None,
        combine_overconstrained_joints_to_d6=lambda *args: 0,
    )
    monkeypatch.setattr(
        mjcf_d6_workaround.importlib,
        "import_module",
        lambda module_name: fake_module,
    )

    mjcf_d6_workaround.install_isaaclab_mjcf_d6_workaround()
    patched = fake_module._convert_overconstrained_group_to_d6
    mjcf_d6_workaround.install_isaaclab_mjcf_d6_workaround()

    assert patched is not original
    assert fake_module._convert_overconstrained_group_to_d6 is patched


def test_mjcf_d6_workaround_repatches_after_converter_reload(monkeypatch):
    original = lambda *args: False
    original_combine = lambda *args: 0
    fake_module = types.SimpleNamespace(
        _convert_overconstrained_group_to_d6=original,
        combine_overconstrained_joints_in_physx_layer=original_combine,
        convert_mjc_to_physx=lambda *args: None,
        combine_overconstrained_joints_to_d6=lambda *args: 0,
    )
    monkeypatch.setattr(
        mjcf_d6_workaround.importlib,
        "import_module",
        lambda module_name: fake_module,
    )

    mjcf_d6_workaround.install_isaaclab_mjcf_d6_workaround()
    first_patch = fake_module._convert_overconstrained_group_to_d6

    # importlib.reload keeps the module object but restores its functions.
    fake_module._convert_overconstrained_group_to_d6 = original
    fake_module.combine_overconstrained_joints_in_physx_layer = original_combine
    mjcf_d6_workaround.install_isaaclab_mjcf_d6_workaround()

    assert fake_module._convert_overconstrained_group_to_d6 is not original
    assert fake_module._convert_overconstrained_group_to_d6 is not first_patch


def test_mjcf_d6_composed_conversion_runs_metadata_conversion_first(
    tmp_path, monkeypatch
):
    from pxr import Sdf, Usd

    events = []
    original_group = lambda *args: False
    original_combine = lambda *args: 0

    def convert_mjc_to_physx(stage):
        events.append(("convert", stage.GetEditTarget().GetLayer().identifier))
        prim = stage.GetPrimAtPath("/World/joint")
        prim.CreateAttribute("test:preserved", Sdf.ValueTypeNames.Float).Set(7.0)

    def combine_d6(stage):
        events.append(("combine", stage.GetEditTarget().GetLayer().identifier))
        return 1

    fake_module = types.SimpleNamespace(
        _convert_overconstrained_group_to_d6=original_group,
        combine_overconstrained_joints_in_physx_layer=original_combine,
        convert_mjc_to_physx=convert_mjc_to_physx,
        combine_overconstrained_joints_to_d6=combine_d6,
    )
    monkeypatch.setattr(
        mjcf_d6_workaround.importlib,
        "import_module",
        lambda module_name: fake_module,
    )
    mjcf_d6_workaround.install_isaaclab_mjcf_d6_workaround()

    physx_path = tmp_path / "physx.usda"
    physx_layer = Sdf.Layer.CreateNew(str(physx_path))
    physx_stage = Usd.Stage.Open(physx_layer)
    physx_stage.DefinePrim("/World/joint", "PhysicsJoint")
    physx_stage.GetRootLayer().Save()
    mujoco_layer = Sdf.Layer.CreateNew(str(tmp_path / "mujoco.usda"))
    mujoco_layer.Save()

    result = fake_module.combine_overconstrained_joints_in_physx_layer(
        str(physx_path)
    )

    assert result == 1
    assert [event[0] for event in events] == ["convert", "combine"]
    assert all(event[1] == physx_layer.identifier for event in events)
    converted_stage = Usd.Stage.Open(str(physx_path))
    assert converted_stage.GetPrimAtPath("/World/joint").GetAttribute(
        "test:preserved"
    ).Get() == 7.0


def test_mjcf_d6_repair_ignores_dropped_duplicate_axis():
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    d6_path = Sdf.Path("/World/hip")
    d6_prim = stage.DefinePrim(d6_path, "PhysicsJoint")
    retained_path = Sdf.Path("/World/hip_x")
    duplicate_path = Sdf.Path("/World/hip_x_duplicate")
    y_path = Sdf.Path("/World/hip_y")
    states = [
        mjcf_d6_workaround._SourceAxisState(
            path=retained_path,
            token="rotX",
            local_rot0=None,
            local_rot1=None,
            drive_state={"stiffness": 10.0},
        ),
        mjcf_d6_workaround._SourceAxisState(
            path=duplicate_path,
            token="rotX",
            local_rot0=None,
            local_rot1=None,
            drive_state={"stiffness": 99.0},
        ),
        mjcf_d6_workaround._SourceAxisState(
            path=y_path,
            token="rotY",
            local_rot0=None,
            local_rot1=None,
            drive_state={"stiffness": 20.0},
        ),
    ]

    mjcf_d6_workaround._repair_d6_group(
        stage,
        states,
        {retained_path: d6_path, y_path: d6_path},
        UsdPhysics,
    )

    assert UsdPhysics.DriveAPI(d6_prim, "rotX").GetStiffnessAttr().Get() == 10.0
    assert UsdPhysics.DriveAPI(d6_prim, "rotY").GetStiffnessAttr().Get() == 20.0


def test_isaaclab_config_defaults_to_xyzw():
    config = IsaacLabSimulatorConfig(
        headless=True,
        num_envs=1,
        experiment_name="unit",
    )
    assert config.w_last is True


def test_humanoid_robot_configs_use_mjcf_only():
    for config_cls in (G1RobotConfig, SmplRobotConfig):
        asset = config_cls().asset
        assert not hasattr(asset, "usd_asset_file_name")
        assert not hasattr(asset, "usd_bodies_root_prim_path")
        assert asset.asset_file_name.endswith(".xml")


def test_tracked_resolved_sidecars_omit_legacy_usd_fields():
    artifacts_root = Path(__file__).parents[2] / "data" / "pretrained_models"
    for yaml_path in artifacts_root.glob("**/resolved_configs*.yaml"):
        text = yaml_path.read_text()
        assert "usd_asset_file_name:" not in text
        assert "usd_bodies_root_prim_path:" not in text

    for pt_path in artifacts_root.glob("**/resolved_configs*.pt"):
        configs = torch.load(pt_path, weights_only=False)
        asset = configs["robot"].asset
        assert not hasattr(asset, "usd_asset_file_name")
        assert not hasattr(asset, "usd_bodies_root_prim_path")


def test_resolve_robot_mjcf_path_and_predicted_usd(tmp_path):
    mjcf = tmp_path / "robot.xml"
    mjcf.write_text("<mujoco/>")
    asset = RobotAssetConfig(
        asset_root=str(tmp_path),
        asset_file_name="robot.xml",
        self_collisions=True,
    )
    resolved = resolve_robot_mjcf_path(asset)
    assert resolved == str(mjcf.resolve())

    usd_dir = tmp_path / "usd_out"
    predicted = predicted_converted_usd_path(resolved, str(usd_dir))
    assert predicted == str((usd_dir / "robot" / "robot.usda").resolve())


def test_resolve_robot_mjcf_path_honors_asset_root_override(tmp_path, monkeypatch):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    mjcf = asset_root / "robot.xml"
    mjcf.write_text("<mujoco/>")
    monkeypatch.setenv("PROTOMOTIONS_ASSET_ROOT", str(asset_root))

    asset = RobotAssetConfig(asset_file_name="robot.xml")

    assert resolve_robot_mjcf_path(asset) == str(mjcf.resolve())

    monkeypatch.delenv("PROTOMOTIONS_ASSET_ROOT")
    monkeypatch.chdir(tmp_path)
    packaged_asset = RobotAssetConfig(asset_file_name="mjcf/g1_bm.xml")
    assert Path(resolve_robot_mjcf_path(packaged_asset)).is_file()


def test_path_helpers_expand_user_paths(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    mjcf = home / "robot.xml"
    mjcf.write_text("<mujoco/>")

    asset = RobotAssetConfig(asset_root="~", asset_file_name="robot.xml")
    assert resolve_robot_mjcf_path(asset) == str(mjcf.resolve())

    predicted = predicted_converted_usd_path("~/robot.xml", "~/.cache/usd")
    assert predicted == str((home / ".cache/usd/robot/robot.usda").resolve())

    kwargs = build_mjcf_converter_cfg_kwargs("~/robot.xml", usd_dir="~/usd")
    assert kwargs["asset_path"] == str(mjcf.resolve())
    assert kwargs["usd_dir"] == str((home / "usd").resolve())

    cache_dir = default_usd_cache_dir("~/robot.xml", {})
    assert cache_dir.startswith(str(home / ".cache"))
    assert "~" not in cache_dir


def test_build_mjcf_converter_cfg_kwargs_matches_isaaclab3_fields(tmp_path):
    mjcf = tmp_path / "g1.xml"
    mjcf.write_text("<mujoco/>")
    kwargs = build_mjcf_converter_cfg_kwargs(
        str(mjcf),
        usd_dir=str(tmp_path / "cache"),
        force_usd_conversion=True,
        self_collision=True,
        fix_base=False,
        merge_mesh=False,
        collision_from_visuals=False,
    )
    assert kwargs["asset_path"] == str(mjcf.resolve())
    assert kwargs["usd_dir"] == str((tmp_path / "cache").resolve())
    assert kwargs["force_usd_conversion"] is True
    assert kwargs["self_collision"] is True
    assert kwargs["fix_base"] is False
    assert "import_sites" not in kwargs
    assert "make_instanceable" not in kwargs


def test_convert_mjcf_to_usd_caches_and_accepts_factory(tmp_path):
    clear_mjcf_usd_conversion_cache()
    mjcf = tmp_path / "bot.xml"
    mjcf.write_text("<mujoco a/>")
    calls = {"n": 0}

    def factory(**cfg_kwargs):
        calls["n"] += 1
        return dry_run_mjcf_converter_factory(**cfg_kwargs)

    first = convert_mjcf_to_usd(
        str(mjcf),
        converter_factory=factory,
        usd_dir=str(tmp_path / "cache"),
        self_collision=True,
    )
    second = convert_mjcf_to_usd(
        str(mjcf),
        converter_factory=factory,
        usd_dir=str(tmp_path / "cache"),
        self_collision=True,
    )
    assert first == second
    assert calls["n"] == 1

    mjcf.write_text("<mujoco b/>")
    third = convert_mjcf_to_usd(
        str(mjcf),
        converter_factory=factory,
        usd_dir=str(tmp_path / "cache"),
        self_collision=True,
    )
    assert third == first
    assert calls["n"] == 2
    assert first.endswith("bot/bot.usda")

    key = conversion_cache_key(
        build_mjcf_converter_cfg_kwargs(
            str(mjcf),
            usd_dir=str(tmp_path / "cache"),
            self_collision=True,
        )
    )
    assert key[1] == str(mjcf.resolve())


def test_convert_mjcf_to_usd_forces_existing_unmarked_output(tmp_path):
    clear_mjcf_usd_conversion_cache()
    mjcf = tmp_path / "bot.xml"
    mjcf.write_text("<mujoco/>")
    usd_dir = tmp_path / "cache"
    stale_path = Path(predicted_converted_usd_path(str(mjcf), str(usd_dir)))
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text("stale")
    seen_force_values = []

    def factory(**cfg_kwargs):
        seen_force_values.append(cfg_kwargs["force_usd_conversion"])
        stale_path.write_text("fresh")
        return str(stale_path)

    result = convert_mjcf_to_usd(
        str(mjcf),
        converter_factory=factory,
        cache={},
        usd_dir=str(usd_dir),
    )

    assert result == str(stale_path.resolve())
    assert seen_force_values == [True]
    assert stale_path.read_text() == "fresh"


def test_convert_mjcf_to_usd_fingerprints_dependencies_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    mjcf = tmp_path / "bot.xml"
    mjcf.write_text("<mujoco/>")
    calls = []
    real_fingerprint = mjcf_to_usd._mjcf_fingerprint

    def recording_fingerprint(path):
        calls.append(path)
        return real_fingerprint(path)

    monkeypatch.setattr(mjcf_to_usd, "_mjcf_fingerprint", recording_fingerprint)

    convert_mjcf_to_usd(
        str(mjcf),
        converter_factory=dry_run_mjcf_converter_factory,
        cache={},
    )

    assert calls == [str(mjcf.resolve())]


def test_convert_mjcf_to_usd_serializes_shared_cache_publication(tmp_path):
    clear_mjcf_usd_conversion_cache()
    mjcf = tmp_path / "bot.xml"
    mjcf.write_text("<mujoco/>")
    usd_dir = tmp_path / "cache"
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_run_concurrent_mjcf_conversion,
            args=(str(mjcf), str(usd_dir), result_queue),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)

    assert [process.exitcode for process in processes] == [0, 0]
    results = [result_queue.get(timeout=1) for _ in processes]
    assert all(status == "ok" for status, _ in results)
    assert results[0][1] == results[1][1]
    assert (tmp_path / "factory-calls.txt").read_text().splitlines() == ["called"]


def test_convert_mjcf_to_usd_invalidates_referenced_mesh_changes(tmp_path):
    clear_mjcf_usd_conversion_cache()
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    mesh = mesh_dir / "body.stl"
    mesh.write_text("A")
    mjcf = tmp_path / "bot.xml"
    mjcf.write_text(
        '<mujoco><compiler meshdir="mesh"/><asset><mesh file="body.stl"/></asset></mujoco>'
    )
    calls = {"n": 0}

    def factory(**cfg_kwargs):
        calls["n"] += 1
        return dry_run_mjcf_converter_factory(**cfg_kwargs)

    convert_mjcf_to_usd(
        str(mjcf), converter_factory=factory, usd_dir=str(tmp_path / "cache")
    )
    convert_mjcf_to_usd(
        str(mjcf), converter_factory=factory, usd_dir=str(tmp_path / "cache")
    )
    assert calls["n"] == 1

    mesh.write_text("B")
    convert_mjcf_to_usd(
        str(mjcf), converter_factory=factory, usd_dir=str(tmp_path / "cache")
    )
    assert calls["n"] == 2


def test_convert_robot_mjcf_to_usd_uses_asset_physics_options(tmp_path):
    clear_mjcf_usd_conversion_cache()
    mjcf = tmp_path / "humanoid.xml"
    mjcf.write_text("<mujoco/>")
    asset = RobotAssetConfig(
        asset_root=str(tmp_path),
        asset_file_name="humanoid.xml",
        self_collisions=True,
        fix_base_link=True,
    )
    seen = {}

    def factory(**cfg_kwargs):
        seen.update(cfg_kwargs)
        return dry_run_mjcf_converter_factory(**cfg_kwargs)

    usd_path = convert_robot_mjcf_to_usd(
        asset,
        converter_factory=factory,
        usd_dir=str(tmp_path / "cache"),
    )
    assert seen["self_collision"] is True
    assert seen["fix_base"] is True
    assert usd_path.endswith("humanoid/humanoid.usda")


def test_convert_robot_mjcf_to_usd_allows_explicit_fix_base_override(tmp_path):
    mjcf = tmp_path / "humanoid.xml"
    mjcf.write_text("<mujoco/>")
    asset = RobotAssetConfig(
        asset_root=str(tmp_path),
        asset_file_name="humanoid.xml",
        fix_base_link=True,
    )
    seen = {}

    def factory(**cfg_kwargs):
        seen.update(cfg_kwargs)
        return dry_run_mjcf_converter_factory(**cfg_kwargs)

    convert_robot_mjcf_to_usd(
        asset,
        converter_factory=factory,
        cache={},
        usd_dir=str(tmp_path / "cache"),
        fix_base=False,
    )

    assert seen["fix_base"] is False


def test_offline_cleaned_mjcf_preserves_relative_asset_directories(tmp_path):
    converter = _load_offline_converter()
    source_dir = tmp_path / "source" / "mjcf"
    source_dir.mkdir(parents=True)
    mjcf = source_dir / "robot.xml"
    mjcf.write_text(
        '<mujoco><compiler meshdir="../mesh" texturedir="../textures"/>'
        '<asset><mesh name="body" file="body.stl"/>'
        '<texture name="skin" type="2d" file="skin.png"/></asset></mujoco>'
    )
    cleaned = tmp_path / "cleaned" / "robot.xml"
    cleaned.parent.mkdir()

    converter.strip_mjcf(str(mjcf), str(cleaned))

    compiler = ET.parse(cleaned).getroot().find("compiler")
    assert compiler.get("meshdir") == str((source_dir / "../mesh").resolve())
    assert compiler.get("texturedir") == str(
        (source_dir / "../textures").resolve()
    )


def test_offline_converter_rejects_unflattened_includes(tmp_path):
    converter = _load_offline_converter()
    mjcf = tmp_path / "robot.xml"
    mjcf.write_text('<mujoco><include file="defaults.xml"/></mujoco>')

    assert any(
        "include" in issue
        for issue in converter.verify_mjcf_is_flat(str(mjcf))
    )


def test_resolve_body_prim_paths_nested_and_flat():
    records = [
        {
            "name": "pelvis",
            "full_path": "/robot/Geometry/pelvis",
            "is_rigid_body": True,
        },
        {
            "name": "left_ankle_roll_link",
            "full_path": "/robot/Geometry/pelvis/left_ankle_roll_link",
            "is_rigid_body": True,
        },
        {
            "name": "visual_only",
            "full_path": "/robot/Geometry/pelvis/visual_only",
            "is_rigid_body": False,
        },
        {
            "name": "pelvis",
            "full_path": "/other/pelvis",
            "is_rigid_body": True,
        },
    ]
    paths = resolve_body_prim_paths_from_records(
        records,
        ["pelvis", "left_ankle_roll_link"],
        default_path="/robot",
    )
    assert paths["pelvis"] == "Geometry/pelvis"
    assert paths["left_ankle_roll_link"] == "Geometry/pelvis/left_ankle_roll_link"
    assert (
        contact_sensor_prim_path("pelvis", paths)
        == "/World/envs/env_.*/Robot/Geometry/pelvis"
    )

    with pytest.raises(ValueError, match="visual_only"):
        resolve_body_prim_paths_from_records(
            records, ["visual_only"], default_path="/robot"
        )


def test_resolve_articulation_root_prim_path_nested_and_flat():
    records = [
        {
            "full_path": "/robot/Geometry/pelvis",
            "is_articulation_root": True,
        },
        {
            "full_path": "/robot/Geometry/pelvis/torso_link",
            "is_articulation_root": False,
        },
    ]
    assert (
        resolve_articulation_root_prim_path_from_records(
            records, default_path="/robot"
        )
        == "/Geometry/pelvis"
    )

    with pytest.raises(ValueError, match="exactly one"):
        resolve_articulation_root_prim_path_from_records([], default_path="/robot")


def test_resolve_articulation_root_prim_path_accepts_stage_factory():
    class FakeStage:
        default_path = "/robot"
        body_records = [
            {
                "full_path": "/robot",
                "is_articulation_root": True,
            }
        ]

    assert (
        resolve_articulation_root_prim_path(
            "ignored.usda", stage_factory=lambda _path: FakeStage()
        )
        == "/"
    )


def test_resolve_body_prim_paths_accepts_stage_factory():
    class FakeStage:
        default_path = "/robot"
        body_records = [
            {
                "name": "torso_link",
                "full_path": "/robot/Geometry/torso_link",
                "is_rigid_body": True,
            }
        ]

    paths = resolve_body_prim_paths(
        "ignored.usda",
        ["torso_link"],
        stage_factory=lambda _path: FakeStage(),
    )
    assert paths == {"torso_link": "Geometry/torso_link"}


def test_resolve_robot_prim_paths_inspects_stage_once():
    class FakeStage:
        default_path = "/robot"
        body_records = [
            {
                "name": "pelvis",
                "full_path": "/robot/Geometry/pelvis",
                "is_rigid_body": True,
                "is_articulation_root": True,
            }
        ]

    calls = []

    def stage_factory(path):
        calls.append(path)
        return FakeStage()

    articulation_root, body_paths = usd_body_paths.resolve_robot_prim_paths(
        "robot.usda",
        ["pelvis"],
        stage_factory=stage_factory,
    )

    assert calls == ["robot.usda"]
    assert articulation_root == "/Geometry/pelvis"
    assert body_paths == {"pelvis": "Geometry/pelvis"}


def test_scene_module_uses_mjcf_conversion_helpers():
    scene_path = (
        Path(__file__).parents[1] / "simulator" / "isaaclab" / "utils" / "scene.py"
    )
    tree = ast.parse(scene_path.read_text())
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "convert_robot_mjcf_to_usd" in imported
    assert "resolve_robot_prim_paths" in imported
    assert "contact_sensor_prim_path" in imported
    source = scene_path.read_text()
    assert "usd_asset_file_name" not in source
    assert "usd_bodies_root_prim_path" not in source


def test_simulator_leaves_mjcf_extension_lifecycle_to_converter():
    simulator_path = (
        Path(__file__).parents[1] / "simulator" / "isaaclab" / "simulator.py"
    )
    source = simulator_path.read_text()

    assert (
        'set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)'
        not in source
    )


def test_simulator_disables_failed_perspective_viewer(monkeypatch):
    import sys
    import types

    try:
        from protomotions.simulator.base_simulator.simulator import Simulator
        from protomotions.simulator.isaaclab.simulator import IsaacLabSimulator
    except ImportError as exc:
        pytest.skip(f"IsaacLab runtime is unavailable: {exc}")

    viewer_attempts = []
    base_renders = []

    class FailingPerspectiveViewer:
        def __init__(self):
            viewer_attempts.append(1)
            raise RuntimeError("viewer unavailable")

    viewer_module = types.ModuleType("perspective_viewer")
    viewer_module.PerspectiveViewer = FailingPerspectiveViewer
    monkeypatch.setitem(
        sys.modules,
        "protomotions.simulator.isaaclab.utils.perspective_viewer",
        viewer_module,
    )
    monkeypatch.setattr(Simulator, "render", lambda _self: base_renders.append(1))

    simulator = object.__new__(IsaacLabSimulator)
    simulator.headless = False
    simulator._perspective_view_failed = False

    simulator.render()
    simulator.render()

    assert len(viewer_attempts) == 1
    assert len(base_renders) == 2
    assert simulator._perspective_view_failed is True


def test_low_level_converter_exposes_output_directory_contract():
    repo_root = Path(__file__).parents[2]
    converter_source = (
        repo_root / "usd_convert" / "convert_mjcf_to_usd.py"
    ).read_text()
    wrapper_source = (
        repo_root / "usd_convert" / "convert_robot_mjcf_to_usda.py"
    ).read_text()

    converter_tree = ast.parse(converter_source)
    positional_args = [
        call.args[0].value
        for call in ast.walk(converter_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    ]

    assert "output_dir" in positional_args
    assert "args_cli.output_dir" in converter_source
    assert "os.path.dirname(dest_path)" not in converter_source
    assert '"_unused.usda"' not in wrapper_source


def test_public_install_metadata_targets_isaaclab3_stack():
    repo_root = Path(__file__).parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    requirements = pyproject["project"]["optional-dependencies"]["isaaclab"]

    assert any("isaaclab==12.0.0" in requirement for requirement in requirements)
    assert any("isaacsim[all,extscache]==6.0.0.1" in requirement for requirement in requirements)

    public_docs = [
        repo_root / "README.md",
        repo_root / "docs" / "source" / "index.rst",
        repo_root / "docs" / "source" / "getting_started" / "installation.rst",
    ]
    for path in public_docs:
        source = path.read_text()
        assert "IsaacLab-2.3.2" not in source
        assert "isaaclab[isaacsim,all]==2.3.2.post1" not in source
        assert "isaacsim[all]==5.1.0.0" not in source

    installation = public_docs[-1].read_text()
    assert "4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8" in installation
    assert "Python 3.12" in installation

    dependency_section = installation.split("IsaacLab as a dependency", 1)[1]
    dependency_section = dependency_section.split("Choose Your Simulator(s)", 1)[0]
    assert 'requires-python = "==3.11.*"' not in dependency_section
    assert '"torch==2.7.0"' not in dependency_section
    assert "pytorch-cu128" not in dependency_section
