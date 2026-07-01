# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for config parsing and override utilities."""

import sys
import types
from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from protomotions.utils.config_utils import (
    apply_config_overrides,
    clean_dict_for_storage,
    import_experiment_relative_eval_overrides,
    load_resolved_configs_from_checkpoint,
    make_json_serializable,
    parse_cli_overrides,
)


class _Mode(Enum):
    FAST = "fast"
    SLOW = "slow"

    @classmethod
    def from_str(cls, value: str):
        return cls(value.lower())


class _PlainMode(Enum):
    FAST = "fast"
    SLOW = "slow"


class _BadStringKey:
    def __str__(self):
        raise RuntimeError("cannot stringify key")


class _BadItemsDict(dict):
    def items(self):
        raise RuntimeError("cannot iterate items")


class _BadNameObject:
    def __getattribute__(self, name):
        if name == "__name__":
            raise RuntimeError("cannot read name")
        return super().__getattribute__(name)


def test_parse_cli_overrides_coerces_literal_values_and_keeps_strings():
    overrides = parse_cli_overrides(
        [
            "env.max_episode_length=1000",
            "simulator.headless=True",
            "robot.asset=None",
            "agent.name=gpc_prior",
            "terrain.scale=0.05",
        ]
    )

    assert overrides == {
        "env.max_episode_length": 1000,
        "simulator.headless": True,
        "robot.asset": None,
        "agent.name": "gpc_prior",
        "terrain.scale": 0.05,
    }


def test_parse_cli_overrides_rejects_missing_separator_and_empty_key():
    for value in ["env.max_episode_length", "=123"]:
        try:
            parse_cli_overrides([value])
        except ValueError as error:
            assert "Invalid override" in str(error)
        else:
            raise AssertionError(f"Expected invalid override to fail: {value}")


def test_apply_config_overrides_updates_nested_objects_dicts_and_enums():
    env = SimpleNamespace(
        max_episode_length=100,
        reward_components={"tracking": {"weight": 1.0}},
        mode=_Mode.FAST,
        terrain_proportions=[0.0, 1.0],
    )
    simulator = SimpleNamespace(headless=False)
    robot = SimpleNamespace(name="smpl")
    agent = SimpleNamespace(batch_size=32)

    apply_config_overrides(
        {
            "env.max_episode_length": 200,
            "env.reward_components.tracking.weight": 2.5,
            "env.mode": "slow",
            "env.terrain_proportions": [0.2, 0.8],
            "simulator.headless": True,
            "robot.name": "g1",
            "agent.batch_size": 64,
        },
        env_config=env,
        simulator_config=simulator,
        robot_config=robot,
        agent_config=agent,
    )

    assert env.max_episode_length == 200
    assert env.reward_components["tracking"]["weight"] == 2.5
    assert env.mode is _Mode.SLOW
    assert env.terrain_proportions == [0.2, 0.8]
    assert simulator.headless is True
    assert robot.name == "g1"
    assert agent.batch_size == 64


def test_apply_config_overrides_rejects_unknown_paths_and_unsupported_values():
    env = SimpleNamespace(nested=SimpleNamespace(value=1), unsupported=object())
    simulator = SimpleNamespace()
    robot = SimpleNamespace()

    try:
        apply_config_overrides(
            {"env.missing.value": 1},
            env_config=env,
            simulator_config=simulator,
            robot_config=robot,
        )
    except ValueError as error:
        assert "Field 'missing' not found" in str(error)
    else:
        raise AssertionError("Expected unknown override path to fail")

    try:
        apply_config_overrides(
            {"env.unsupported": [1, 2, 3]},
            env_config=env,
            simulator_config=simulator,
            robot_config=robot,
        )
    except ValueError as error:
        assert "unsupported type" in str(error)
    else:
        raise AssertionError("Expected unsupported field type to fail")

    try:
        apply_config_overrides(
            {"env.nested.missing": 1},
            env_config=env,
            simulator_config=simulator,
            robot_config=robot,
        )
    except ValueError as error:
        assert "Field 'missing' not found" in str(error)
    else:
        raise AssertionError("Expected missing final object field to fail")

    try:
        apply_config_overrides(
            {"env.storage.values": 1},
            env_config=SimpleNamespace(storage={"values": object()}),
            simulator_config=simulator,
            robot_config=robot,
        )
    except ValueError as error:
        assert "unsupported type" in str(error)
    else:
        raise AssertionError("Expected unsupported dict field type to fail")


def test_apply_config_overrides_handles_optional_configs_and_error_branches():
    env = SimpleNamespace()
    simulator = SimpleNamespace()
    robot = SimpleNamespace()
    terrain = SimpleNamespace(scale=1.0)
    motion_lib = SimpleNamespace(motion_file=None)
    scene_lib = {"sampling": {"mode": "all"}}
    agent = SimpleNamespace(mode=_PlainMode.FAST)

    apply_config_overrides({}, env, simulator, robot)
    apply_config_overrides(
        {
            "terrain.scale": 0.25,
            "motion_lib.motion_file": "motions.yaml",
            "scene_lib.sampling.mode": "random",
            "agent.mode": "slow",
        },
        env_config=env,
        simulator_config=simulator,
        robot_config=robot,
        agent_config=agent,
        terrain_config=terrain,
        motion_lib_config=motion_lib,
        scene_lib_config=scene_lib,
    )

    assert terrain.scale == 0.25
    assert motion_lib.motion_file == "motions.yaml"
    assert scene_lib["sampling"]["mode"] == "random"
    assert agent.mode is _PlainMode.SLOW

    for overrides, expected in [
        ({"env": 1}, "Invalid override key format"),
        ({"unknown.value": 1}, "Unknown config type"),
        ({"agent.value": 1}, "agent_config not provided"),
        ({"terrain.value": 1}, "terrain_config not provided"),
        ({"motion_lib.value": 1}, "motion_lib_config not provided"),
        ({"scene_lib.value": 1}, "scene_lib_config not provided"),
        ({"scene_lib.missing.value": 1}, "Key 'missing' not found"),
        ({"scene_lib.sampling.missing": 1}, "Key 'missing' not found"),
        ({"agent.mode": "invalid"}, "Invalid value 'invalid'"),
    ]:
        try:
            apply_config_overrides(
                overrides,
                env_config=env,
                simulator_config=simulator,
                robot_config=robot,
                agent_config=agent if "invalid" in expected else None,
                scene_lib_config=scene_lib if "Key" in expected else None,
            )
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"Expected override {overrides} to fail")


def test_load_resolved_configs_from_checkpoint_loads_sibling_file(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model.pt"
    checkpoint.write_bytes(b"unused")
    torch.save({"env": {"num_envs": 2}}, run_dir / "resolved_configs.pt")

    loaded = load_resolved_configs_from_checkpoint(str(checkpoint))

    assert loaded == {"env": {"num_envs": 2}}

    with pytest.raises(FileNotFoundError, match="Resolved configs not found"):
        load_resolved_configs_from_checkpoint(str(tmp_path / "missing" / "model.pt"))


def test_load_resolved_configs_from_checkpoint_names_missing_pickle_module(tmp_path):
    module_name = "_missing_resolved_config_module"
    module = types.ModuleType(module_name)
    exec("class MissingConfig:\n    pass\n", module.__dict__)
    sys.modules[module_name] = module

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model.pt"
    checkpoint.write_bytes(b"unused")
    try:
        torch.save({"env": module.MissingConfig()}, run_dir / "resolved_configs.pt")
    finally:
        sys.modules.pop(module_name, None)

    with pytest.raises(ModuleNotFoundError, match="resolved_configs.pt.*missing module"):
        load_resolved_configs_from_checkpoint(str(checkpoint))


def test_import_experiment_relative_eval_overrides_loads_and_validates_modules(
    monkeypatch,
    tmp_path,
):
    import importlib.util
    import inspect

    caller = tmp_path / "caller.py"
    caller.write_text("# caller")
    experiment = tmp_path / "experiment.py"
    experiment.write_text(
        "def apply_inference_overrides(*args):\n"
        "    return ('overrides', args)\n"
    )
    monkeypatch.setattr(
        inspect,
        "stack",
        lambda: [SimpleNamespace(filename="inside.py"), SimpleNamespace(filename=str(caller))],
    )

    overrides = import_experiment_relative_eval_overrides("experiment.py")

    assert overrides(1, 2) == ("overrides", (1, 2))

    missing_hook = tmp_path / "missing_hook.py"
    missing_hook.write_text("VALUE = 1\n")
    with pytest.raises(AttributeError, match="apply_inference_overrides"):
        import_experiment_relative_eval_overrides("missing_hook.py")

    with pytest.raises(FileNotFoundError, match="Experiment module not found"):
        import_experiment_relative_eval_overrides("does_not_exist.py")

    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *args: None)
    with pytest.raises(ImportError, match="Failed to load module spec"):
        import_experiment_relative_eval_overrides("experiment.py")


def test_clean_dict_for_storage_converts_common_non_yaml_scalars():
    from protomotions.envs.mdp_component import MdpComponent

    def helper():
        return None

    def compute_value():
        return torch.tensor(1.0)

    data = {
        "tensor": torch.tensor([1, 2]),
        "array": np.array([3, 4]),
        "enum": _Mode.FAST,
        "callable": helper,
        "nested": [{"tensor": torch.tensor([5])}],
        "component": MdpComponent(
            compute_func=compute_value,
            dynamic_vars={},
            static_params={"weight": torch.tensor([0.5])},
        ),
    }

    clean = clean_dict_for_storage(data)

    assert clean["tensor"] == [1, 2]
    assert clean["array"] == [3, 4]
    assert clean["enum"] == "fast"
    assert clean["callable"] == "helper"
    assert clean["nested"][0]["tensor"] == [5]
    assert clean["component"]["compute_func"] == "compute_value"
    assert clean["component"]["weight"] == [0.5]


def test_make_json_serializable_handles_unknown_objects_and_depth_limit():
    assert make_json_serializable(None) is None
    assert make_json_serializable(True) is True
    assert make_json_serializable(3) == 3
    assert make_json_serializable({"already": "serializable"}) == {
        "already": "serializable"
    }
    assert make_json_serializable(lambda: None) == "<<lambda>>"

    obj = SimpleNamespace(value=object())

    result = make_json_serializable(
        {"obj": obj, "items": [object()], "nested": {"a": {"b": {"c": object()}}}},
        max_depth=2,
    )

    assert result["obj"] == "<SimpleNamespace>"
    assert result["items"] == ["<object>"]
    assert result["nested"]["a"]["b"] == "<max_depth_reached>"


def test_make_json_serializable_handles_hostile_dict_keys_and_tuple_items():
    result = make_json_serializable(
        {
            _BadStringKey(): object(),
            ("tuple", "key"): (object(), torch.tensor([1.0]), np.array([2.0])),
        }
    )

    assert result["<non-serializable-key: _BadStringKey>"] == "<object>"
    assert result["('tuple', 'key')"] == ("<object>", "<Tensor>", "<ndarray>")
    import json

    json.dumps(result)


def test_make_json_serializable_falls_back_when_nested_values_raise(monkeypatch):
    import json

    def fail_json_dumps(_obj):
        raise TypeError("force recursive conversion")

    monkeypatch.setattr(json, "dumps", fail_json_dumps)

    result = make_json_serializable(
        {
            "bad_dict": _BadItemsDict({"nested": object()}),
            "bad_list": [_BadItemsDict({"nested": object()})],
            "bad_name": _BadNameObject(),
        }
    )

    assert result["bad_dict"] == "<non-serializable: _BadItemsDict>"
    assert result["bad_list"] == "<non-serializable list/tuple of list>"
    assert result["bad_name"] == "<non-serializable>"
