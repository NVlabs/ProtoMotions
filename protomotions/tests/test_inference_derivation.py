# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deriving the inference view (``inference_utils``)."""

import argparse
from types import SimpleNamespace

import pytest

from protomotions.utils import inference_utils
from protomotions.utils.experiment_configs import ExperimentConfigs


def _recording_configs(events):
    """An ExperimentConfigs whose configs record when their defaults run.

    The recording functions are shared across the deep copy ``derive`` makes
    (deepcopy treats functions as atomic), so they still append to ``events``.
    """

    def default(name):
        def _default(*_args):
            events.append(f"default:{name}")

        return _default

    env = SimpleNamespace(
        sensors=None, save_dir="train", apply_inference_defaults=default("env")
    )
    simulator = SimpleNamespace(apply_inference_defaults=default("simulator"))
    motion_lib = SimpleNamespace(
        apply_inference_defaults=default("motion_lib"),
        validate=lambda: events.append("validate"),
    )
    agent = SimpleNamespace(apply_inference_defaults=default("agent"))
    return ExperimentConfigs(
        robot=SimpleNamespace(),
        simulator=simulator,
        terrain=SimpleNamespace(),
        scene_lib=SimpleNamespace(),
        motion_lib=motion_lib,
        env=env,
        agent=agent,
    )


def test_inference_defaults_runs_each_config_in_a_fixed_order():
    events = []
    inference_utils.inference_defaults(_recording_configs(events))
    assert events == [
        "default:env",
        "default:simulator",
        "default:motion_lib",
        "default:agent",
    ]


def test_inference_defaults_switches_present_sensors_to_their_clean_read():
    class _Sensors:
        def for_inference(self):
            return "exact"

    configs = _recording_configs([])
    configs.env.sensors = _Sensors()
    inference_utils.inference_defaults(configs)
    assert configs.env.sensors == "exact"


def test_derive_runs_defaults_then_overrides_then_hooks_then_checks(monkeypatch):
    events = []
    monkeypatch.setattr(
        inference_utils,
        "apply_overrides",
        lambda configs, overrides: events.append("overrides") or {},
    )
    monkeypatch.setattr(
        "protomotions.envs.context_check.run_config_binding_check",
        lambda *a, **k: events.append("binding_check"),
    )

    def hook(view, args):
        events.append("hook")

    inference_utils.derive_inference_configs(
        _recording_configs(events), argparse.Namespace(), {}, [hook]
    )
    assert events == [
        "default:env",
        "default:simulator",
        "default:motion_lib",
        "default:agent",
        "overrides",
        "hook",
        "validate",
        "binding_check",
    ]


def test_derive_refuses_a_hook_that_undoes_an_override(monkeypatch):
    monkeypatch.setattr(
        "protomotions.envs.context_check.run_config_binding_check", lambda *a, **k: None
    )

    def undoing_hook(view, args):
        view.env.save_dir = "hook"

    with pytest.raises(
        ValueError,
        match=r"undid override env\.save_dir='cli': the value is now 'hook'",
    ):
        inference_utils.derive_inference_configs(
            _recording_configs([]),
            argparse.Namespace(),
            {"env.save_dir": "cli"},
            [undoing_hook],
        )


def test_recipe_hooks_rejects_the_eight_parameter_form(tmp_path):
    (tmp_path / "experiment_config.py").write_text(
        "def apply_inference_overrides(a, b, c, d, e, f, g, h):\n    pass\n"
    )
    with pytest.raises(
        TypeError,
        match=r"must take \(configs, args\); the eight-parameter form was retired",
    ):
        inference_utils.recipe_hooks(tmp_path, None)


def test_recipe_hooks_requires_an_experiment_config_py(tmp_path):
    with pytest.raises(
        FileNotFoundError, match="no experiment_config.py beside the checkpoint"
    ):
        inference_utils.recipe_hooks(tmp_path, None)
