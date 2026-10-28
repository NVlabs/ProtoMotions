# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exported deployment metadata names the anchor body consistently.

The anchor has one source, ``anchor_body_name``; ``anchor_body_index`` and the
exported element labels derive from it, so the sidecar always names the body the
policy is anchored on. Deployment hosts (RoboJuDo) select the anchor body by that
name.
"""

import argparse
from types import SimpleNamespace

import pytest
import torch
import yaml


def _g1_tracker_configs():
    from examples.experiments.mimic import mlp_bm_l2c2 as recipe
    from protomotions.robot_configs.factory import robot_config

    robot = robot_config("g1")
    args = argparse.Namespace(
        motion_file="motions.pt",
        scenes_file=None,
        batch_size=32,
        training_max_steps=1024,
    )
    recipe.configure_robot_and_simulator(robot, SimpleNamespace(), args)
    assert robot.anchor_body_name == "torso_link"
    # Move the anchor to the pelvis by name; the index and exported name follow.
    robot.update_fields(anchor_body_name="pelvis")
    assert robot.anchor_body_index == 0
    env = recipe.env_config(robot, args)
    agent = recipe.agent_config(robot, env, args)
    return robot, env, agent


def _keep_initial_actor_weights(monkeypatch):
    """Let an empty checkpoint load: the actor keeps the weights it was built with."""
    load_state_dict = torch.nn.Module.load_state_dict

    def load(self, state_dict, *args, **kwargs):
        if not state_dict:
            state_dict = self.state_dict()
        return load_state_dict(self, state_dict, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "load_state_dict", load)


def test_tracker_export_names_the_anchor_body_training_indexed(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from deployment import export_to_onnx

    robot, env, agent = _g1_tracker_configs()
    # The exporter reads robot/env/agent/simulator off the one (training) bundle;
    # the other roots go unread here, so None is fine.
    from protomotions.utils.experiment_configs import (
        ExperimentConfigs,
        save_experiment_configs,
    )

    configs = ExperimentConfigs(
        robot=robot,
        simulator=None,
        terrain=None,
        scene_lib=None,
        motion_lib=None,
        env=env,
        agent=agent,
    )
    save_experiment_configs(configs, tmp_path)
    torch.save({"model": {}}, tmp_path / "last.ckpt")
    _keep_initial_actor_weights(monkeypatch)

    onnx_path = export_to_onnx.export_tracker(
        str(tmp_path / "last.ckpt"), str(tmp_path / "out")
    )

    sidecar = yaml.safe_load(onnx_path.with_suffix(".yaml").read_text())
    assert sidecar["robot"]["anchor_body_index"] == 0
    assert sidecar["robot"]["anchor_body_name"] == "pelvis"
    assert sidecar["sensors"]["anchor_body_name"] == "pelvis"
    # Apart from the full body list, nothing may still name the torso -- including
    # every input's element_names, which must have followed the anchor rename.
    sidecar.pop("body_names", None)
    sidecar["robot"].pop("body_names")
    assert "torso_link" not in yaml.safe_dump(sidecar)
