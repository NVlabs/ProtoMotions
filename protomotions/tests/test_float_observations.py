# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Float-only observations stay eager without changing their export contract."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.cast import to_float
from protomotions.utils.export_utils import ObservationExportModule


def test_masked_mimic_float_observations_are_eager():
    from examples.experiments.masked_mimic.transformer import env_config
    from protomotions.robot_configs.g1 import G1RobotConfig

    config = env_config(G1RobotConfig(), SimpleNamespace(expert_model_path=None))
    casts = [
        component
        for component in config.observation_components.values()
        if component.compute_func is to_float
    ]
    assert len(casts) == 2
    assert all(
        getattr(component.compute_func, "run_eagerly", False) for component in casts
    )


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float64])
def test_eager_cast_runtime_and_onnx_export(shape, dtype, monkeypatch, tmp_path):
    onnx = pytest.importorskip("onnx")
    x = torch.ones(shape, dtype=dtype)
    component = MdpComponent(
        compute_func=to_float,
        dynamic_vars={"x": EnvContext.current.dof_pos},
    )

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Float-only observations must not invoke torch.compile")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    manager = ComponentManager(torch.device("cpu"))
    context = SimpleNamespace(current=SimpleNamespace(dof_pos=x))
    actual = manager.execute_all({"cast": component}, context)["cast"]
    torch.testing.assert_close(actual, x.float())

    module = ObservationExportModule({"cast": component}, context, torch.device("cpu"))
    path = tmp_path / "cast.onnx"
    torch.onnx.export(module, (x,), str(path), dynamo=False)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    assert any(node.op_type == "Cast" for node in model.graph.node)
