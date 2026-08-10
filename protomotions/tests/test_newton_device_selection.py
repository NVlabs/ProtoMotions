# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


NEWTON_SIMULATOR = (
    Path(__file__).resolve().parents[1] / "simulator" / "newton" / "simulator.py"
)


def test_warp_device_is_set_before_newton_simulation_allocation():
    tree = ast.parse(NEWTON_SIMULATOR.read_text(), filename=str(NEWTON_SIMULATOR))
    simulator_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NewtonSimulator"
    )
    create_simulation = next(
        node
        for node in simulator_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_create_simulation"
    )

    first_statement = create_simulation.body[1]
    assert isinstance(first_statement, ast.Expr)
    assert isinstance(first_statement.value, ast.Call)

    set_device_call = first_statement.value
    assert isinstance(set_device_call.func, ast.Attribute)
    assert isinstance(set_device_call.func.value, ast.Name)
    assert set_device_call.func.value.id == "wp"
    assert set_device_call.func.attr == "set_device"
    assert len(set_device_call.args) == 1

    device_arg = set_device_call.args[0]
    assert isinstance(device_arg, ast.Call)
    assert isinstance(device_arg.func, ast.Name)
    assert device_arg.func.id == "str"
    assert len(device_arg.args) == 1
    torch_device = device_arg.args[0]
    assert isinstance(torch_device, ast.Attribute)
    assert isinstance(torch_device.value, ast.Name)
    assert torch_device.value.id == "self"
    assert torch_device.attr == "device"
