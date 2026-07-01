# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from protomotions.components.scene_lib import ObjectOptions


def test_object_options_exposes_optional_material_properties():
    options = ObjectOptions(
        static_friction=1.2,
        dynamic_friction=0.9,
        restitution=0.05,
    )

    assert options.physics_material_kwargs() == {
        "static_friction": 1.2,
        "dynamic_friction": 0.9,
        "restitution": 0.05,
    }
    assert options.to_dict()["static_friction"] == 1.2


def test_object_options_omits_material_properties_by_default():
    options = ObjectOptions()

    assert options.physics_material_kwargs() == {}
    assert "static_friction" not in options.to_dict()


def test_object_options_single_friction_prefers_static_friction():
    options = ObjectOptions(static_friction=1.2, dynamic_friction=0.9)

    assert options.single_friction() == 1.2
    assert ObjectOptions(dynamic_friction=0.9).single_friction() == 0.9


def test_object_options_exposes_single_friction_backend_material_properties():
    options = ObjectOptions(
        static_friction=1.2,
        dynamic_friction=0.9,
        restitution=0.05,
    )

    assert options.single_friction_material_kwargs() == {
        "friction": 1.2,
        "restitution": 0.05,
    }
    assert ObjectOptions(dynamic_friction=0.9).single_friction_material_kwargs(
        friction_key="mu"
    ) == {"mu": 0.9}
