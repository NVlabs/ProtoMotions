# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
