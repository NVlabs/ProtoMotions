# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton partner-resolved contact matrix against the real Newton API.

Index bookkeeping without Newton lives in test_newton_contact_body_labels.py.
The two-world checks run with the production solver on CPU Warp; the G1
simulator smoke additionally needs CUDA for the full robot model.
"""

import pytest
import torch

from protomotions.simulator.base_simulator.contacts import ContactLayout
from protomotions.simulator.newton.contact_utils import (
    contact_force_matrix,
    resolve_world_body_indices,
    resolve_world_projectile_indices,
    shape_axes,
)

ALL_BODIES = ("left", "right")  # the robot's full common body list


def _layout(rows, partners):
    """Selected rows/partners; the layout keeps canonical order on both axes."""
    return ContactLayout.build(
        ALL_BODIES,
        num_objects=0,
        num_projectiles=1,
        contact_bodies=rows if isinstance(rows, str) else list(rows),
        contact_partners=partners if isinstance(partners, str) else list(partners),
    )


def _two_world_model(newton, wp):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    left = builder.add_body(
        xform=wp.transform((0, 0, 0.09), wp.quat_identity()), label="rob/wb/left"
    )
    builder.add_shape_sphere(left, radius=0.1)
    right = builder.add_body(
        xform=wp.transform((1, 0, 0.09), wp.quat_identity()), label="rob/wb/right"
    )
    builder.add_shape_sphere(right, radius=0.1)
    # Two collision shapes on one partner body, plus a visual-only shape.
    builder.add_shape_box(right, hx=0.05, hy=0.05, hz=0.05)
    visual = newton.ModelBuilder.ShapeConfig()
    visual.flags = 0
    builder.add_shape_sphere(right, radius=0.01, cfg=visual)
    projectile = builder.add_body(xform=wp.transform((3, 0, 1.0), wp.quat_identity()))
    builder.add_shape_box(projectile, hx=0.1, hy=0.1, hz=0.1)
    scene = newton.ModelBuilder()
    scene.replicate(builder, 2)
    scene.add_ground_plane()
    model = scene.finalize()
    model.set_gravity((0.0, 0.0, -9.81))
    return model


def _axes(newton, model, layout, num_projectiles=1):
    """Mirror NewtonSimulator._setup_contact_matrix_maps on a bare model."""
    body_world = model.body_world.numpy().tolist()
    robot = resolve_world_body_indices(model.body_label, body_world, 2, ALL_BODIES)
    projectiles = resolve_world_projectile_indices(
        body_world, 2, robot, num_projectiles
    )
    num_rows = len(layout.body_index)
    row_of_body, column_of_body = {}, {}
    for world in range(2):
        by_name = dict(zip(ALL_BODIES, robot[world]))
        for name, row in layout.body_index.items():
            row_of_body[by_name[name]] = world * num_rows + row
        for name, index in by_name.items():
            column = layout.partner_index.get(f"robot/{name}")
            if column is not None:
                column_of_body[index] = column
        for slot, index in enumerate(projectiles[world]):
            column = layout.partner_index.get(f"projectile/{slot}")
            if column is not None:
                column_of_body[index] = column
    return shape_axes(
        model.shape_body.numpy().tolist(),
        model.shape_world.numpy().tolist(),
        model.shape_flags.numpy().tolist(),
        int(newton.ShapeFlags.COLLIDE_SHAPES),
        row_of_body,
        column_of_body,
        layout.partner_index.get("terrain"),
    )


def test_shape_axes_on_a_real_model_follow_layout_and_skip_visual_shapes():
    newton = pytest.importorskip("newton")
    import warp as wp

    model = _two_world_model(newton, wp)
    # Shapes per world: left sphere, right sphere, right box, right visual,
    # projectile box; then the global ground plane.
    layout = _layout(ALL_BODIES, "all")
    assert layout.is_complete
    rows, columns = _axes(newton, model, layout)
    assert rows.tolist() == [0, 1, 1, -1, -1, 2, 3, 3, -1, -1, -1]
    assert columns.tolist() == [1, 2, 2, -1, 3, 1, 2, 2, -1, 3, 0]
    # Row and partner subsets: one row (left), robot/right only -> no ground.
    rows, columns = _axes(newton, model, _layout(("left",), ("robot/right",)))
    assert rows.tolist() == [0, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1]
    assert columns.tolist() == [-1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1]
    # Terrain-only partners: exactly one usable column, the ground plane.
    _, columns = _axes(newton, model, _layout(ALL_BODIES, ("terrain",)))
    assert columns.tolist() == [-1] * 10 + [0]


def _step_mujoco(newton, model, steps=120, dt=1.0 / 240.0):
    """Step the production solver (SolverMuJoCo) and refresh per-contact forces."""
    model.request_contact_attributes("force")
    state_0, state_1, control = model.state(), model.state(), model.control()
    contacts = model.contacts()
    solver = newton.solvers.SolverMuJoCo(model)
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, dt=dt)
        state_0, state_1 = state_1, state_0
    solver.update_contacts(contacts, state_0)
    return contacts


def test_newton_mujoco_ground_forces_land_in_terrain_column_in_env_body_order():
    """Production solver on CPU Warp (the pinned Newton runtime builds it)."""
    newton = pytest.importorskip("newton")
    import warp as wp

    model = _two_world_model(newton, wp)
    # Robot/right is deliberately unselected; terrain is not column 0 by name.
    layout = _layout(ALL_BODIES, ("robot/left", "terrain", "projectile/0"))
    assert not layout.is_complete
    shape_row, shape_column = _axes(newton, model, layout)
    contacts = _step_mujoco(newton, model)
    matrix = contact_force_matrix(
        wp.to_torch(contacts.rigid_contact_count).cpu(),
        wp.to_torch(contacts.rigid_contact_shape0).cpu(),
        wp.to_torch(contacts.rigid_contact_shape1).cpu(),
        wp.to_torch(contacts.force).cpu(),
        wp.to_torch(contacts.rigid_contact_normal).cpu(),
        shape_row,
        shape_column,
        2,
        2,
        len(layout.partner_index),
    )
    mass = torch.as_tensor(model.body_mass.numpy())
    weight = torch.stack([mass[[0, 1]], mass[[3, 4]]]) * 9.81
    terrain_column = layout.partner_index["terrain"]
    terrain = matrix[:, :, terrain_column]
    torch.testing.assert_close(terrain[..., 2], weight, atol=1.0, rtol=0.02)
    torch.testing.assert_close(
        terrain[..., :2], torch.zeros(2, 2, 2), atol=1e-3, rtol=0
    )
    # Nothing touches the selected robot link or the airborne projectile.
    others = [c for c in range(len(layout.partner_index)) if c != terrain_column]
    other = matrix[:, :, others]
    torch.testing.assert_close(other, torch.zeros_like(other))


@pytest.mark.needs_lfs
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SolverMuJoCo needs CUDA Warp"
)
def test_newton_simulator_matrix_separates_grounded_and_airborne_g1():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from protomotions.components.scene_lib import SceneLib
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.components.terrains.terrain import Terrain
    from protomotions.robot_configs.factory import robot_config
    from protomotions.simulator.newton.config import (
        NewtonSimParams,
        NewtonSimulatorConfig,
    )
    from protomotions.simulator.newton.simulator import NewtonSimulator

    device = torch.device("cuda")
    robot_cfg = robot_config("g1")
    feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
    # Include robot partners explicitly for the antisymmetry check below.
    robot_cfg.update_fields(contact_bodies=feet, contact_partners="all")
    terrain = Terrain(
        config=TerrainConfig(
            map_length=20.0,
            map_width=20.0,
            border_size=40.0,
            num_levels=1,
            num_terrains=1,
            terrain_proportions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            horizontal_scale=0.1,
            vertical_scale=0.005,
        ),
        num_envs=2,
        device=device,
    )
    simulator = NewtonSimulator(
        config=NewtonSimulatorConfig(
            num_envs=2,
            headless=True,
            sim=NewtonSimParams(fps=200),  # multiple of g1 control_hz=50
            experiment_name="newton_contact_matrix_test",
        ),
        robot_config=robot_cfg,
        terrain=terrain,
        device=device,
        scene_lib=SceneLib.empty(num_envs=2, device=str(device), terrain=terrain),
    )
    simulator._initialize_with_markers({})
    layout = simulator.contact_layout
    rows, partners = len(layout.body_index), len(layout.partner_index)
    assert simulator.get_contact_forces().shape == (2, rows, partners, 3)

    # Env 0 starts on the ground; env 1 is dropped from 5 m and is still falling
    # after half a second of zero-action stepping.
    state = simulator.get_default_robot_reset_state()
    state.root_pos[1, 2] += 5.0
    simulator.reset_envs(state, env_ids=torch.arange(2, device=device))
    actions = torch.zeros(2, robot_cfg.number_of_actions, device=device)
    for _ in range(15):
        simulator.step(actions)
    matrix = simulator.get_contact_forces()
    assert torch.isfinite(matrix).all()
    terrain_norm = matrix[:, :, layout.partner_index["terrain"]].norm(dim=-1)
    assert terrain_norm[0].max() > 1.0, terrain_norm[0]
    torch.testing.assert_close(matrix[1], torch.zeros_like(matrix[1]))
    # Robot/robot antisymmetry over the selected rows: force on i from j equals
    # minus the force on j from i whenever both bodies are rows and partners.
    for i, name_i in enumerate(layout.body_index):
        for j, name_j in enumerate(layout.body_index):
            col_i = layout.partner_index.get(f"robot/{name_i}")
            col_j = layout.partner_index.get(f"robot/{name_j}")
            if col_i is None or col_j is None:
                continue
            torch.testing.assert_close(
                matrix[:, i, col_j], -matrix[:, j, col_i], atol=0.05, rtol=1e-4
            )
