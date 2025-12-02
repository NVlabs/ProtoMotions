# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
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
#
"""
Simple empirical test to determine IsaacGym's friction combine mode.

Test setup:
- Box on flat ground
- Apply horizontal force and measure if/when it slides
- Compare with theoretical predictions for different combine modes

Friction coefficients:
- Ground: 0.4
- Box: 1.6

Expected effective friction for different modes:
- Average: (0.4 + 1.6) / 2 = 1.0
- Min: min(0.4, 1.6) = 0.4
- Max: max(0.4, 1.6) = 1.6
- Multiply: 0.4 * 1.6 = 0.64

Test: Apply horizontal force = weight * mu, see if it slides
"""

# IMPORTANT: Import isaacgym before torch!
from isaacgym import gymapi, gymtorch
import torch
import numpy as np


def test_friction_combine_mode():
    print("=" * 80)
    print("Testing IsaacGym Friction Combine Mode")
    print("=" * 80)

    # Initialize gym
    gym = gymapi.acquire_gym()

    # Create sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60
    sim_params.substeps = 2
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = True
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    # Create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    cam_pos = gymapi.Vec3(4, 3, 2)
    cam_target = gymapi.Vec3(0, 0, 0.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # Create ground plane with friction = 0.4
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.static_friction = 0.4
    plane_params.dynamic_friction = 0.4
    plane_params.restitution = 0.0
    gym.add_ground(sim, plane_params)

    # Create box asset with friction = 1.6
    asset_options = gymapi.AssetOptions()
    asset_options.density = 1000.0
    asset_options.fix_base_link = False

    box_asset = gym.create_box(sim, 0.5, 0.5, 0.5, asset_options)

    # Set box friction to 1.6
    shape_props = gym.get_asset_rigid_shape_properties(box_asset)
    for prop in shape_props:
        prop.friction = 1.6
    gym.set_asset_rigid_shape_properties(box_asset, shape_props)

    # Create environments to test different force levels
    num_envs = 4
    envs = []
    box_handles = []

    spacing = 2.0
    lower = gymapi.Vec3(-spacing, 0.0, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)

    # Test forces as multiples of weight: 0.3, 0.6, 1.2, 1.8 times weight
    force_multipliers = [0.3, 0.6, 1.2, 1.8]

    for i in range(num_envs):
        env = gym.create_env(sim, lower, upper, int(np.sqrt(num_envs)))
        envs.append(env)

        # Create box actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 1.0)  # Start 1m above ground
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        box_handle = gym.create_actor(env, box_asset, pose, f"box_{i}", i, 0)
        box_handles.append(box_handle)

    # Prepare sim
    gym.prepare_sim(sim)

    # Get tensors
    gym.refresh_actor_root_state_tensor(sim)
    actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(actor_root_state)

    # Let boxes settle
    print("\nLetting boxes settle on ground...")
    for _ in range(120):  # 2 seconds
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.refresh_actor_root_state_tensor(sim)

    # Record initial positions
    initial_positions = root_states[:, 0:3].clone()

    print("\nInitial box positions (x, y, z):")
    for i in range(num_envs):
        print(f"  Box {i}: {initial_positions[i].cpu().numpy()}")

    # Apply horizontal forces
    print("\nApplying horizontal forces...")
    print(
        f"Box mass ≈ {0.5 * 0.5 * 0.5 * 1000.0} kg, Weight ≈ {0.5 * 0.5 * 0.5 * 1000.0 * 9.81} N"
    )
    print("Watch the visualization - boxes will slide if force > friction threshold")
    print("Press ESC or close window to end test early\n")

    weight = 0.5 * 0.5 * 0.5 * 1000.0 * 9.81  # mass * g

    # Apply forces for 2 seconds with visualization
    for step in range(120):
        # Check for viewer close
        if gym.query_viewer_has_closed(viewer):
            break

        forces = torch.zeros((num_envs, 3), dtype=torch.float32, device="cuda:0")

        for i in range(num_envs):
            # Apply horizontal force = weight * multiplier
            forces[i, 0] = weight * force_multipliers[i]

        forces_tensor = gymtorch.unwrap_tensor(forces)
        gym.apply_rigid_body_force_tensors(sim, forces_tensor, None, gymapi.LOCAL_SPACE)

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)

        # Render
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    # Check final positions
    final_positions = root_states[:, 0:3].clone()
    displacements = (final_positions - initial_positions)[:, 0]  # X displacement

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print("Ground friction: 0.4, Box friction: 1.6")
    print()

    for i in range(num_envs):
        mu_required = force_multipliers[i]
        displacement = displacements[i].item()
        slid = abs(displacement) > 0.01  # 1cm threshold

        print(f"Env {i}: Force = {mu_required:.1f} × weight")
        print(f"  Displacement: {displacement:.4f} m")
        print("  Slid: " + ("YES" if slid else "NO"))
        print()

    print("Expected sliding behavior for different combine modes:")
    print("  Average (mu=1.0): Slide when force > 1.0×weight → Envs 2,3 slide")
    print("  Min (mu=0.4):     Slide when force > 0.4×weight → Envs 1,2,3 slide")
    print("  Max (mu=1.6):     Slide when force > 1.6×weight → Only env 3 slides")
    print("  Multiply (mu=0.64): Slide when force > 0.64×weight → Envs 2,3 slide")
    print()

    # Determine combine mode
    slid_flags = [abs(displacements[i].item()) > 0.01 for i in range(num_envs)]

    if slid_flags == [False, False, True, True]:
        print("CONCLUSION: Friction combine mode is AVERAGE (mu_eff = 1.0)")
    elif slid_flags == [False, True, True, True]:
        print("CONCLUSION: Friction combine mode is MIN (mu_eff = 0.4)")
    elif slid_flags == [False, False, False, True]:
        print("CONCLUSION: Friction combine mode is MAX (mu_eff = 1.6)")
    else:
        print(f"CONCLUSION: Unexpected behavior - slid_flags = {slid_flags}")
        print(
            "This could indicate MULTIPLY mode or friction is not behaving as expected"
        )

    print("=" * 80)

    # Keep viewer open to see final positions
    print("\nViewer will stay open for 5 seconds to inspect final positions...")
    print("(Or close window to exit)")

    for _ in range(300):  # 5 seconds
        if gym.query_viewer_has_closed(viewer):
            break
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    # Cleanup
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    test_friction_combine_mode()
