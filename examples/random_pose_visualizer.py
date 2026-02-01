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
Random Pose Visualizer for Humanoid Robots

This tool visualizes random humanoid poses by:
1. Loading a specified robot (e.g., g1, rigv1, smpl)
2. Generating random joint configurations within joint limits
3. Displaying the poses with visualization markers on key body parts
"""

from typing import Dict, List
import argparse
from dataclasses import dataclass
import math

# Parse arguments first (argparse is safe, doesn't import torch)
parser = argparse.ArgumentParser(
    description="Random Pose Visualizer for Humanoid Robots"
)
parser.add_argument(
    "--simulator",
    type=str,
    choices=["isaacgym", "isaaclab", "newton"],
    default="isaacgym",
    help="Simulator to use (isaacgym, isaaclab, newton)",
)
parser.add_argument(
    "--robot",
    type=str,
    choices=["g1", "rigv1", "smpl"],
    default="g1",
    help="Robot to load (g1, rigv1, or smpl)",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument(
    "--cpu-only",
    action="store_true",
    default=False,
    help="Use CPU only for simulation (experimental, GPU is default)",
)
args = parser.parse_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
import torch  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402

from protomotions.simulator.base_simulator.config import (  # noqa: E402
    VisualizationMarkerConfig,
    MarkerConfig,
    MarkerState,
)
from protomotions.simulator.factory import simulator_config  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402
from protomotions.utils.rotations import quat_from_euler_xyz  # noqa: E402


@dataclass
class RobotSpec:
    """Robot specification with body names for visualization"""

    # Body names to visualize (these are the rigid body names, not joint names)
    viz_bodies: List[str]


# Define robot specifications
ROBOT_SPECS = {
    "g1": RobotSpec(
        viz_bodies=[
            "pelvis",
            "torso_link",
            "left_knee_link",
            "right_knee_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ],
    ),
    "rigv1": RobotSpec(
        viz_bodies=["Hips", "Spine2", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot"],
    ),
    "smpl": RobotSpec(
        viz_bodies=["Pelvis", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"],
    ),
}


class RandomPoseVisualizer:
    def __init__(
        self,
        robot_name: str = "g1",
        num_envs: int = 1,
        simulator_type: str = "isaacgym",
        headless: bool = False,
        cpu_only: bool = False,
        extra_simulator_params: dict = None,
    ):
        self.robot_name = robot_name
        self.robot_spec = ROBOT_SPECS[robot_name]
        self.num_envs = num_envs
        self.simulator_type = simulator_type
        self.headless = headless
        self.device = torch.device("cuda:0" if not cpu_only else "cpu")

        # Load robot configuration using factory function
        self.robot_cfg = robot_config(robot_name)

        # Create simulator configuration using factory function
        self.simulator_cfg = simulator_config(
            simulator_type,
            self.robot_cfg,
            headless=headless,
            num_envs=num_envs,
            experiment_name="random_pose_viz",
        )

        # Override robot asset settings for pose visualization
        self.robot_cfg.asset.disable_gravity = True
        self.robot_cfg.asset.fix_base_link = False  # Allow free movement
        self.robot_cfg.asset.self_collisions = False  # Disable self-collisions

        # Use torque control (zero torque) to hold poses without movement
        from protomotions.robot_configs.base import ControlType

        self.robot_cfg.control.control_type = ControlType.TORQUE

        # Create visualization markers
        self.viz_markers = self._create_visualization_markers()

        # No terrain needed for pose visualization
        terrain = None

        # Create empty scene_lib (no scenes, no terrain needed)
        from protomotions.components.scene_lib import SceneLib

        scene_lib = SceneLib.empty(
            num_envs=self.simulator_cfg.num_envs, device=self.device
        )

        # Get simulator class and instantiate
        SimulatorClass = get_class(self.simulator_cfg._target_)

        extra_params = extra_simulator_params or {}
        self.simulator = SimulatorClass(
            config=self.simulator_cfg,
            robot_config=self.robot_cfg,
            terrain=terrain,
            device=self.device,
            scene_lib=scene_lib,
            **extra_params,
        )

        # Initialize the simulator with visualization markers
        self.simulator._initialize_with_markers(self.viz_markers)

        print(f"Loaded {robot_name} robot using {simulator_type}")
        print(f"Robot config: {type(self.robot_cfg).__name__}")
        print(f"Number of actions: {self.robot_cfg.number_of_actions}")
        print(f"Number of DOFs: {self.robot_cfg.kinematic_info.num_dofs}")
        print(f"Visualizing bodies: {self.robot_spec.viz_bodies}")
        print("Press 'R' to generate a new random pose")

        self.simulator.user_requested_reset = True

    def _create_visualization_markers(self) -> Dict[str, VisualizationMarkerConfig]:
        """Create visualization markers for specified body locations"""
        # Create one marker config for each body we want to visualize
        marker_configs = [
            MarkerConfig(size="regular") for _ in self.robot_spec.viz_bodies
        ]

        # Create a single visualization marker group for all bodies
        markers = {
            "body_markers": VisualizationMarkerConfig(
                type="sphere", color=(1.0, 0.0, 0.0), markers=marker_configs
            )
        }

        return markers

    def _gen_random_pose(self):
        """Generate a random pose within joint limits"""

        print("Generating new random pose")

        dof_limits_lower = self.robot_cfg.kinematic_info.dof_limits_lower.to(
            self.device
        )
        dof_limits_upper = self.robot_cfg.kinematic_info.dof_limits_upper.to(
            self.device
        )
        print("dof_limits_lower=", dof_limits_lower)
        print("dof_limits_upper =", dof_limits_upper)
        # Generate random DOF positions within limits
        random_dof_pos = torch.rand(
            self.num_envs,
            len(dof_limits_lower),
            device=self.device,
            requires_grad=False,
        )

        # Scale to joint limits
        dof_ranges = dof_limits_upper - dof_limits_lower
        random_dof_pos = (dof_limits_lower + random_dof_pos * dof_ranges).detach()

        return random_dof_pos

    def _gen_random_root_rotation(self):
        """Generate random root rotation quaternion"""
        # Generate random euler angles (roll, pitch, yaw)
        random_roll = (
            (torch.rand(self.num_envs, device=self.device) - 0.5) * 2 * torch.pi
        )  # [-π, π]
        random_pitch = (
            (torch.rand(self.num_envs, device=self.device) - 0.5) * 2 * torch.pi
        )  # [-π, π]
        random_yaw = (
            (torch.rand(self.num_envs, device=self.device) - 0.5) * 2 * torch.pi
        )  # [-π, π]

        # Convert to quaternion (xyzw format since w_last=True)
        random_quat = quat_from_euler_xyz(
            random_roll, random_pitch, random_yaw, w_last=True
        )

        return random_quat

    def _get_updated_marker_positions(self):
        """Update marker positions to follow the specified bodies"""
        if not self.viz_markers:
            return

        # this will convert to sim common ordering, which is the MJCF ordering
        current_state = self.simulator.get_bodies_state()

        idx_in_common = [
            self.simulator._body_names.index(body_name)
            for body_name in self.robot_spec.viz_bodies
        ]

        all_positions = (
            current_state.rigid_body_pos[:, idx_in_common, :].detach().clone()
        )
        all_orientations = (
            current_state.rigid_body_rot[:, idx_in_common, :].detach().clone()
        )

        # # surgery on the 1st marker
        # root_orientation = all_orientations[:, 0, :].detach().clone()
        # root_offset = torch.tensor([0.0, 0.1, 0.0], device=self.device)
        # root_offset = root_offset.repeat(self.num_envs, 1)
        # all_positions[:, 0, :] += quat_apply(root_orientation, root_offset, w_last=True)

        marker_states = {
            "body_markers": MarkerState(
                translation=all_positions, orientation=all_orientations
            )
        }

        return marker_states

    def run(self):
        """Main simulation loop"""
        step_count = 0

        # Parameters
        spacing = 4.0  # spacing between humanoids

        # Determine the grid size along each axis (cube root rounded up)
        grid_size = math.ceil(self.num_envs ** (1 / 3))

        # Create grid coordinates
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(grid_size, device=self.device),
                torch.arange(grid_size, device=self.device),
                torch.arange(grid_size, device=self.device),
                indexing="ij",  # ensures x,y,z layout
            ),
            dim=-1,
        ).reshape(-1, 3)

        # Scale by spacing and take only first N positions
        root_positions = coords[: self.num_envs] * spacing  # shape: (self.num_envs, 3)
        print("@@@@@@@@@@@@@@@@@@")
        while True:
            # Check for reset request (R key press triggers this in simulator)
            if self.simulator.user_requested_reset:
                current_state = self.simulator.get_robot_state()

                random_dof_pos = self._gen_random_pose()
                random_root_rot = self._gen_random_root_rotation()

                # since all sim are in reduced coordinate
                # we only need to set the root state and dof pos (vel)

                current_state.dof_pos = random_dof_pos.detach()
                current_state.dof_vel = torch.zeros_like(random_dof_pos).detach()
                current_state.rigid_body_pos[:, 0, :] = root_positions
                # NOTE: we use xyzw quaternion ordering for the common state shared by all simulators
                # current_state.rigid_body_rot[:, 0, :] = torch.tensor([0, 0, 0, 1.0], device=self.device).repeat(self.num_envs, 1)
                current_state.rigid_body_rot[:, 0, :] = random_root_rot.detach()
                current_state.rigid_body_vel[:, 0, :] = torch.zeros(
                    self.num_envs, 3, device=self.device
                )
                current_state.rigid_body_ang_vel[:, 0, :] = torch.zeros(
                    self.num_envs, 3, device=self.device
                )

                env_ids = torch.arange(self.num_envs, device=self.device)
                self.simulator.reset_envs(
                    current_state, new_object_states=None, env_ids=env_ids
                )

                # # we could set the full maximal coordinate state, but it's not necessary

                # random_dof_pos_w_root = torch.cat([
                #     torch.zeros(self.num_envs, 3, device=self.device),
                #     torch.tensor([1.0, 0, 0, 0], device=self.device).repeat(self.num_envs, 1),  # mjcf uses wxyz quaternion ordering
                #     random_dof_pos
                # ], dim=1).detach()

                # # pose lib fk function uses MJCF convention.
                # fk_state = fk_batch_mjcf_with_velocities(
                #     self.kinematic_info,
                #     random_dof_pos_w_root,
                #     fps=None,
                #     compute_velocities=False,
                # )

                # # Detach all tensors to avoid gradient issues
                # fk_state.dof_pos = random_dof_pos.detach()
                # fk_state.dof_vel = torch.zeros_like(random_dof_pos).detach()
                # fk_state.rigid_body_pos = fk_state.rigid_body_pos.detach()
                # fk_state.rigid_body_rot = fk_state.rigid_body_rot.detach()
                # fk_state.rigid_body_vel = torch.zeros_like(fk_state.rigid_body_pos).detach()
                # fk_state.rigid_body_ang_vel = torch.zeros_like(fk_state.rigid_body_pos).detach()

                # env_ids = torch.arange(self.num_envs, device=self.device)
                # self.simulator.reset_envs(fk_state, env_ids=env_ids)

            # zero torque control, so should stay at the reset random pose without moving (gravity off)
            _common_actions = torch.zeros(
                self.num_envs, self.robot_cfg.number_of_actions, device=self.device
            )

            marker_states = self._get_updated_marker_positions()

            self.simulator.step(_common_actions, markers_callback=lambda: marker_states)

            # self.simulator.user_requested_reset = False

            # self.simulator._common_actions = torch.zeros(self.num_envs, self.robot_spec.num_dofs, device=self.device)
            # self.simulator._apply_motor_forces()

            # # if self.simulator.control_type == ControlType.BUILT_IN_PD:
            # #     self.simulator._common_actions = torch.zeros(self.num_envs, self.robot_spec.num_dofs, device=self.device)
            # #     self.simulator._apply_pd_control()
            # # else:

            # self.simulator._simulate()
            # self.simulator._refresh_sim_tensors()

            # self.simulator.render()

            step_count += 1


def main():
    # Use the global args that were parsed early
    global args, AppLauncher

    device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

    # Extra simulator parameters for IsaacLab
    extra_simulator_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulation_app = app_launcher.app
        extra_simulator_params["simulation_app"] = simulation_app

    visualizer = RandomPoseVisualizer(
        robot_name=args.robot,
        num_envs=args.num_envs,
        simulator_type=args.simulator,
        headless=args.headless,
        cpu_only=args.cpu_only,
        extra_simulator_params=extra_simulator_params,
    )

    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        visualizer.simulator.close()


if __name__ == "__main__":
    main()
