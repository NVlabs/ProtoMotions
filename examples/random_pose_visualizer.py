# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Random-pose viewer: the first check for a new robot or a new simulator backend.

The viewer loads one robot into the chosen simulator with gravity off and zero
torque, then every two seconds of wall time writes a fresh random pose: joint
angles drawn uniformly inside the joint limits and a random root orientation.
Red spheres mark a few bodies. For the robots listed in ROBOT_SPECS they are
hand-picked: g1 marks the pelvis, torso, knees and ankles; smpl marks the
pelvis, knees and ankles. Any other registered robot gets a neutral default,
the root body plus whichever torso and foot bodies its config names in
common_naming_to_robot_body_names. A robot that holds each pose without
drifting, with the spheres sitting on the right bodies, has loaded correctly:
its joint limits, its body order and its asset agree.

Usage:
    python examples/random_pose_visualizer.py --robot g1 --simulator isaacgym

--robot accepts every registered robot name (one per module under
protomotions/robot_configs/); --simulator is isaacgym, isaaclab or newton.
There is no key for requesting a pose: the viewer re-poses on its own. Q closes
the viewer, as in every ProtoMotions simulator window.
"""

from typing import Dict, List
import argparse
from dataclasses import dataclass
import math
import time

from protomotions.robot_configs.factory import registered_robot_names

# Seconds between two random poses.
REPOSE_INTERVAL_S = 2.0

# Parse arguments first (argparse and the robot registry import no torch)
parser = argparse.ArgumentParser(
    description=(
        "Random-pose viewer: holds a robot in zero gravity with zero torque and "
        f"writes a new random pose every {REPOSE_INTERVAL_S:g} seconds."
    )
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
    choices=registered_robot_names(),
    default="g1",
    help="Robot to load; any registered robot name",
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
from protomotions.simulator.base_simulator.simulator_state import (  # noqa: E402
    ResetState,
    StateConversion,
)
from protomotions.simulator.factory import simulator_config  # noqa: E402
from protomotions.robot_configs.base import RobotConfig  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402
from protomotions.utils.rotations import quat_from_euler_xyz  # noqa: E402


@dataclass
class RobotSpec:
    """Robot specification with body names for visualization"""

    # Body names to visualize (these are the rigid body names, not joint names)
    viz_bodies: List[str]


# Hand-picked marker bodies. A robot that is not listed here gets the neutral
# default built by robot_spec_for().
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
    "smpl": RobotSpec(
        viz_bodies=["Pelvis", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"],
    ),
}

# The semantic body names the neutral default marks, when the robot config
# declares them (common_naming_to_robot_body_names keys).
NEUTRAL_SEMANTIC_BODIES = (
    "torso_body_name",
    "all_left_foot_bodies",
    "all_right_foot_bodies",
)


def robot_spec_for(robot_name: str, robot_cfg: RobotConfig) -> RobotSpec:
    """The marker bodies for a robot: its ROBOT_SPECS entry, or a neutral default.

    The default is the root body (body 0 of the kinematic tree) plus the torso
    and foot bodies the config names, in that order, each body once.
    """
    spec = ROBOT_SPECS.get(robot_name)
    if spec is not None:
        return spec
    body_names = robot_cfg.kinematic_info.body_names
    viz_bodies = [body_names[0]]
    for key in NEUTRAL_SEMANTIC_BODIES:
        for body in robot_cfg.common_naming_to_robot_body_names.get(key, []):
            if body in body_names and body not in viz_bodies:
                viz_bodies.append(body)
    return RobotSpec(viz_bodies=viz_bodies)


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
        self.num_envs = num_envs
        self.simulator_type = simulator_type
        self.headless = headless
        self.device = torch.device("cuda:0" if not cpu_only else "cpu")

        # Load robot configuration using factory function
        self.robot_cfg = robot_config(robot_name)
        self.robot_spec = robot_spec_for(robot_name, self.robot_cfg)

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

        kinematic_info = self.robot_cfg.kinematic_info
        print(f"Loaded {robot_name} robot using {simulator_type}")
        print(f"Robot config: {type(self.robot_cfg).__name__}")
        print(f"Number of actions: {self.robot_cfg.number_of_actions}")
        print(f"Number of DOFs: {kinematic_info.num_dofs}")
        print(f"Joint limits, lower: {kinematic_info.dof_limits_lower}")
        print(f"Joint limits, upper: {kinematic_info.dof_limits_upper}")
        print(f"Visualizing bodies: {self.robot_spec.viz_bodies}")
        print(f"A new random pose is written every {REPOSE_INTERVAL_S:g} s; Q closes")

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
        dof_limits_lower = self.robot_cfg.kinematic_info.dof_limits_lower.to(
            self.device
        )
        dof_limits_upper = self.robot_cfg.kinematic_info.dof_limits_upper.to(
            self.device
        )
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

    def _grid_root_positions(self) -> torch.Tensor:
        """Root positions on a cubic grid, one cell per environment."""
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
        return coords[: self.num_envs].float() * spacing  # shape: (num_envs, 3)

    def _write_random_pose(self, root_positions: torch.Tensor) -> None:
        """Write a new random pose to every environment.

        All simulators work in reduced coordinates, so the root pose and the
        joint positions are enough; the simulator computes the body poses.
        """
        print("Generating new random pose")
        random_dof_pos = self._gen_random_pose()
        zeros = torch.zeros(self.num_envs, 3, device=self.device)
        reset_state = ResetState(
            root_pos=root_positions.clone(),
            # xyzw quaternion ordering for the common state shared by all simulators
            root_rot=self._gen_random_root_rotation().detach(),
            root_vel=zeros,
            root_ang_vel=zeros.clone(),
            dof_pos=random_dof_pos,
            dof_vel=torch.zeros_like(random_dof_pos),
            state_conversion=StateConversion.COMMON,
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.simulator.reset_envs(reset_state, new_object_states=None, env_ids=env_ids)

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

        marker_states = {
            "body_markers": MarkerState(
                translation=all_positions, orientation=all_orientations
            )
        }

        return marker_states

    def run(self):
        """Main simulation loop: a new random pose every REPOSE_INTERVAL_S seconds."""
        root_positions = self._grid_root_positions()

        # zero torque control, so the robot stays at the written pose (gravity off)
        zero_actions = torch.zeros(
            self.num_envs, self.robot_cfg.number_of_actions, device=self.device
        )

        next_pose_at = time.monotonic()  # the first pose is written at once
        while self.simulator.is_simulation_running():
            now = time.monotonic()
            if now >= next_pose_at:
                self._write_random_pose(root_positions)
                next_pose_at = now + REPOSE_INTERVAL_S

            marker_states = self._get_updated_marker_positions()
            self.simulator.step(zero_actions, markers_callback=lambda: marker_states)


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
