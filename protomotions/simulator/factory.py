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
#
import dataclasses

from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.robot_configs.base import RobotConfig
import logging

log = logging.getLogger(__name__)


def get_simulator_config_class(simulator: str):
    """Get the simulator config class for a given simulator type.

    Args:
        simulator: Simulator type (isaacgym, isaaclab, newton, genesis)

    Returns:
        The simulator config class

    Raises:
        ValueError: If simulator type is not supported
    """
    if simulator == "isaacgym":
        from protomotions.simulator.isaacgym.config import IsaacGymSimulatorConfig

        return IsaacGymSimulatorConfig
    elif simulator == "isaaclab":
        from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig

        return IsaacLabSimulatorConfig
    elif simulator == "newton":
        from protomotions.simulator.newton.config import NewtonSimulatorConfig

        return NewtonSimulatorConfig
    elif simulator == "genesis":
        from protomotions.simulator.genesis.config import GenesisSimulatorConfig

        return GenesisSimulatorConfig
    elif simulator == "mujoco":
        from protomotions.simulator.mujoco.config import MujocoSimulatorConfig

        return MujocoSimulatorConfig
    else:
        raise ValueError(f"Unsupported simulator: {simulator}")


def simulator_config(
    simulator: str,
    robot_config: RobotConfig,
    headless: bool,
    num_envs: int,
    experiment_name: str,
) -> SimulatorConfig:
    """Factory function to create simulator configuration based on simulator type.

    Args:
        simulator: Simulator type (isaacgym, isaaclab, newton, genesis)
        robot_config: Robot configuration object
        headless: Whether to run in headless mode
        num_envs: Number of parallel environments
        experiment_name: Name of the experiment

    Returns:
        SimulatorConfig: Simulator configuration object

    Raises:
        ValueError: If simulator type is not supported
    """
    SimConfigClass = get_simulator_config_class(simulator)

    return SimConfigClass(
        sim=getattr(robot_config.simulation_params, simulator),
        headless=headless,
        num_envs=num_envs,
        experiment_name=experiment_name,
    )


def update_simulator_config_for_test(
    current_simulator_config: SimulatorConfig,
    new_simulator: str,
    robot_config: RobotConfig,
) -> SimulatorConfig:
    """Update simulator config when switching simulators during inference.

    This function handles the case where inference uses a different simulator than training.
    It dynamically instantiates the new simulator's config class to extract its default values
    (_target_, w_last, etc.) and updates the current config accordingly.

    Args:
        current_simulator_config: The simulator config from training (loaded from resolved_configs_inference.pt)
        new_simulator: The new simulator type to use (e.g., 'isaaclab', 'isaacgym', 'genesis', 'newton', 'mujoco')
        robot_config: Robot configuration containing simulation_params for all simulators

    Returns:
        SimulatorConfig: Updated simulator config with new simulator's settings

    Raises:
        ValueError: If simulator type is not supported or robot config lacks params for the simulator
    """
    # Check if the simulator params exist; if not, add defaults for newly added simulators
    if not hasattr(robot_config.simulation_params, new_simulator):
        log.warning(
            f"Robot config missing simulation_params for '{new_simulator}'. "
            f"Adding default params (likely from an older checkpoint)."
        )
        # Get the simulator config class to extract default sim params
        SimConfigClass = get_simulator_config_class(new_simulator)
        default_sim_params = SimConfigClass.__dataclass_fields__["sim"].default_factory()
        setattr(robot_config.simulation_params, new_simulator, default_sim_params)
    
    if getattr(robot_config.simulation_params, new_simulator) is None:
        raise ValueError(
            f"Robot config does not have simulation_params for '{new_simulator}'. "
            f"Available simulators: {list(robot_config.simulation_params.__dataclass_fields__.keys())}"
        )

    log.info(
        f"Updating simulator config from training to use '{new_simulator}' for inference"
    )

    # Get the simulator config class and extract its default values
    SimConfigClass = get_simulator_config_class(new_simulator)

    # Extract defaults from the config class's dataclass fields
    # This way we don't hardcode _target_, w_last, or any other simulator-specific defaults
    config_fields = SimConfigClass.__dataclass_fields__

    new_target = config_fields["_target_"].default
    new_w_last = config_fields["w_last"].default
    new_sim_params = getattr(robot_config.simulation_params, new_simulator)

    # Update the current simulator config with new values
    current_simulator_config._target_ = new_target
    current_simulator_config.w_last = new_w_last
    current_simulator_config.sim = new_sim_params

    # Copy any simulator-specific fields that don't exist on the old config object
    # (e.g. MujocoSimulatorConfig.use_implicit_pd when switching from IsaacGym/Lab)
    for field_name, field_def in config_fields.items():
        if field_name in ("_target_", "w_last", "sim"):
            continue  # Already handled above
        if not hasattr(current_simulator_config, field_name):
            default = (
                field_def.default_factory()
                if field_def.default_factory is not dataclasses.MISSING
                else field_def.default
            )
            setattr(current_simulator_config, field_name, default)
            log.info(f"  {field_name} -> {default} (new field, default value)")

    log.info(f"  _target_ -> {current_simulator_config._target_}")
    log.info(f"  w_last -> {current_simulator_config.w_last}")
    log.info(
        f"  sim params -> {new_simulator} (fps={current_simulator_config.sim.fps}, decimation={current_simulator_config.sim.decimation})"
    )

    return current_simulator_config
