# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure eager G1 sensor estimation and history cost, not full training speed.

Run from the repository root with ``python -m scripts.benchmark_sensor_derivation``.
The history-reconstruction case illustrates the removed storage design using
the same estimator; it is not a benchmark of the old implementation's compiler.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Callable

import torch

from protomotions.envs.base_env.publishers import StatePublisher
from protomotions.envs.context_sources import (
    PRIVILEGED,
    SENSOR_SOURCES,
    SOURCES,
    sensed,
)
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.humanoid import compute_humanoid_reduced_coords_observations
from protomotions.envs.obs.state_history_buffer import HistorySample, StateHistoryBuffer
from protomotions.envs.sensors.config import (
    BaseVelocitySensor,
    IMUSensor,
    JointEncoderSensor,
    LocalizationSensor,
    SensorConfig,
)
from protomotions.envs.sensors.estimation import estimate_body_state
from protomotions.envs.sensors.measurement import SensorReadings
from protomotions.envs.sensors.runtime import SensorEstimates, SensorRuntime
from protomotions.robot_configs.g1 import G1RobotConfig
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)


def measure_time(
    run: Callable[[], object], device: torch.device, iterations: int
) -> dict:
    """Return wall times in ms, synchronizing CUDA so queued kernels are counted."""
    for _ in range(3):
        run()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    result = {
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[max(0, (95 * len(samples) + 99) // 100 - 1)],
        "min_ms": min(samples),
        "max_ms": max(samples),
    }
    if device.type == "cuda":
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    return result


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations) < 1 or args.history < 0:
        parser.error("num-envs and iterations must be positive; history nonnegative")
    device = torch.device(args.device)
    robot = G1RobotConfig()
    config = SensorConfig(
        joint_encoders=JointEncoderSensor(pos_noise=0.01, vel_noise=0.1),
        imu=IMUSensor(quat_noise=0.01, gyro_noise=0.1),
        localization=LocalizationSensor(body_name="pelvis", noise=0.01),
        base_velocity=BaseVelocitySensor(noise=0.1),
        seed=23,
    )
    runtime = SensorRuntime(config, robot, args.num_envs, device)
    joints = torch.zeros(args.num_envs, robot.kinematic_info.num_dofs, device=device)
    quat = joints.new_zeros(args.num_envs, 4)
    quat[:, 3] = 1
    xyz = joints.new_zeros(args.num_envs, 3)
    readings = SensorReadings(
        joints, joints.clone(), quat, xyz, xyz.clone(), xyz.clone()
    )
    ideal = estimate_body_state(readings, runtime.fk, runtime.mounts)
    state = RobotState(
        state_conversion=StateConversion.COMMON,
        rigid_body_pos=ideal.rigid_body_pos,
        rigid_body_rot=ideal.rigid_body_rot,
        rigid_body_vel=ideal.rigid_body_vel,
        rigid_body_ang_vel=ideal.rigid_body_ang_vel,
        dof_pos=ideal.dof_pos,
        dof_vel=ideal.dof_vel,
    )
    ids = torch.arange(args.num_envs, device=device)
    estimates = runtime.reset(ids, state)
    actions = torch.zeros_like(joints)

    def sample(estimates: SensorEstimates) -> HistorySample:
        states = {PRIVILEGED: ideal}
        states.update({source: getattr(estimates, source) for source in SENSOR_SOURCES})
        return HistorySample(states, actions, actions)

    history = StateHistoryBuffer(args.history, sample(estimates))
    snapshot = history.capture()
    storage_bytes = sum(
        value.numel() * value.element_size() for _, value in snapshot.tensors()
    )

    def step_and_capture() -> HistorySample:
        history.push(sample(runtime.step(state)))
        return history.capture()

    # The kernel binds the privileged names; sensed() repoints a copy at each
    # sensor source (and is the identity for the privileged one).
    reduced = MdpComponent.from_kernel(
        compute_humanoid_reduced_coords_observations,
        root_vel_obs=True,
        root_rot=EnvContext.current.root_rot,
        root_vel=EnvContext.current.root_vel,
    )
    observations = {
        source: sensed({"reduced": reduced}, source)["reduced"] for source in SOURCES
    }

    def observe(
        ring: StateHistoryBuffer, sources: tuple[str, ...]
    ) -> list[torch.Tensor]:
        # The env's own publisher: one capture of the ring, the derived names
        # of every source, flat under the context paths the components bind.
        context = StatePublisher(ring, 0).publish(None)
        return [observations[source].compute(context) for source in sources]

    def step_store_and_observe() -> list[torch.Tensor]:
        step_and_capture()
        return observe(history, SOURCES)

    privileged_sample = HistorySample({PRIVILEGED: ideal}, actions, actions)
    privileged_history = StateHistoryBuffer(args.history, privileged_sample)

    def privileged_step_and_observe() -> list[torch.Tensor]:
        privileged_history.push(privileged_sample)
        return observe(privileged_history, (PRIVILEGED,))

    reset_ids = ids[::16]
    reset_state = state[reset_ids]

    def partial_reset() -> HistorySample:
        estimates = runtime.reset(reset_ids, reset_state)
        states = {PRIVILEGED: ideal.map(lambda v: v[reset_ids])}
        states.update({source: getattr(estimates, source) for source in SENSOR_SOURCES})
        subset = HistorySample(states, actions[reset_ids], actions[reset_ids])
        history.reset(reset_ids, subset)
        return history.capture(reset_ids)

    combined = SensorReadings.cat([readings, readings])
    repeated = SensorReadings.cat([readings] * (2 * (args.history + 1)))
    reconstructed = estimate_body_state(combined, runtime.fk, runtime.mounts)
    for name, value in ideal.tensors():
        torch.testing.assert_close(getattr(reconstructed, name)[: args.num_envs], value)

    print(
        json.dumps(
            {
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu",
                "torch": torch.__version__,
                "envs": args.num_envs,
                "history": args.history,
                "reset_envs": len(reset_ids),
                "history_storage_mib": storage_bytes / 2**20,
                "owned_context_snapshot_mib": storage_bytes / 2**20,
                "scope": "sensor/history microbenchmark; excludes simulator, rewards and policy",
            }
        ),
        flush=True,
    )
    cases = {
        "reconstruct_current_and_history": lambda: estimate_body_state(
            repeated, runtime.fk, runtime.mounts
        ),
        "reconstruct_current_only": lambda: estimate_body_state(
            combined, runtime.fk, runtime.mounts
        ),
        "measure_noise_estimate": lambda: runtime.step(state),
        "measure_noise_estimate_store_capture": step_and_capture,
        "sensor_store_capture_views_reduced_obs": step_store_and_observe,
        "privileged_store_capture_views_reduced_obs": privileged_step_and_observe,
        "partial_sensor_reset_store_capture": partial_reset,
    }
    for name, run in cases.items():
        print(
            json.dumps({"case": name, **measure_time(run, device, args.iterations)}),
            flush=True,
        )


if __name__ == "__main__":
    main()
