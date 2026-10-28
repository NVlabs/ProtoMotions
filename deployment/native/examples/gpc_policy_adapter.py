# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Code-only ONNX Runtime adapter for a user-provided G1 GPC graph."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


class GpcPolicy:
    """Run one exported GPC sampling graph through a compatible sensor bridge.

    The caller supplies the bridge and owns safety/command acknowledgement.
    This module does not provide a hardware launcher or download model weights.

    The graph consumes the same reduced observation as the native runner:
    29 DOF positions, 29 DOF velocities, pelvis/root local angular velocity,
    and projected gravity. Its output is already the ProtoMotions BM-PD
    processed joint-position target.
    """

    def __init__(
        self,
        artifact: Path,
        manifest: Path,
        provider: str,
        trt_cache_dir: Path | None = None,
        trt_cuda_graph: bool = False,
        startup_profile: bool = True,
        seed: int | None = None,
    ) -> None:
        self.artifact = artifact.resolve()
        self.manifest_path = manifest.resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("contract") != "protomotions.gpc.g1.unconditional.v1":
            raise ValueError("unsupported GPC manifest contract")
        if self.manifest.get("sampling_mode") != "gumbel":
            raise ValueError("this adapter is for the Gumbel GPC artifact")
        output = self.manifest.get("outputs", [{}])[0]
        if (
            output.get("name") != "action"
            or output.get("shape") != [1, 29]
            or output.get("dtype") != "float32"
            or output.get("semantics") != "processed_joint_position_target"
        ):
            raise ValueError("GPC artifact must emit processed 29-DOF targets")
        post = self.manifest.get("action_postprocessing", {})
        if post.get("output_domain") != "processed_joint_position_target":
            raise ValueError("GPC artifact is missing its processed-action contract")
        self._stiffness = np.asarray(post["stiffness"], dtype=np.float64)
        self._damping = np.asarray(post["damping"], dtype=np.float64)
        if self._stiffness.shape != (29,) or self._damping.shape != (29,):
            raise ValueError("GPC gain vectors must have 29 elements")

        providers: dict[str, list[Any]] = {
            "cpu": ["CPUExecutionProvider"],
            "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "tensorrt": [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        }
        if provider not in providers:
            raise ValueError(f"unsupported provider {provider!r}")
        available = ort.get_available_providers()
        requested = providers[provider]
        if not set(requested).issubset(set(available)):
            raise RuntimeError(
                f"requested providers {requested} unavailable; found {available}"
            )
        session_providers: list[Any] = requested
        self.trt_cuda_graph = bool(trt_cuda_graph and provider == "tensorrt")
        if provider == "tensorrt" and os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
            # GPC sampling is discrete: TF32 rounding can change an argmax
            # token even when the continuous action error looks small. Force
            # the exact FP32 TensorRT tactic family unless the caller already
            # selected the safe override.
            os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
            if startup_profile:
                print(
                    "startup GPC: forcing NVIDIA_TF32_OVERRIDE=0 for sampling parity",
                    flush=True,
                )
        if provider == "tensorrt" and (
            trt_cache_dir is not None or self.trt_cuda_graph
        ):
            cache_dir = Path(trt_cache_dir or "/tmp/protomotions_gpc_trt")
            if os.environ.get(
                "NVIDIA_TF32_OVERRIDE"
            ) == "0" and not cache_dir.name.endswith("_notf32"):
                # Never reuse a cache built with TF32 after forcing exact
                # discrete-token sampling. Keep the safe cache visibly
                # separate from older user-provided cache directories.
                cache_dir = cache_dir.with_name(cache_dir.name + "_notf32")
            cache_dir.mkdir(parents=True, exist_ok=True)
            options: dict[str, Any] = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_dir),
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": str(cache_dir),
            }
            if self.trt_cuda_graph:
                options["trt_cuda_graph_enable"] = True
                options["trt_auxiliary_streams"] = 0
            session_providers = [
                ("TensorrtExecutionProvider", options),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        session_start = time.perf_counter()
        self.session = ort.InferenceSession(
            str(self.artifact), providers=session_providers
        )
        if startup_profile:
            print(
                f"startup GPC: ORT/{provider} session ready in {time.perf_counter() - session_start:.3f}s "
                f"providers={self.session.get_providers()} graph={self.trt_cuda_graph} tf32=disabled",
                flush=True,
            )
        self.input_names = [entry.name for entry in self.session.get_inputs()]
        self.output_names = [entry.name for entry in self.session.get_outputs()]
        expected = ["reduced_coords_obs", "previous_processed_action", "gumbel_noise"]
        if self.input_names != expected or self.output_names != ["action"]:
            raise ValueError(
                f"unexpected GPC graph names: inputs={self.input_names!r} outputs={self.output_names!r}"
            )
        self._observation = np.zeros(64, dtype=np.float32)
        self._previous_action = np.zeros(29, dtype=np.float32)
        noise_shape = tuple(int(value) for value in self.manifest["inputs"][2]["shape"])
        self._noise = np.zeros(noise_shape, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._feed = {
            "reduced_coords_obs": self._observation[None, :],
            "previous_processed_action": self._previous_action[None, :],
            "gumbel_noise": self._noise,
        }
        self._graph_io_binding = None
        self._graph_inputs: dict[str, Any] = {}
        self._graph_outputs: dict[str, Any] = {}
        if self.trt_cuda_graph:
            self._init_cuda_graph_bindings(ort)
        self._steps = 0

    def _init_cuda_graph_bindings(self, ort_module: Any) -> None:
        """Bind persistent CUDA input/output buffers for ORT graph capture."""
        self._graph_io_binding = self.session.io_binding()
        for entry in self.session.get_inputs():
            shape = tuple(
                1 if dimension is None or isinstance(dimension, str) else int(dimension)
                for dimension in entry.shape
            )
            value = ort_module.OrtValue.ortvalue_from_shape_and_type(
                shape, np.float32, "cuda", 0
            )
            self._graph_inputs[entry.name] = value
            self._graph_io_binding.bind_ortvalue_input(entry.name, value)
        for entry in self.session.get_outputs():
            shape = tuple(
                1 if dimension is None or isinstance(dimension, str) else int(dimension)
                for dimension in entry.shape
            )
            value = ort_module.OrtValue.ortvalue_from_shape_and_type(
                shape, np.float32, "cuda", 0
            )
            self._graph_outputs[entry.name] = value
            self._graph_io_binding.bind_ortvalue_output(entry.name, value)

    def _run_session(self) -> np.ndarray:
        if not self.trt_cuda_graph:
            return np.asarray(
                self.session.run(self.output_names, self._feed)[0], dtype=np.float32
            )
        if self._graph_io_binding is None:
            raise RuntimeError("GPC CUDA graph bindings were not initialized")
        for name, value in self._feed.items():
            self._graph_inputs[name].update_inplace(np.ascontiguousarray(value))
        self._graph_io_binding.synchronize_inputs()
        self.session.run_with_iobinding(self._graph_io_binding)
        self._graph_io_binding.synchronize_outputs()
        return np.asarray(self._graph_outputs["action"].numpy(), dtype=np.float32)

    def _fill_gumbel_noise(self) -> None:
        # Reuse the persistent TensorRT input buffer. This avoids allocating a
        # 0.8 MB uniform array and two temporary log arrays every control tick.
        self._rng.random(self._noise.shape, dtype=np.float32, out=self._noise)
        np.clip(
            self._noise,
            np.float32(1.0e-6),
            np.float32(1.0 - 1.0e-6),
            out=self._noise,
        )
        np.log(self._noise, out=self._noise)
        np.negative(self._noise, out=self._noise)
        np.log(self._noise, out=self._noise)
        np.negative(self._noise, out=self._noise)

    def warmup(self) -> None:
        self._observation.fill(0.0)
        self._previous_action.fill(0.0)
        self._fill_gumbel_noise()
        self._run_session()
        if self.trt_cuda_graph:
            self._run_session()
        print("startup GPC: provider warmup complete", flush=True)

    def reset(self) -> None:
        self._previous_action.fill(0.0)
        self._steps = 0

    def enter_idle(self) -> None:
        self.reset()

    def start_tracking(self) -> None:
        self.reset()

    def step(
        self, bridge: Any, controls: Any, state: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del controls
        position = np.ctypeslib.as_array(state.contents.dof_position, shape=(29,))
        velocity = np.ctypeslib.as_array(state.contents.dof_velocity, shape=(29,))
        root_ang_vel = np.asarray(bridge.policy_root_angular_velocity, dtype=np.float32)
        if state.contents.has_projected_gravity:
            projected_gravity = np.ctypeslib.as_array(
                state.contents.projected_gravity, shape=(3,)
            )
        else:
            projected_gravity = np.asarray(
                bridge.policy_projected_gravity, dtype=np.float32
            )
        self._observation[:29] = position
        self._observation[29:58] = velocity
        self._observation[58:61] = root_ang_vel
        self._observation[61:64] = projected_gravity
        self._fill_gumbel_noise()
        output = self._run_session()
        action = np.asarray(output[0], dtype=np.float32)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise FloatingPointError("GPC returned an invalid action")
        np.copyto(self._previous_action, action)
        self._steps += 1
        return action, self._stiffness, self._damping
