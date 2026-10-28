# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the G1 unconditional GPC prior as a fixed-shape native graph.

The training checkpoint contains an autoregressive discrete prior. This
exporter unrolls its eight token decisions into one graph and leaves the
sampling noise as an input for the Gumbel and top-p variants. That keeps the
TensorRT engine static and CUDA-graph compatible while allowing the native
launcher to refresh the random input every control cycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
NUM_DOFS = 29
OBSERVATION_DIM = 64
PREVIOUS_ACTION_DIM = 29
NUM_TOKENS = 8
VOCAB_SIZE = 6561
FSQ_LEVELS = 9
FSQ_SCALARS_PER_TOKEN = 4


def _load_model(experiment_dir: Path):
    sys.path.insert(0, str(REPO_ROOT))

    # The checkpoint was produced before a public helper rename. This alias
    # is local to the exporter process and does not modify ProtoMotions.
    import protomotions.envs.obs.target_poses as target_poses

    if not hasattr(
        target_poses, "build_reference_relative_reduced_coords_target_poses"
    ):
        target_poses.build_reference_relative_reduced_coords_target_poses = (
            target_poses.build_reduced_coords_target_poses
        )

    from protomotions.agents.supervised.latent_prior_model import (
        DiscreteAutoregressiveLatentPriorModel,
    )
    from protomotions.agents.utils.normalization import (
        materialize_lazy_running_stats_from_state_dict,
    )

    resolved = torch.load(
        experiment_dir / "resolved_configs_inference.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = resolved["agent"].model
    action_config = resolved["env"].action_config
    action_fn = action_config.get("fn")
    if getattr(action_fn, "__name__", None) != "bm_pd_action":
        raise RuntimeError(
            "the GPC exporter currently requires the ProtoMotions bm_pd_action "
            f"processor, got {action_fn!r}"
        )
    model = DiscreteAutoregressiveLatentPriorModel(config)
    state_dict = torch.load(
        experiment_dir / "inference_last.ckpt",
        map_location="cpu",
        weights_only=False,
    )["model"]
    model.materialize_from_state_dict(state_dict)
    materialize_lazy_running_stats_from_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, action_config


class GPCExportModule(nn.Module):
    """Tensor-only unrolled GPC graph.

    The argmax mode is deterministic and has two inputs. The sampling modes
    add gumbel_noise with shape [1, 8, 6561]. A standard Gumbel input makes
    Gumbel-max sampling mathematically equivalent to categorical sampling and
    keeps random-number generation outside TensorRT.
    """

    def __init__(
        self,
        model: nn.Module,
        action_config,
        mode: str,
        top_p: float,
        temperature: float,
    ):
        super().__init__()
        self.mode = mode
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        prior = model.prior
        decoder = model.latent_decoder.decoder
        self.context_mlp = prior._context_encoder.models[0]
        self.token_mlp = prior._token_encoder.models[0]
        self.output_mlp = prior._output_head.models[0]
        self.transformer = prior._transformer
        self.pos_emb = prior._pos_emb
        self.decoder_mlp = decoder
        # Keep the action contract identical to ProtoMotions inference. The
        # decoder emits the raw policy action; deployment consumes the
        # processed joint-position target used by bm_pd_action.
        self.register_buffer(
            "pd_action_offset",
            torch.as_tensor(action_config["pd_action_offset"], dtype=torch.float32),
        )
        self.register_buffer(
            "action_scale",
            torch.as_tensor(action_config["action_scale"], dtype=torch.float32),
        )
        if (
            self.pd_action_offset.numel() != NUM_DOFS
            or self.action_scale.numel() != NUM_DOFS
        ):
            raise RuntimeError("GPC BM action processor must have 29-DOF constants")
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _mlp(module: nn.Module, values: torch.Tensor) -> torch.Tensor:
        # The saved GPC configs use one Forward operation: normalize, then MLP.
        return module.mlp(module.norm(values))

    def _next_logits(
        self, context: torch.Tensor, generated: list[torch.Tensor]
    ) -> torch.Tensor:
        if generated:
            tokens = torch.stack(generated, dim=1)
            token_one_hot = F.one_hot(tokens.long(), VOCAB_SIZE).float()
            token_embedding = self._mlp(self.token_mlp, token_one_hot)
            sequence = torch.cat([context.unsqueeze(1), token_embedding], dim=1)
        else:
            sequence = context.unsqueeze(1)

        sequence = sequence + self.pos_emb[:, : sequence.shape[1], :]
        sequence_length = sequence.shape[1]
        target_length = sequence_length - 1
        mask = torch.zeros(
            (sequence_length, sequence_length),
            dtype=sequence.dtype,
            device=sequence.device,
        )
        mask[0, 1:] = float("-inf")
        if target_length > 0:
            causal = torch.triu(
                torch.ones(
                    (target_length, target_length),
                    dtype=sequence.dtype,
                    device=sequence.device,
                ),
                diagonal=1,
            )
            mask[1:, 1:] = causal.masked_fill(causal == 1, float("-inf"))
        # Explicitly disable automatic causal-mask detection. Its Python bool
        # guard is data-dependent under torch.export even though the mask is
        # static for every unrolled step.
        hidden = self.transformer(sequence, mask=mask, is_causal=False)
        return self._mlp(self.output_mlp, hidden[:, -1])

    def _sample_token(
        self,
        logits: torch.Tensor,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        scaled_logits = logits / self.temperature
        if self.mode == "argmax":
            return torch.argmax(scaled_logits, dim=-1)

        if noise is None:
            raise RuntimeError("sampling mode requires gumbel noise")
        if self.mode == "gumbel":
            return torch.argmax(scaled_logits + noise, dim=-1)

        probabilities = torch.softmax(scaled_logits, dim=-1)
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities, descending=True, dim=-1
        )
        cutoff = torch.cumsum(sorted_probabilities, dim=-1) > self.top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        keep_mask = torch.zeros_like(probabilities, dtype=torch.bool)
        keep_mask.scatter_(-1, sorted_indices, ~cutoff)
        filtered_logits = torch.where(
            keep_mask,
            scaled_logits,
            torch.full_like(scaled_logits, float("-inf")),
        )
        return torch.argmax(filtered_logits + noise, dim=-1)

    def _decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        remainder = tokens
        groups = []
        for exponent in range(FSQ_SCALARS_PER_TOKEN - 1, -1, -1):
            basis = FSQ_LEVELS**exponent
            groups.append(torch.div(remainder, basis, rounding_mode="floor"))
            remainder = torch.remainder(remainder, basis)
        indices = torch.stack(groups[::-1], dim=-1).reshape(tokens.shape[0], -1)
        codes = indices.float() - (FSQ_LEVELS - 1) / 2.0
        return codes

    def forward(
        self,
        reduced_coords_obs: torch.Tensor,
        previous_processed_action: torch.Tensor,
        gumbel_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self._mlp(self.context_mlp, reduced_coords_obs)
        generated: list[torch.Tensor] = []
        for step in range(NUM_TOKENS):
            logits = self._next_logits(context, generated)
            noise = None if gumbel_noise is None else gumbel_noise[:, step, :]
            generated.append(self._sample_token(logits, noise))
        tokens = torch.stack(generated, dim=1)
        codes = self._decode_tokens(tokens)
        decoder_input = torch.cat(
            [reduced_coords_obs, previous_processed_action, codes], dim=-1
        )
        raw_action = self._mlp(self.decoder_mlp, decoder_input)
        return self.pd_action_offset + self.action_scale * raw_action


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rewrite_top_p_with_plugin(path: Path, top_p: float) -> None:
    """Replace each exported full sort with the exact GPU TopPFilter plugin."""
    import onnx
    from onnx import helper

    model = onnx.load(path, load_external_data=True)
    nodes = list(model.graph.node)
    replacements = []
    for topk in nodes:
        if topk.op_type != "TopK" or len(topk.input) < 1:
            continue
        softmax_name = topk.input[0]
        softmax_index = next(
            (i for i, node in enumerate(nodes) if softmax_name in node.output), None
        )
        if softmax_index is None or nodes[softmax_index].op_type != "Softmax":
            continue
        logits_name = nodes[softmax_index].input[0]
        where_index = next(
            (
                i
                for i in range(softmax_index + 1, len(nodes))
                if nodes[i].op_type == "Where" and logits_name in nodes[i].input
            ),
            None,
        )
        if where_index is None:
            raise RuntimeError(f"could not find top-p filter boundary for {topk.name}")
        where = nodes[where_index]
        if len(where.output) != 1:
            raise RuntimeError(f"unexpected top-p Where output for {where.name}")
        boundary_output = where.output[0]
        subgraph_nodes = nodes[softmax_index : where_index + 1]
        for node in subgraph_nodes:
            for output in node.output:
                if output == boundary_output:
                    continue
                consumers = [
                    consumer
                    for consumer in nodes
                    if consumer not in subgraph_nodes and output in consumer.input
                ]
                if consumers:
                    raise RuntimeError(
                        f"top-p rewrite would remove externally consumed value {output}"
                    )
        plugin = helper.make_node(
            "TopPFilter",
            [logits_name],
            list(where.output),
            name=f"{topk.name}_TopPFilter",
            domain="",
            top_p=float(top_p),
        )
        replacements.append((softmax_index, where_index, plugin))

    if len(replacements) != NUM_TOKENS:
        raise RuntimeError(
            f"expected {NUM_TOKENS} top-p subgraphs, found {len(replacements)}"
        )
    for start, end, plugin in reversed(replacements):
        nodes[start : end + 1] = [plugin]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # ONNX checker intentionally rejects the deployment-only custom op;
    # TensorRT validates it through the registered TopPFilter creator.
    data_path = path.with_name(path.name + ".data")
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("argmax", "gumbel", "topp"), default="argmax"
    )
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("top-p must be in (0, 1]")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    experiment_dir = args.experiment_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, action_config = _load_model(experiment_dir)
    exporter = GPCExportModule(
        model, action_config, args.mode, args.top_p, args.temperature
    )

    observation = torch.zeros(1, OBSERVATION_DIM)
    previous_action = torch.zeros(1, PREVIOUS_ACTION_DIM)
    inputs = (observation, previous_action)
    input_names = ["reduced_coords_obs", "previous_processed_action"]
    if args.mode != "argmax":
        noise = torch.zeros(1, NUM_TOKENS, VOCAB_SIZE)
        inputs = (observation, previous_action, noise)
        input_names.append("gumbel_noise")

    output = output_dir / "unified_pipeline.onnx"
    with torch.inference_mode():
        reference = exporter(*inputs)
        torch.onnx.export(
            exporter,
            inputs,
            output,
            input_names=input_names,
            output_names=["action"],
            opset_version=args.opset,
            dynamo=True,
            do_constant_folding=True,
        )
    reference_onnx_name = None
    if args.mode == "topp":
        # Keep the unmodified sort/cumsum/threshold graph for numerical parity
        # checks. TensorRT uses the rewritten graph, while ORT runs this exact
        # reference graph on CPU.
        import onnx

        reference_output = output_dir / "reference_pipeline.onnx"
        reference_data = reference_output.with_name(reference_output.name + ".data")
        reference_model = onnx.load(output, load_external_data=True)
        if reference_data.exists():
            reference_data.unlink()
        onnx.save_model(
            reference_model,
            reference_output,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=reference_data.name,
            size_threshold=1024,
        )
        reference_onnx_name = reference_output.name
        _rewrite_top_p_with_plugin(output, args.top_p)

    manifest = {
        "contract": "protomotions.gpc.g1.unconditional.v1",
        "policy_family": "gpc",
        "sampling_mode": args.mode,
        "top_p": args.top_p if args.mode == "topp" else None,
        "temperature": args.temperature,
        "greedy": args.mode == "argmax",
        "reference_onnx": reference_onnx_name,
        "inputs": [
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": "float32",
            }
            for name, tensor in zip(input_names, inputs)
        ],
        "outputs": [
            {
                "name": "action",
                "shape": [1, NUM_DOFS],
                "dtype": "float32",
                "semantics": "processed_joint_position_target",
            }
        ],
        "observation": {
            "name": "reduced_coords_obs",
            "dimension": OBSERVATION_DIM,
            "layout": [
                "dof_pos[29]",
                "dof_vel[29]",
                "root_local_ang_vel[3]",
                "projected_gravity[3]",
            ],
        },
        "previous_action": {
            "name": "previous_processed_action",
            "dimension": PREVIOUS_ACTION_DIM,
            "semantics": "previous processed joint-position target",
        },
        "action_postprocessing": {
            "owner": "ProtoMotions bm_pd_action",
            "output_domain": "processed_joint_position_target",
            "offset": torch.as_tensor(action_config["pd_action_offset"]).tolist(),
            "scale": torch.as_tensor(action_config["action_scale"]).tolist(),
            "stiffness": torch.as_tensor(action_config["stiffness"]).tolist(),
            "damping": torch.as_tensor(action_config["damping"]).tolist(),
        },
        "model_sha256": _sha256(experiment_dir / "inference_last.ckpt"),
        "reference_zero_input_action": reference[0].tolist(),
        "tensorrt_plugins": (
            [{"name": "TopPFilter", "version": "1", "domain": ""}]
            if args.mode == "topp"
            else []
        ),
    }
    (output_dir / "unified_pipeline.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"exported {args.mode} GPC graph: {output}", flush=True)
    print(f"manifest: {output_dir / 'unified_pipeline.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
