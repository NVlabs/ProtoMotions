"""GPC prior evaluation: teacher-distribution fit and autonomous upright survival.

No reference-tracking failure is used for autonomous control. Offline sufficient
statistics are globally summed before normalization; rollout means are globally
item-weighted. All ranks return the same scalars to the existing logging path.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
import csv
import json
import logging
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from protomotions.agents.common.latent import LATENT_LOGITS_KEY, TARGET_LATENT_KEY
from protomotions.agents.evaluators.gpc_prior_offline import evaluate_prior_offline, evaluation_rng, frozen_encoder_digest
from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.gpc_prior_config import GPCPriorEvaluatorConfig


log = logging.getLogger(__name__)


def token_metrics(logits, targets):
    """Unsmoothed NLL and exact-match diagnostics, with one result per sample."""
    logits = logits.float()
    nll = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
    nll = nll.reshape_as(targets)
    correct = logits.argmax(-1).eq(targets)
    top5 = logits.topk(min(5, logits.shape[-1]), dim=-1).indices.eq(targets[..., None]).any(-1)
    result = {
        "nll": nll.mean(-1),
        "token_accuracy": correct.float().mean(-1),
        "top5_accuracy": top5.float().mean(-1),
        "sequence_accuracy": correct.all(-1).float(),
    }
    for index in range(targets.shape[-1]):
        result[f"token_{index}/nll"] = nll[:, index]
        result[f"token_{index}/accuracy"] = correct[:, index].float()
    return result


def upright_failure(height, rotation_xyzw, min_height, min_up_z):
    """Anchor local +Z projected on world +Z; rotation convention is XYZW."""
    up_z = 1 - 2 * (rotation_xyzw[..., 0].square() + rotation_xyzw[..., 1].square())
    finite = torch.isfinite(height) & torch.isfinite(rotation_xyzw).all(-1)
    return ~finite | (height < min_height) | (up_z < min_up_z)


class SurvivalAccumulator:
    def __init__(self, initial_bad, dt, max_steps, persistence):
        self.alive = ~initial_bad
        self.streak = torch.zeros_like(initial_bad, dtype=torch.long)
        self.seconds = torch.full_like(initial_bad, max_steps * dt, dtype=torch.float)
        self.seconds[initial_bad] = 0
        self.dt, self.persistence = dt, persistence

    def update(self, bad, step):
        self.streak = torch.where(bad, self.streak + 1, 0)
        failed = self.alive & (self.streak >= self.persistence)
        self.seconds[failed] = step * self.dt
        self.alive &= ~failed


def select_motion_ids(count, limit, seed, rank=0, world_size=1, shared=True):
    """Fixed without-replacement cohort; shared validation is disjoint by rank."""
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randperm(count, generator=generator)
    if shared:
        ids = ids[rank::world_size]
    if ids.numel() == 0:
        raise ValueError("Motion library must contain at least one motion per evaluation rank")
    return ids[:limit]


class GPCPriorEvaluator(BaseEvaluator):
    config: GPCPriorEvaluatorConfig

    def __init__(self, agent, fabric, config):
        config.validate()
        super().__init__(agent, fabric, config)
        self._validation_lib = None
        self._offline_batches = {}
        self._encoder_digest = None

    @contextmanager
    def _evaluation_scope(self):
        env = self.env
        snapshot = env.save_state()
        original_lib, original_manager = env.motion_lib, env.motion_manager
        agent_lib = self.agent.motion_lib
        original_noise = env.robot_config.reset_noise
        original_push = env.simulator._push_enabled
        was_training = self.agent.model.training
        py_rng, np_rng = random.getstate(), np.random.get_state()
        sampling = {k: getattr(self.agent.model, k) for k in ("temperature", "top_p")
                    if hasattr(self.agent.model, k)}
        try:
            with evaluation_rng(self.config.seed + self.fabric.global_rank, self.device):
                self.agent.eval()
                for key in sampling:
                    setattr(self.agent.model, key, getattr(self.config, key))
                env.robot_config.reset_noise = None
                env.simulator._push_enabled = False
                yield
        finally:
            env.motion_lib, env.motion_manager = original_lib, original_manager
            self.agent.motion_lib = agent_lib
            env.robot_config.reset_noise = original_noise
            env.simulator._push_enabled = original_push
            for key, value in sampling.items():
                setattr(self.agent.model, key, value)
            self.agent.model.train(was_training)
            random.setstate(py_rng)
            np.random.set_state(np_rng)
            # restore_state can invoke simulator operations: keep their RNG local too.
            with evaluation_rng(self.config.seed, self.device):
                env.restore_state(snapshot)

    def _install_library(self, lib):
        from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
        from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager

        self.env.motion_lib = self.agent.motion_lib = lib
        self.env.motion_manager = MimicMotionManager(
            MimicMotionManagerConfig(init_start_prob=1.0), self.num_envs,
            self.env.dt, self.device, lib,
        )

    def _reset_cohort(self, ids):
        manager = self.env.motion_manager
        # Inactive environments still participate in physics, but never in metrics.
        manager.motion_ids[:] = ids[0]
        manager.motion_ids[:len(ids)] = ids
        manager.motion_times.zero_()
        obs, _ = self.env.reset(sample_flat=True, disable_motion_resample=True)
        return obs

    def _obs(self, obs, n):
        return self.agent.obs_dict_to_tensordict(obs)[:n].clone()

    def _step(self, action, n):
        actions = torch.zeros((self.num_envs, action.shape[-1]), device=self.device)
        actions[:n] = action
        return self.env.step(actions)

    def _collect_offline_batches(self, ids):
        """Fixed expert-state cohort. No prior samples can affect collection RNG."""
        n = len(ids)
        obs = self._reset_cohort(ids)
        batches = []
        model = self.agent.model
        keys = list(dict.fromkeys(model.get_inference_in_keys() + [TARGET_LATENT_KEY]))
        for step in range(self.config.teacher_steps):
            td = model.collect_expert_rollout(self._obs(obs, n))
            action = td["mean_action"].clone()
            if step % self.config.teacher_metric_stride == 0:
                batches.append(td.select(*keys).detach().cpu().clone())
            obs, _, dones, _, _ = self._step(action, n)
            reset_ids = dones[:n].nonzero(as_tuple=True)[0]
            if reset_ids.numel():
                self.env.motion_manager.motion_times[reset_ids] = 0
                obs, _ = self.env.reset(reset_ids, sample_flat=True, disable_motion_resample=True)
        return batches

    def _teacher(self, ids, split="train"):
        if split not in self._offline_batches:
            self._offline_batches[split] = self._collect_offline_batches(ids)
        if self._encoder_digest is None:
            self._encoder_digest = frozen_encoder_digest(self.agent.model)
        return evaluate_prior_offline(
            self.agent.model, self._offline_batches[split], device=self.device,
            seed=self.config.seed + self.fabric.global_rank,
            top_p=self.config.top_p, temperature=self.config.temperature, fabric=self.fabric)

    def _physical_state(self, n):
        state = self.env.simulator.get_robot_state()
        anchor = self.env.robot_config.anchor_body_index
        pos = state.rigid_body_pos[:n, anchor]
        ground = self.env.terrain.get_ground_heights(pos).reshape(-1)
        bad = upright_failure(pos[:, 2] - ground, state.rigid_body_rot[:n, anchor],
                              self.config.min_anchor_height, self.config.min_up_z)
        return state, bad

    def _autonomous(self, ids):
        n = len(ids)
        obs = self._reset_cohort(ids)
        _, initial_bad = self._physical_state(n)
        tracker = SurvivalAccumulator(initial_bad, self.env.dt, self.config.max_eval_steps,
                                      self.config.failure_persistence_steps)
        action_delta_sum = torch.zeros(n, device=self.device)
        action_delta_count = torch.zeros_like(action_delta_sum)
        slip_sum, slip_count = torch.zeros_like(action_delta_sum), torch.zeros_like(action_delta_sum)
        feet = self.env.contact_body_ids
        contact_available = feet.numel() > 0
        previous_action = None
        invalid_episode = torch.zeros(n, device=self.device, dtype=torch.bool)
        survival = {}
        horizon_steps = {math.ceil(t / self.env.dt): t for t in (5, 10, 20)
                         if math.ceil(t / self.env.dt) <= self.config.max_eval_steps}
        for step in range(1, self.config.max_eval_steps + 1):
            # Explicit generate prevents teacher labels from reaching the policy.
            action = self.agent.model.generate(self._obs(obs, n))[0]
            active = tracker.alive.clone()
            finite_action = torch.isfinite(action).all(-1)
            tracker.seconds[active & ~finite_action] = (step - 1) * self.env.dt
            invalid_episode |= ~finite_action
            tracker.alive &= finite_action
            active &= finite_action
            action = torch.nan_to_num(action)
            if previous_action is not None:
                delta = (action - previous_action).square().mean(-1)
                action_delta_sum += torch.where(active, delta, 0)
                action_delta_count += active
            previous_action = action.clone()
            # Continue sampled control after a fall: terminal survival allows recovery.
            # Never replace finite actions with zeros merely because the robot fell.
            obs, _, _, _, _ = self._step(action, n)
            state, bad = self._physical_state(n)
            # Actual simulator contact flags; no reference contact labels/heights.
            contact = state.rigid_body_contacts[:n, feet].bool() & active[:, None]
            speed = state.rigid_body_vel[:n, feet, :2].norm(dim=-1)
            slip_sum += torch.where(contact, speed, 0).sum(-1)
            slip_count += contact.sum(-1)
            tracker.update(bad, step)
            if step in horizon_steps:
                survival[f"upright_survival_{horizon_steps[step]}s"] = float(tracker.alive.float().mean())
        survival.update({
            "survival_rate": float((~bad & ~invalid_episode).float().mean()),
            "never_fell_rate": float(tracker.alive.float().mean()),
            "episode_count": n,
            "upright_survival_horizon": float(tracker.alive.float().mean()),
            "initial_upright_fraction": float((~initial_bad).float().mean()),
            "mean_capped_survival_seconds": float(tracker.seconds.mean()),
            "horizon_seconds": self.config.max_eval_steps * self.env.dt,
            "action_delta_rms": float((action_delta_sum / action_delta_count.clamp_min(1)).sqrt().mean()),
            "contact_sensing_available": float(contact_available),
            "contact_observed_fraction": float((slip_count > 0).float().mean()),
        })
        if contact_available:
            survival["mean_contact_body_xy_speed_mps"] = float((slip_sum / slip_count.clamp_min(1)).mean())
        details = {"terminal_survived": (~bad & ~invalid_episode).cpu().tolist(),
                   "recovery_success_rate": None,
                   "recovery_status": "not_implemented_p1",
                   "motion_ids": ids.cpu().tolist(), "initial_upright": (~initial_bad).cpu().tolist(),
                   "survived_horizon": tracker.alive.cpu().tolist(),
                   "capped_survival_seconds": tracker.seconds.cpu().tolist()}
        return survival, details

    def evaluate_prior_rollout(self, ids):
        """Free autoregressive policy -> frozen decoder -> physics, no mid-run reset."""
        return self._autonomous(ids)

    def _sum(self, tensor):
        return self.fabric.all_reduce(tensor, reduce_op="sum") if self.fabric.world_size > 1 else tensor

    @torch.no_grad()
    def evaluate(self):
        from protomotions.components.motion_lib import MotionLib, MotionLibConfig

        if not self.config.enabled:
            return {}, None, 0
        if self.config.validation_motion_file and self._validation_lib is None:
            self._validation_lib = MotionLib(
                MotionLibConfig(motion_file=self.config.validation_motion_file), device=self.device)
        libraries = {"train": self.agent.motion_lib}
        if self._validation_lib is not None:
            libraries["validation"] = self._validation_lib
        split_audit = {"validation_present": self._validation_lib is not None}
        if self._validation_lib is not None:
            training_names = set(self.agent.motion_lib.motion_files)
            validation_names = set(self._validation_lib.motion_files)
            overlaps = len(training_names & validation_names)
            overlaps = int(self._sum(torch.tensor(overlaps, device=self.device)))
            same_file = Path(self.agent.motion_lib.motion_file).resolve() == Path(self._validation_lib.motion_file).resolve()
            conflicts = int(self._sum(torch.tensor(int(same_file), device=self.device)))
            if overlaps or conflicts:
                raise ValueError("Training/validation clip overlap detected; supply a disjoint clip-level validation library")
            split_audit.update(exact_source_path_overlap=overlaps,
                               limitation="Exact source-path audit cannot detect renamed or duplicated clips")
        rank, world = self.fabric.global_rank, self.fabric.world_size
        cohorts = {}
        for split, lib in libraries.items():
            shared = split == "validation" or not lib.different_motion_files_across_ranks
            cohorts[split] = select_motion_ids(lib.num_motions(), min(self.num_envs, self.config.num_eval_envs),
                                               self.config.seed, rank, world, shared).to(self.device)
        n = min(len(ids) for ids in cohorts.values())
        global_n = int(self._sum(torch.tensor(n, device=self.device)))
        metrics, details, histograms = {}, {}, {}
        ema = getattr(self.agent, "ema", None)
        has_ema = ema is not None and ema.num_updates > 0
        loaded_source = getattr(self.agent, "loaded_inference_weights", "raw")
        primary = self.config.weights if has_ema else loaded_source
        if not has_ema and self.config.compare_raw_ema:
            raise ValueError("Raw/EMA comparison requires a checkpoint with historical EMA weights")
        if not has_ema and self.config.weights != primary:
            log.warning("Requested %s but checkpoint only has %s weights; reporting actual source", self.config.weights, primary)
        sources = [primary]
        if self.config.compare_raw_ema and has_ema:
            sources.append("raw" if primary == "ema" else "ema")
        score_split = "validation" if "validation" in libraries else "train"
        for source in sources:
            source_details = {}
            weight_scope = getattr(self.agent, "evaluation_weights", None)
            with (weight_scope(source) if weight_scope and has_ema else nullcontext()), self._evaluation_scope():
                for split, lib in libraries.items():
                    self._install_library(lib)
                    ids = cohorts[split][:n]
                    values, record = {}, {"motion_ids": ids.cpu().tolist()}
                    log.info("GPC %s/%s: evaluating %d motions", source, split, n)
                    if self.config.mode != "rollout":
                        # Separate RNG scopes make rollout independent of cache misses,
                        # mode selection, label budgets, and raw/EMA comparison order.
                        with evaluation_rng(self.config.seed + rank, self.device):
                            teacher, gt, sampled = self._teacher(ids, split)
                        teacher.update(nll=teacher["val_nll"], token_accuracy=teacher["token_top1_acc"],
                                       top5_accuracy=teacher["token_top5_acc"])
                        histograms[f"{source}_{split}"] = (gt, sampled)
                        values.update({f"prior/{key}": value for key, value in teacher.items()})
                        if source == primary and split == "validation":
                            metrics.update({f"eval/{key}": value for key, value in teacher.items()})
                    if self.config.mode != "offline":
                        with evaluation_rng(self.config.seed + rank, self.device):
                            rollout, record = self.evaluate_prior_rollout(ids)
                        for key, value in rollout.items():
                            total = self._sum(torch.tensor(value * n, dtype=torch.float64, device=self.device))
                            rollout[key] = float(total / global_n)
                        rollout["episode_count"] = global_n
                        values.update({f"rollout/{key}": value for key, value in rollout.items()})
                        if source == primary and split == score_split:
                            metrics.update({f"rollout/{key}": value for key, value in rollout.items()})
                    metrics.update({f"eval_{source}_{split}/{key}": value for key, value in values.items()})
                    if source == primary:
                        metrics.update({f"eval_{split}/{key}": value for key, value in values.items()})
                    source_details[split] = dict(motion_file=lib.motion_file, **record)
            details[source] = source_details
        self.last_evaluation_weights = primary
        score = metrics.get("rollout/survival_rate")
        self.eval_count += 1
        output = Path(self.config.output_dir) if self.config.output_dir else Path(self.root_dir) / "gpc_eval"
        output.mkdir(parents=True, exist_ok=True)
        stem = f"epoch_{self.agent.current_epoch}_eval_{self.eval_count}"
        report = {"checkpoint": getattr(self.agent, "evaluation_checkpoint", None),
                  "agent_epoch": self.agent.current_epoch, "config": asdict(self.config), "rank": rank, "world_size": world,
                  "num_items_per_split": global_n, "score_split": score_split, "score": score,
                  "temperature": self.config.temperature, "top_p": self.config.top_p,
                  "weights": primary, "encoder_sha256": self._encoder_digest, "ema_num_updates": self.agent.ema.num_updates if has_ema else 0,
                  "control_dt": self.env.dt,
                  "simulator": self.env.simulator.config._target_,
                  "resolved_config_directory": str(self.root_dir),
                  "survival_definition": "terminal anchor height above terrain AND local up-Z thresholds; no resets",
                  "offline_definition": "frozen expert states; one generated sequence per state; all grouped tokens valid",
                  "split_audit": split_audit, "metrics": metrics, "cohorts": details[primary], "weight_cohorts": details}
        (output / f"{stem}_rank_{rank}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        # Every rank exports its own reproducible states/labels. Values are tensors only.
        for split, batches in self._offline_batches.items():
            cache_path = output / f"offline_{split}_rank_{rank}.pt"
            if self.eval_count == 1:
                torch.save({"batches": [dict(batch.items()) for batch in batches],
                            "motion_file": libraries[split].motion_file,
                            "motion_ids": cohorts[split][:n].cpu(), "seed": self.config.seed,
                            "encoder_sha256": self._encoder_digest,
                            "group_size": self.agent.model.latent_tokenization.fsq_scalars_per_prior_token}, cache_path)
        if rank == 0:
            with (output / f"{stem}.csv").open("w", newline="") as f:
                writer = csv.writer(f); writer.writerow(["metric", "value"])
                writer.writerows(sorted(metrics.items()))
            for label, (gt, sampled) in histograms.items():
                self._save_histogram(output / f"{stem}_{label}", gt, sampled)
        return metrics, score, n

    def _save_histogram(self, path, gt, sampled):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        groups, vocab = gt.shape
        fig, axes = plt.subplots(groups, 1, figsize=(12, 2.4 * groups), squeeze=False)
        with path.with_suffix(".csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["group", "token", "gt_count", "sample_count", "gt_frequency", "sample_frequency"])
            for j, ax in enumerate(axes[:, 0]):
                p, q = gt[j].double() / gt[j].sum(), sampled[j].double() / sampled[j].sum()
                # Compare the same IDs, selected by combined frequency.
                ids = (p + q).topk(min(self.config.histogram_top_k, vocab)).indices
                x = np.arange(len(ids))
                ax.bar(x - .2, p[ids].numpy(), .4, label="GT")
                ax.bar(x + .2, q[ids].numpy(), .4, label="Sampled")
                ax.set_xticks(x, ids.tolist(), rotation=60)
                ax.set_title(f"Group {j}; K={vocab}; GT N={int(gt[j].sum())}; Sample N={int(sampled[j].sum())}")
                ax.legend()
                for k in ids.tolist():
                    writer.writerow([j, k, int(gt[j, k]), int(sampled[j, k]), float(p[k]), float(q[k])])
        fig.tight_layout(); fig.savefig(path.with_suffix(".png"), dpi=120); plt.close(fig)
