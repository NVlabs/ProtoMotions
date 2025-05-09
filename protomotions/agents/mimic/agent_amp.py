import torch
import logging
from pathlib import Path

from rich.progress import track

from typing import List, Tuple, Dict, Optional

from protomotions.envs.mimic.env import Mimic as MimicEnv
from protomotions.agents.amp.agent import AMP

log = logging.getLogger(__name__)


class Mimic(AMP):
    env: MimicEnv

    # -----------------------------
    # Motion Mapping and Data Distribution
    # -----------------------------
    def map_motions_to_iterations(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Maps motion IDs to iterations for distributed processing.

        Distributes motion IDs across available ranks and creates a mapping of motions 
        to be processed in each iteration. Ensures equal distribution and proper scene sampling.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples where each tuple contains:
                - motion_ids: Tensor of motion IDs for an iteration.
                - requires_scene: Tensor of booleans indicating if the motion requires a scene.
        """
        world_size = self.fabric.world_size
        global_rank = self.fabric.global_rank
        num_motions = self.motion_lib.num_motions()

        if self.env.config.motion_manager.fixed_motion_id is not None:
            motion_ids = torch.tensor(
                [self.env.config.motion_manager.fixed_motion_id], device=self.device
            )
            requires_scene = self.env.get_motion_requires_scene(motion_ids)
            return [(motion_ids, requires_scene)], 1

        base_motions_per_rank = num_motions // world_size
        extra_motions = num_motions % world_size
        motions_per_rank = base_motions_per_rank + (1 if global_rank < extra_motions else 0)
        start_motion = base_motions_per_rank * global_rank + min(global_rank, extra_motions)
        end_motion = start_motion + motions_per_rank
        motion_range = torch.arange(start_motion, end_motion, device=self.device)

        motion_map = []
        for i in range(0, len(motion_range), self.num_envs):
            batch_motion_ids = motion_range[i : i + self.num_envs]
            requires_scene = self.env.get_motion_requires_scene(batch_motion_ids)
            motion_map.append((batch_motion_ids, requires_scene))

        return motion_map, motions_per_rank

    # -----------------------------
    # Evaluation and Metrics Collection
    # -----------------------------
    @torch.no_grad()
    def calc_eval_metrics(self) -> Tuple[Dict, Optional[float]]:
        self.eval()
        if self.env.config.motion_manager.fixed_motion_id is not None:
            num_motions = 1
        else:
            num_motions = self.motion_lib.num_motions()

        metrics = {
            "evaluated": torch.zeros(num_motions, device=self.device, dtype=torch.bool)
        }
        for k in self.config.eval_metric_keys:
            metrics[k] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_max"] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_min"] = 3 * torch.ones(num_motions, device=self.device)

        root_dir = Path(self.fabric.loggers[0].root_dir)
        motion_map, remaining_motions = self.map_motions_to_iterations()
        num_outer_iters = len(motion_map)
        max_iters = max(self.fabric.all_gather(len(motion_map)))

        for outer_iter in track(
            range(max_iters),
            description=f"Evaluating... {remaining_motions} motions remain..."
        ):
            motion_pointer = outer_iter % num_outer_iters
            motion_ids, requires_scene = motion_map[motion_pointer]
            num_motions_this_iter = len(motion_ids)
            metrics["evaluated"][motion_ids] = True

            self.env.agent_in_scene[:] = False
            self.env.motion_manager.motion_ids[:num_motions_this_iter] = motion_ids
            self.env.agent_in_scene[:num_motions_this_iter] = requires_scene
            self.env.force_respawn_on_flat = True

            env_ids = torch.arange(0, num_motions_this_iter, dtype=torch.long, device=self.device)
            dt: float = self.env.dt
            motion_lengths = self.motion_lib.get_motion_length(motion_ids)
            motion_num_frames = (motion_lengths / dt).floor().long()
            max_len = (
                motion_num_frames.max().item()
                if self.config.eval_length is None
                else self.config.eval_length
            )

            for eval_episode in range(self.config.eval_num_episodes):
                elapsed_time = torch.rand_like(self.motion_lib.state.motion_lengths[motion_ids]) * dt
                self.env.motion_manager.motion_times[:num_motions_this_iter] = elapsed_time
                self.env.motion_manager.reset_track_steps.reset_steps(env_ids)
                self.env.disable_reset = True
                self.env.motion_manager.disable_reset_track = True

                obs = self.env.reset(
                    torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
                )

                for l in range(max_len):
                    actions = self.model.act(obs)
                    obs, rewards, dones, terminated, extras = self.env_step(actions)
                    elapsed_time += dt
                    clip_done = (motion_lengths - dt) < elapsed_time
                    clip_not_done = torch.logical_not(clip_done)
                    for k in self.config.eval_metric_keys:
                        if k in self.env.mimic_info_dict:
                            value = self.env.mimic_info_dict[k].detach()
                        else:
                            raise ValueError(f"Key {k} not found in mimic_info_dict")
                        metric = value[:num_motions_this_iter]
                        metrics[k][motion_ids[clip_not_done]] += metric[clip_not_done]
                        metrics[f"{k}_max"][motion_ids[clip_not_done]] = torch.maximum(
                            metrics[f"{k}_max"][motion_ids[clip_not_done]],
                            metric[clip_not_done]
                        )
                        metrics[f"{k}_min"][motion_ids[clip_not_done]] = torch.minimum(
                            metrics[f"{k}_min"][motion_ids[clip_not_done]],
                            metric[clip_not_done]
                        )

        print("Evaluation done, now aggregating data.")
        if self.env.config.motion_manager.fixed_motion_id is None:
            motion_lengths = self.motion_lib.state.motion_lengths[:]
            motion_num_frames = (motion_lengths / dt).floor().long()

        # Save metrics per rank; distributed all_gather does not support dictionaries.
        with open(root_dir / f"{self.fabric.global_rank}_metrics.pt", "wb") as f:
            torch.save(metrics, f)
        self.fabric.barrier()
        # All ranks aggregrate data from all ranks.
        for rank in range(self.fabric.world_size):
            with open(root_dir / f"{rank}_metrics.pt", "rb") as f:
                other_metrics = torch.load(f, map_location=self.device)
            other_evaluated_indices = torch.nonzero(other_metrics["evaluated"]).flatten()
            for k in other_metrics.keys():
                metrics[k][other_evaluated_indices] = other_metrics[k][other_evaluated_indices]
            metrics["evaluated"][other_evaluated_indices] = True

        assert metrics["evaluated"].all(), "Not all motions were evaluated."
        self.fabric.barrier()
        (root_dir / f"{self.fabric.global_rank}_metrics.pt").unlink()

        to_log = {}
        for k in self.config.eval_metric_keys:
            mean_tracking_errors = metrics[k] / (motion_num_frames * self.config.eval_num_episodes)
            to_log[f"eval/{k}"] = mean_tracking_errors.detach().mean().item()
            to_log[f"eval/{k}_max"] = metrics[f"{k}_max"].detach().mean().item()
            to_log[f"eval/{k}_min"] = metrics[f"{k}_min"].detach().mean().item()

        if "gt_err" in self.config.eval_metric_keys:
            tracking_failures = (metrics["gt_err_max"] > 0.5).float()
            to_log["eval/tracking_success_rate"] = 1.0 - tracking_failures.detach().mean().item()

            failed_motions = torch.nonzero(tracking_failures).flatten().tolist()
            print(f"Saving to: {root_dir / f'failed_motions_{self.fabric.global_rank}.txt'}")
            with open(root_dir / f"failed_motions_{self.fabric.global_rank}.txt", "w") as f:
                for motion_id in failed_motions:
                    f.write(f"{motion_id}\n")
                    
            new_weights = torch.ones(self.motion_lib.num_motions(), device=self.device) * 1e-4
            new_weights[failed_motions] = 1.0
            self.env.motion_manager.update_sampling_weights(new_weights)

        stop_early = (
            self.config.training_early_termination.early_terminate_cart_err is not None
            or self.config.training_early_termination.early_terminate_success_rate is not None
        ) and self.fabric.global_rank == 0

        if self.config.training_early_termination.early_terminate_cart_err is not None:
            cart_err = to_log["eval/cartesian_err"]
            stop_early = stop_early and (cart_err <= self.config.training_early_termination.early_terminate_cart_err)
        if self.config.training_early_termination.early_terminate_success_rate is not None:
            tracking_success_rate = to_log["eval/tracking_success_rate"]
            stop_early = stop_early and (tracking_success_rate >= self.config.training_early_termination.early_terminate_success_rate)

        if stop_early:
            print("Stopping early! Target error reached")
            if "tracking_success_rate" in self.config.eval_metric_keys:
                print(f"tracking_success_rate: {to_log['eval/tracking_success_rate']}")
            if "cartesian_err" in self.config.eval_metric_keys:
                print(f"cartesian_err: {to_log['eval/cartesian_err']}")
            evaluated_score = self.fabric.broadcast(to_log["eval/tracking_success_rate"], src=0)
            self.best_evaluated_score = evaluated_score
            self.save(new_high_score=True)
            self.terminate_early()

        self.env.disable_reset = False
        self.env.motion_manager.disable_reset_track = False
        self.env.force_respawn_on_flat = False
        all_ids = torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
        self.env.motion_manager.reset_envs(all_ids)
        self.force_full_restart = True
        return to_log, to_log.get("eval/tracking_success_rate", to_log.get("eval/cartesian_err", None))
