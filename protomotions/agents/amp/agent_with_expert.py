import torch
import logging
import time
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
from rich.progress import track

from protomotions.agents.amp.agent import AMP
from protomotions.agents.ppo.model import PPOModel
from protomotions.envs.base_env.env import BaseEnv
from lightning.fabric import Fabric
from protomotions.agents.utils.data_utils import ExperienceBuffer
from protomotions.agents.ppo.utils import discount_values

log = logging.getLogger(__name__)


class AMPWithExpert(AMP):
    """
    AMP agent that can load expert models from masked_mimic checkpoints.
    This combines AMP functionality with expert model loading capabilities.
    """
    
    def __init__(self, fabric: Fabric, env: BaseEnv, config):
        super().__init__(fabric, env, config)
        self.expert_model = None
        self.expert_model_config = None

    def setup(self):
        # Call parent setup first
        super().setup()
        
        # Optionally load a pre-trained expert model if provided
        if self.config.expert_model_path is not None:
            print(
                "Loading pre-trained expert model from:",
                self.config.expert_model_path,
            )
            # "score_based.ckpt" is the name of the file that is saved when a new high score is achieved
            checkpoint_path = self.config.expert_model_path + "/score_based.ckpt"
            if not Path(checkpoint_path).exists():
                checkpoint_path = self.config.expert_model_path + "/last.ckpt"

            self.expert_model_config = OmegaConf.load(
                Path(self.config.expert_model_path) / "config.yaml"
            )
            
            # Load the expert model
            expert_model: PPOModel = instantiate(
                self.expert_model_config.agent.config.model
            )
            self.expert_model = self.fabric.setup(expert_model)
            self.expert_model.mark_forward_method("act")

            # loading should be done after fabric.setup to ensure the model is on the correct fabric.device
            pre_trained_expert = torch.load(checkpoint_path, map_location=self.fabric.device) 
            self.expert_model.load_state_dict(pre_trained_expert["model"])
            for param in self.expert_model.parameters():
                param.requires_grad = False
            self.expert_model.eval()  # Just in case
        else:
            self.expert_model = None

    def register_extra_experience_buffer_keys(self):
        super().register_extra_experience_buffer_keys()
        if self.expert_model is not None:
            self.experience_buffer.register_key(
                "expert_actions", shape=(self.env.config.robot.number_of_actions,)
            )

    def env_step(self, actions):
        """Perform an environment step."""
        obs, rewards, dones, terminated, extras = super().env_step(actions)
        return obs, rewards, dones, terminated, extras

    def fit(self):
        """Override fit to collect expert actions during training."""
        # Setup experience buffer
        self.experience_buffer = ExperienceBuffer(self.num_envs, self.num_steps).to(
            self.device
        )
        self.experience_buffer.register_key(
            "self_obs", shape=(self.env.config.robot.self_obs_size,)
        )
        self.experience_buffer.register_key(
            "actions", shape=(self.env.config.robot.number_of_actions,)
        )
        self.experience_buffer.register_key("rewards")
        self.experience_buffer.register_key("extra_rewards")
        self.experience_buffer.register_key("total_rewards")
        self.experience_buffer.register_key("dones", dtype=torch.long)
        self.experience_buffer.register_key("values")
        self.experience_buffer.register_key("next_values")
        self.experience_buffer.register_key("returns")
        self.experience_buffer.register_key("advantages")
        self.experience_buffer.register_key("neglogp")
        self.register_extra_experience_buffer_keys()

        if self.config.get("extra_inputs", None) is not None:
            obs = self.env.get_obs()
            for key in self.config.extra_inputs.keys():
                assert (
                    key in obs
                ), f"Key {key} not found in obs returned from env: {obs.keys()}"
                env_tensor = obs[key]
                shape = env_tensor.shape
                dtype = env_tensor.dtype
                self.experience_buffer.register_key(key, shape=shape[1:], dtype=dtype)

        # Force reset on fit start
        done_indices = None
        if self.fit_start_time is None:
            self.fit_start_time = time.time()
        self.fabric.call("on_fit_start", self)

        while self.current_epoch < self.config.max_epochs:
            self.epoch_start_time = time.time()

            # Set networks in eval mode so that normalizers are not updated
            self.eval()
            with torch.no_grad():
                self.fabric.call("before_play_steps", self)

                for step in track(
                    range(self.num_steps),
                    description=f"Epoch {self.current_epoch}, collecting data...",
                ):
                    obs = self.handle_reset(done_indices)
                    self.experience_buffer.update_data("self_obs", step, obs["self_obs"])
                    if self.config.get("extra_inputs", None) is not None:
                        for key in self.config.extra_inputs:
                            self.experience_buffer.update_data(key, step, obs[key])

                    action, neglogp, value = self.model.get_action_and_value(obs)
                    self.experience_buffer.update_data("actions", step, action)
                    self.experience_buffer.update_data("neglogp", step, neglogp)
                    if self.config.normalize_values:
                        value = self.running_val_norm.normalize(value, un_norm=True)
                    self.experience_buffer.update_data("values", step, value)

                    # Collect expert actions if expert model is available
                    if self.expert_model is not None:
                        # Add VAE noise for masked_mimic expert model
                        expert_obs = obs.copy()
                        if "vae_noise" not in expert_obs:
                            # Generate VAE noise for expert model
                            vae_latent_dim = self.expert_model_config.agent.config.model.config.vae_latent_dim
                            expert_obs["vae_noise"] = torch.randn(
                                self.num_envs, vae_latent_dim, device=self.device
                            )
                        expert_action = self.expert_model.act(expert_obs)
                        self.experience_buffer.update_data("expert_actions", step, expert_action)

                    # Check for NaNs in observations and actions
                    for key in obs.keys():
                        if torch.isnan(obs[key]).any():
                            print(f"NaN in {key}: {obs[key]}")
                            raise ValueError("NaN in obs")
                    if torch.isnan(action).any():
                        raise ValueError(f"NaN in action: {action}")

                    # Step the environment
                    next_obs, rewards, dones, terminated, extras = self.env_step(action)

                    all_done_indices = dones.nonzero(as_tuple=False)
                    done_indices = all_done_indices.squeeze(-1)

                    # Update logging metrics with the environment feedback
                    self.post_train_env_step(rewards, dones, done_indices, extras, step)

                    self.experience_buffer.update_data("rewards", step, rewards)
                    self.experience_buffer.update_data("dones", step, dones)

                    next_value = self.model._critic(next_obs).flatten()
                    if self.config.normalize_values:
                        next_value = self.running_val_norm.normalize(
                            next_value, un_norm=True
                        )
                    next_value = next_value * (1 - terminated.float())
                    self.experience_buffer.update_data("next_values", step, next_value)

                    self.step_count += self.get_step_count_increment()

                # After data collection, compute rewards, advantages, and returns.
                rewards = self.experience_buffer.rewards
                extra_rewards = self.calculate_extra_reward()
                self.experience_buffer.batch_update_data("extra_rewards", extra_rewards)
                total_rewards = rewards + extra_rewards
                self.experience_buffer.batch_update_data("total_rewards", total_rewards)

                advantages = discount_values(
                    self.experience_buffer.dones,
                    self.experience_buffer.values,
                    total_rewards,
                    self.experience_buffer.next_values,
                    self.gamma,
                    self.tau,
                )
                returns = advantages + self.experience_buffer.values
                self.experience_buffer.batch_update_data("returns", returns)

                if self.config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                self.experience_buffer.batch_update_data("advantages", advantages)

            training_log_dict = self.optimize_model()
            training_log_dict["epoch"] = self.current_epoch
            self.current_epoch += 1
            self.fabric.call("after_train", self)

            # Save model checkpoint at specified intervals before evaluation.
            if self.current_epoch % self.config.manual_save_every == 0:
                self.save()

            if (
                self.config.eval_metrics_every is not None
                and self.current_epoch > 0
                and self.current_epoch % self.config.eval_metrics_every == 0
            ):
                eval_log_dict, evaluated_score = self.calc_eval_metrics()
                evaluated_score = self.fabric.broadcast(evaluated_score, src=0)
                if evaluated_score is not None:
                    if (
                        self.best_evaluated_score is None
                        or evaluated_score >= self.best_evaluated_score
                    ):
                        self.best_evaluated_score = evaluated_score
                        self.save(new_high_score=True)
                training_log_dict.update(eval_log_dict)

            self.post_epoch_logging(training_log_dict)
            self.env.on_epoch_end(self.current_epoch)

            if self.should_stop:
                self.save()
                return

        self.time_report.report()
        self.save()
        self.fabric.call("on_fit_end", self) 