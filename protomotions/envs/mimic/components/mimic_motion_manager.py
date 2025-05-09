from protomotions.envs.base_env.env_utils.general import StepTracker
from protomotions.envs.base_env.components.motion_manager import MotionManager


class MimicMotionManager(MotionManager):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.disable_reset_track = False

        self.reset_track_steps = StepTracker(
            self.env.num_envs,
            min_steps=self.config.reset_track.steps_min,
            max_steps=self.config.reset_track.steps_max,
            device=self.env.device,
        )

    def get_done_tracks(self):
        end_times = (
            self.env.motion_lib.state.motion_lengths[self.motion_ids]
        )
        done_clip = (self.motion_times + self.env.dt) >= end_times
        return done_clip

    def get_has_reset_grace(self):
        return self.reset_track_steps.steps <= self.config.reset_track.grace_period

    def post_physics_step(self):
        self.motion_times += self.env.dt

    def handle_reset_track(self):
        if self.disable_reset_track:
            return
        self.reset_track_steps.advance()

    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        self.reset_track(env_ids)

    def sample_motions(self, env_ids, new_motion_ids=None):
        """
        Reset the motion and scene for a set of environments.
        This method handles the process of resetting the motion and scene for a specified set of environments.
        It ensures that the reset process is correctly handled based on the current configuration.

        Args:
            env_ids (Tensor): Indices of the environments to reset.
            new_motion_ids (Tensor, optional): New motion IDs for the reset environments.
        Returns:
            Tuple[Tensor, Tensor]: New motion IDs and times for the reset environments.
        """
        if self.disable_reset_track:
            return

        super().sample_motions(env_ids, new_motion_ids)

    def reset_track(self, env_ids):
        self.reset_track_steps.reset_steps(env_ids)
