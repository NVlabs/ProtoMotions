class BaseCallback(object):
    def __init__(self, config, env):
        self.config = config
        self.env = env

    # These are the supported callback hooks:

    # def get_obs(self) -> Dict[str, Tensor]:  # This function returns a dictionary of observations. One or many.
    # def compute_reward(self) -> Tensor:  # Returns a reward tensor.

    # These functions don't return a thing, but are used to update the callback's internal state.
    # def pre_physics_step(self):
    # def post_physics_step(self):
    # def compute_observations(self, env_ids):
    # def reset_envs(self, env_ids):
