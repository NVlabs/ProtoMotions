from pytorch_lightning import LightningModule
from protomotions.utils.device_dtype_mixin import DeviceDtypeModuleMixin


class RL_EvalCallback(DeviceDtypeModuleMixin):
    def __init__(self, config, training_loop: LightningModule):
        super().__init__()
        self.config = config
        self.training_loop = training_loop
        self.to(self.training_loop.device)

    def on_pre_evaluate_policy(self):
        pass

    def on_pre_eval_env_step(self, actor_state):
        return actor_state

    def on_post_eval_env_step(self, actor_state, done_indices):
        return actor_state

    def on_post_evaluate_policy(self):
        pass
