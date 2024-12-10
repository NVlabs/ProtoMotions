import logging
import time

import wandb

from pytorch_lightning import Callback, LightningModule, Trainer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phys_anim.agents.ppo import PPO
else:
    PPO = object

log = logging.getLogger(__name__)


def wandb_run_exists():
    return isinstance(wandb.run, wandb.sdk.wandb_run.Run)


class AutoResumeCallbackSrun(Callback):
    def __init__(self, autoresume_after=12600) -> None:
        self.start_time = time.time()
        self.autoresume_after = autoresume_after

        print("************************************")
        print("will autoresume after ", self.autoresume_after)

    def _check_autoresume(self, agent: PPO):
        # agent.fabric.strategy.barrier()
        if time.time() - self.start_time >= self.autoresume_after:

            log.info("Should autoresume!")

            agent.save()

            # if agent.fabric.global_rank == 0:
            #     wandb_id = wandb.run.id if wandb_run_exists() else "",
            #     message = f"Terminating... wandb_id: {wandb_id}"
            #     log.info(message)
            # print(f"[Auto Resume] Rank {agent.fabric.global_rank} exiting.", flush=True)

            agent._should_stop = True
            log.info(f"should stop, {agent.should_stop}")
            # agent.fabric.strategy.barrier()

    def before_play_steps(self, agent: PPO) -> None:
        self._check_autoresume(agent)

    def on_fit_start(self, agent: PPO) -> None:
        pass

    def on_fit_end(self, agent: PPO) -> None:
        pass


# 
# class AutoResumeCallbackSrun(Callback):
#     def __init__(self, autoresume_after=12600) -> None:
#         self.start_time = time.time()
#         self.exit_signal_received = False

#         # Register the signal handler for SIGUSR2 (signal 12)
#         signal.signal(signal.SIGUSR2, self._handle_exit_signal)

#     def _handle_exit_signal(self, signum, frame):
#         """
#         Signal handler for SIGUSR2 (signal 12).
#         """
#         log.info("Received signal %d, initiating save and exit.", signum)
#         self.exit_signal_received = True

#     def _check_autoresume(self, agent: PPO):
#         # Check if signal has been received
#         if self.exit_signal_received:
#             log.info("Signal received, saving agent state and stopping.")
#             agent.save()

#             # Set the stop flag
#             agent._should_stop = True
#             log.info(f"should stop, {agent.should_stop}")
#             return

#     def before_play_steps(self, agent: PPO) -> None:
#         self._check_autoresume(agent)

#     def on_fit_start(self, agent: PPO) -> None:
#         pass

#     def on_fit_end(self, agent: PPO) -> None:
#         pass