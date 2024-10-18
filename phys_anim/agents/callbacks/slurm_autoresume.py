# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from typing import Type


# This is useful for autocomplete. Import is ignored at runtime.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phys_anim.agents.ppo import PPO
else:
    PPO = object

try:
    import sys

    sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
    from userlib.auto_resume import AutoResume
except ModuleNotFoundError:
    AutoResume = None


class EmptyCallBack:
    def __init__(self):
        self.details = {}


class SlurmAutoResume(EmptyCallBack):
    def __new__(cls: Type["SlurmAutoResume"], *args, **kwargs):
        if AutoResume is not None:
            return super().__new__(cls, *args, **kwargs)
        else:
            blank = super().__new__(EmptyCallBack)
            blank.details = {}
            return blank

    def __init__(self) -> None:
        AutoResume.init()
        self.details = AutoResume.get_resume_details() or {
            "id": os.environ.get("SLURM_JOB_ID")
        }
        self._autoresume_sent = False

    def _request_autoresume(self, save_path) -> None:
        if not self._autoresume_sent:
            self.details["checkpoint"] = save_path
            AutoResume.request_resume(user_dict=self.details)
            print("Finished requesting autoresume.")

    @property
    def terminate(self) -> bool:
        return AutoResume.termination_requested()

    def _check_autoresume(self, agent: PPO) -> None:
        # Only rank 0 should check and request auto-resume. It should then update the rest.
        if self.terminate:
            print("Should autoresume!")
            agent.save()
            if agent.fabric.global_rank == 0:
                self._request_autoresume(
                    f"{agent.fabric.loggers[0].root_dir}/last.ckpt"
                )
            agent._should_stop = True
            self._autoresume_sent = True

    def on_fit_start(self, agent: PPO) -> None:
        if agent.fabric.global_rank == 0:
            self._request_autoresume(f"{agent.fabric.loggers[0].root_dir}/last.ckpt")
        self._autoresume_sent = True

    def before_play_steps(self, agent: PPO) -> None:
        self._check_autoresume(agent)

    def on_fit_end(self, agent: PPO) -> None:
        if agent.fabric.global_rank == 0:
            AutoResume.stop_resuming()
