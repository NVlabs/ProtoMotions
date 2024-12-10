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

import torch
import os.path
import yaml
import tempfile

from phys_anim.envs.env_utils.general import StepTracker

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phys_anim.envs.masked_mimic.tasks.user_control.isaacgym import (
        MaskedMimicUserControlHumanoid,
    )
else:
    MaskedMimicUserControlHumanoid = object


class BaseMaskedMimicUserControl(MaskedMimicUserControlHumanoid):  # type: ignore[misc]
    def __init__(self, config, device, *args, **kwargs):
        super().__init__(config, device, *args, **kwargs)
        self.text_command = None

    ###############################################################
    # Environment step logic
    ###############################################################
    def compute_observations(self, env_ids=None):
        if self.text_command is not None:
            self.motion_text_embeddings[:] = self.text_command

        super().compute_observations(env_ids)

    def update_inference_parameters(self):
        # Get the appropriate temporary directory
        tmp_dir = tempfile.gettempdir()
        inference_file_path = os.path.join(
            tmp_dir, "masked_mimic_inference_parameters.yaml"
        )

        # Print the directory and inference file path
        print(f"Temporary directory: {tmp_dir}")
        print(f"Inference file path: {inference_file_path}")

        # Check if the inference file exists and create a default configuration otherwise
        if not os.path.exists(inference_file_path):
            tmp_parameters = {
                "constrained_joints": [
                    {"body_name": "L_Ankle", "constraint_state": 1},
                    {"body_name": "R_Ankle", "constraint_state": 1},
                    {"body_name": "Pelvis", "constraint_state": 1},
                    {"body_name": "Head", "constraint_state": 1},
                    {"body_name": "L_Hand", "constraint_state": 1},
                    {"body_name": "R_Hand", "constraint_state": 1},
                ],
                "inbetweening": False,
                "object_conditioning": False,
                "text_conditioned": False,
                "motion_id": 0,
                "text_command": "a person is walking casually",
            }
            with open(inference_file_path, "w") as f:
                yaml.dump(tmp_parameters, f)

        with open(inference_file_path, "r") as f:
            inference_parameters = yaml.load(f, Loader=yaml.SafeLoader)

        try:
            from transformers import AutoTokenizer, XCLIPTextModel

            model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

            text_command = [inference_parameters["text_command"]]
            with torch.inference_mode():
                inputs = tokenizer(
                    text_command, padding=True, truncation=True, return_tensors="pt"
                )
                outputs = model(**inputs)
                pooled_output = outputs.pooler_output  # pooled (EOS token) states
                self.text_command = pooled_output[0].to(self.device)
                self.motion_text_embeddings[:] = self.text_command
                # override motion lib since we manually provide the text
                self.motion_lib.state.has_text_embeddings[:] = True

            inference_parameters["masked_mimic_repeat_mask_probability"] = 1.0
            inference_parameters["with_conditioning_max_gap_probability"] = 0.0

            if (
                inference_parameters["inbetweening"]
                or inference_parameters["constrained_joints"] is None
            ):
                inference_parameters["time_gap_mask_min_steps"] = 1000
                inference_parameters["time_gap_mask_max_steps"] = 1001
                inference_parameters["masked_mimic_time_gap_probability"] = 1.0
            else:
                inference_parameters["time_gap_mask_min_steps"] = 0
                inference_parameters["time_gap_mask_max_steps"] = 1
                inference_parameters["masked_mimic_time_gap_probability"] = 0.0

            self.config.masked_mimic_masking.joint_masking.masked_mimic_fixed_conditioning = inference_parameters[
                "constrained_joints"
            ]
            self.config.masked_mimic_masking.joint_masking.masked_mimic_time_gap_probability = inference_parameters[
                "masked_mimic_time_gap_probability"
            ]
            self.config.masked_mimic_masking.joint_masking.masked_mimic_repeat_mask_probability = inference_parameters[
                "masked_mimic_repeat_mask_probability"
            ]
            self.config.masked_mimic_masking.joint_masking.time_gap_mask_min_steps = (
                inference_parameters["time_gap_mask_min_steps"]
            )
            self.config.masked_mimic_masking.joint_masking.time_gap_mask_max_steps = (
                inference_parameters["time_gap_mask_max_steps"]
            )
            self.config.masked_mimic_masking.joint_masking.with_conditioning_max_gap_probability = inference_parameters[
                "with_conditioning_max_gap_probability"
            ]
            self.config.masked_mimic_masking.motion_text_embeddings_visible_prob = (
                1.0 if inference_parameters["text_conditioned"] else 0.0
            )
            self.config.masked_mimic_masking.target_pose_visible_prob = (
                1.0 if inference_parameters["inbetweening"] else 0.0
            )

            self.time_gap_mask_steps = StepTracker(
                self.config.num_envs,
                min_steps=self.config.masked_mimic_masking.joint_masking.time_gap_mask_min_steps,
                max_steps=self.config.masked_mimic_masking.joint_masking.time_gap_mask_max_steps,
                device=self.device,
            )
            self.long_term_gap_probs = (
                torch.ones(self.config.num_envs, dtype=torch.float, device=self.device)
                * self.config.masked_mimic_masking.joint_masking.with_conditioning_max_gap_probability
            )
            self.visible_text_embeddings_probs = (
                torch.ones(self.config.num_envs, dtype=torch.float, device=self.device)
                * self.config.masked_mimic_masking.motion_text_embeddings_visible_prob
            )
            self.visible_target_pose_probs = (
                torch.ones(self.config.num_envs, dtype=torch.float, device=self.device)
                * self.config.masked_mimic_masking.target_pose_visible_prob
            )

            self.config.fixed_motion_id = inference_parameters["motion_id"]

            # Let's force it to reset!
            self.motion_times[:] = 1000
            self.reset_track_steps.reset_steps()
            self.progress_buf[:] = 0

        except Exception as e:
            print(f"Error processing inference parameters.")
            print("An easy fix will be to delete the yaml file and try again.")
            print(e)

    def reset_actors(self, env_ids):
        super().reset_actors(env_ids)

        # Make sure in user-control mode that the history isn't visible.
        self.valid_hist_buf.set_all(False, env_ids)

    def reset_track(self, env_ids, new_motion_ids=None):
        super().reset_track(env_ids, new_motion_ids)

        # Make sure in user-control mode that the history isn't visible.
        self.valid_hist_buf.set_all(False, env_ids)
