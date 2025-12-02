# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
import numpy as np


def build_pd_action_offset_scale(
    hinge_axes_map, dof_limits_lower, dof_limits_upper, action_scale, device
):
    sorted_body_ids = list(hinge_axes_map.keys())
    sorted_body_ids.sort()

    lim_low = dof_limits_lower.cpu().numpy()
    lim_high = dof_limits_upper.cpu().numpy()

    dof_offset = 0

    for body_id in sorted_body_ids:
        dof_size = len(hinge_axes_map[body_id])

        if dof_size == 3:
            curr_low = lim_low[dof_offset : (dof_offset + dof_size)]
            curr_high = lim_high[dof_offset : (dof_offset + dof_size)]
            curr_low = np.max(np.abs(curr_low))
            curr_high = np.max(np.abs(curr_high))
            curr_scale = max([curr_low, curr_high])
            curr_scale = 2 * action_scale * curr_scale
            curr_scale = min([curr_scale, np.pi])

            lim_low[dof_offset : (dof_offset + dof_size)] = -curr_scale
            lim_high[dof_offset : (dof_offset + dof_size)] = curr_scale

        elif dof_size == 1:
            curr_low = lim_low[dof_offset]
            curr_high = lim_high[dof_offset]
            curr_mid = 0.5 * (curr_high + curr_low)

            # extend the action range to be a bit beyond the joint limits so that the motors
            # don't lose their strength as they approach the joint limits
            curr_scale = action_scale * (curr_high - curr_low)
            curr_low = curr_mid - curr_scale
            curr_high = curr_mid + curr_scale

            lim_low[dof_offset] = curr_low
            lim_high[dof_offset] = curr_high

        else:
            raise ValueError(f"Invalid dof size: {dof_size}")

        dof_offset += dof_size

    pd_action_offset = 0.5 * (lim_high + lim_low)
    pd_action_scale = 0.5 * (lim_high - lim_low)
    pd_action_offset = torch.tensor(pd_action_offset, device=device)
    pd_action_scale = torch.tensor(pd_action_scale, device=device)

    return pd_action_offset, pd_action_scale
