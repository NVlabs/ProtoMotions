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
from dataclasses import dataclass, field
from typing import List, Optional
from protomotions.agents.common.config import ConfigBuilder


@dataclass
class EvaluatorConfig(ConfigBuilder):
    """Configuration for evaluator."""

    _target_: str = "protomotions.agents.evaluators.base_evaluator.BaseEvaluator"
    eval_metrics_every: Optional[int] = 200


@dataclass
class MimicEvaluatorConfig(EvaluatorConfig):
    """Configuration for Mimic evaluator."""

    _target_: str = "protomotions.agents.evaluators.mimic_evaluator.MimicEvaluator"
    eval_metric_keys: List[str] = field(default_factory=list)
    max_eval_steps: int = 600
    save_predicted_motion_lib_every: Optional[int] = (
        3  # Save pred_motion_lib_epoch_xxx.pt every M evals (None = disabled)
    )
