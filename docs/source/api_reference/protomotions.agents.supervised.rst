protomotions.agents.supervised package
======================================

External experts and episode-level mixed rollouts
-------------------------------------------------

``rollout_actor`` selects which model forward produces the rollout: the
student's forward, or the model's own ``collect_expert_rollout``. Providing
``expert_model_path`` loads a frozen external expert that labels every sampled
state as ``expert_actions``. ``expert_rollout_probability`` (default 0) is the
per-episode probability that this expert also drives the environment; each
environment keeps its choice until the episode ends, even across
rollout-buffer boundaries.

Student-controlled episodes execute ``privileged_action`` when the student
provides one, otherwise ``action``. The buffer's ``action`` is the executed raw
action; ``privileged_action`` remains the unmixed posterior prediction. Expert
and student rollout state reset together for completed environments. Inference
uses the model's inference path, not the training selector.

Loading an external expert rejects differing joint orders, controller
parameters, and action transforms/scales, because its labels are regressed in
the student's action space. An experiment must explicitly configure matching
robot and action semantics; the agent does not copy them from a checkpoint.

.. automodule:: protomotions.agents.supervised
   :members:
   :show-inheritance:
   :undoc-members:

Submodules
----------

.. toctree::
   :maxdepth: 4

   protomotions.agents.supervised.agent
   protomotions.agents.supervised.config
   protomotions.agents.supervised.expert_utils
   protomotions.agents.supervised.masked_mimic_config
   protomotions.agents.supervised.masked_mimic_model
