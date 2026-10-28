Evaluation components
=====================

An experiment defines what to measure and what counts as failure. The evaluator
runs trials and aggregates their results. Evaluation does not automatically use
the environment's termination flags: ending a training episode and failing a
benchmark can intentionally have different rules.

Configure measurements and failure conditions
-------------------------------------------------

Measurements use ordinary ``MdpComponent`` functions and context bindings, just
like rewards and observations. Optional ``ThresholdFailure`` rules specify how
to reduce those measurements over a trial and which side of a threshold fails.

.. code-block:: python

   from protomotions.agents.evaluators.config import MimicEvaluatorConfig
   from protomotions.agents.evaluators.components import ThresholdFailure
   from protomotions.envs.mdp_component import MdpComponent
   from protomotions.envs.terminations.tracking import mean_body_pos_error

   position_error = MdpComponent.from_kernel(mean_body_pos_error)
   agent_cfg.evaluator = MimicEvaluatorConfig(
       metric_components={"gt_error": position_error},
       failure_components={
           "gt_error": ThresholdFailure(position_error, 0.5),
       },
       score_key="eval/success_rate",
   )

``metric_components`` return one real value per environment and log
``eval/{name}/mean``, ``min`` and ``max``. Boolean metrics log the fraction of
scored steps where the condition was true. Means average each trial's mean,
so longer trials do not receive more weight.

Each entry in ``failure_components`` is a ``ThresholdFailure`` over a scalar
measurement per environment. Its default ``reduction="any"`` preserves
any-frame failure, even if the agent recovers later. Each rule also logs
``eval/{name}/failure_rate``. Comparisons are strict ``>`` by default, or strict
``<`` with ``fail_above=False``; equality passes.

An existing boolean task or termination component can be wrapped with
``ThresholdFailure(component, 0.5)``. Share a component with training only when its geometry,
threshold and warm-up rules really match. For example, the path-following recipe
keeps its 0.5 m evaluation limit separate from its 1.0 m training termination.
Task-specific guards belong in the predicate, bound to progress or control state.

Set ``score_key="eval/success_rate"`` to use success for best-checkpoint selection.
Set ``score_key=None`` for diagnostics without selecting checkpoints. A success
score requires at least one failure component; empty or unmeasured trials do not
produce a perfect score. Empty metric and failure groups disable evaluation.
Other score keys are not supported yet.
Metrics-only runs still save regular checkpoints on the normal training cadence;
they do not select a best-scoring checkpoint or change motion sampling weights.

Temporal rules
--------------

``reduction`` can be ``any``, ``sum``, ``mean``, ``min``, ``max`` or ``last``.
All operate on the scored frames of one trial. ``any`` marks failure immediately.
The other rules are judged at the end of the window: an unachieved goal may still
be achieved later. No full trajectory is stored for these reductions.

.. code-block:: python

   # height is an MdpComponent returning one height per environment.
   evaluator.metric_components = {"height": height}  # logging, no threshold
   evaluator.failure_components = {
       # Fail if height never reaches 0.8 m during the scored window.
       "never_upright": ThresholdFailure(
           height, 0.8, fail_above=False, reduction="max"
       ),
       # Optionally require it at the final scored frame too.
       "not_upright_at_end": ThresholdFailure(
           height, 0.8, fail_above=False, reduction="last"
       ),
   }

These are illustrative height tests, not a complete recovery benchmark. Stable
get-up requires an appropriate upright/contact/hold criterion. ``sum`` counts
samples; a duration metric must return ``condition.float() * dt`` to sum seconds.
A continuous hold requires task state, not merely a total duration.

Shared responsibilities
-----------------------

``EvaluationManager`` extends the existing ``ComponentManager`` with trial
accumulation and reporting. It builds routes once and computes a shared component
only once per frame when multiple rules or logs reuse that component object.
Create a measurement once and reuse that same object in ``metric_components``
and ``ThresholdFailure``; separate component objects are separate computations.
Counts are shared across measurements; no per-step host synchronization is added
by the reductions.

``BaseEvaluator`` owns the single rollout and save/restore loop.
``MimicEvaluator`` adds motion coverage, valid-frame selection and motion-start
resets. ``MaskedMimicEvaluator`` overrides only action selection. No subclass
duplicates the rollout or metric machinery.

Rollout and training behavior
----------------------------------

* ``BaseEvaluator`` scores up to ``max_eval_steps`` per environment. A failed
  trial ends at its failing frame; its environment resets before another action
  but remains unscored for the rest of the batch. Reset includes model rollout
  context. A training termination flag alone does not fail an evaluation trial.
* ``MimicEvaluator`` sweeps the motion library and scores each motion's valid
  frames. Scene-paired deduplication and background-environment resets are
  unchanged.
* ``MaskedMimicEvaluatorConfig.action_source`` uses the enum
  ``MaskedMimicActionSource.POSTERIOR`` or ``MaskedMimicActionSource.PRIOR``.
  CLI overrides accept ``posterior`` and ``prior``. Regenerate saved configs
  that used string-valued action sources. Training configs use the student posterior; inference overrides
  select the prior-only ``forward_deployable`` path. Both run in model eval mode;
  that mode does not choose the action source. Offline reference evaluation can
  explicitly select the posterior too.
* A batch stops as soon as no scored trials remain. Motion evaluation then moves
  to the next batch, so every motion still gets evaluated. No extra reset is
  needed on the last step: the next batch resets, or evaluation restores the
  saved training state. Unscored background environments restart when done.
* Measurements and trajectories include the failing frame, but no later frames.
  This intentionally changes diagnostic averages from the earlier full-window
  statistics; the configured failure thresholds are unchanged. Action smoothness
  and exports use only recorded frames, not the unused tail of the buffer.
* Evaluation restores environment state, model rollout context, mixed module
  train/eval modes and temporary perturbation settings, including on errors.
* ``evaluate()`` returns an ``EvaluationResult`` with logs, score, count and trial
  outcomes, after restoring the environment. The training loop then calls
  ``update_training_state(result)``. Only evaluated motions receive weight
  updates. Offline evaluation does not make that call. Trajectory exports use
  recorded frame counts, not unwritten storage capacity.
* Distributed aggregation weights results by evaluated trials. Empty ranks have
  zero weight, and different tasks may return different metric keys. The existing
  weighted averaging of rank-local statistics is retained; ``num_evaluated`` is
  summed across ranks.

Component snapshots
-------------------

Environment components use ``save_state()`` and ``restore_state(snapshot)`` for
temporary evaluation snapshots. The environment composes these snapshots;
components own their contents. This is separate from model checkpoint loading.

Control components allocate their per-env tensors through
``self.state.add(name, shape, dtype, reset_value)`` on their ``EpisodeTensors``
registry. The registry clones them in ``save_state()`` and restores them in place
in ``load_state(snapshot)``, so existing views remain valid, and resets the rows
the environment asks it to. Constants go through ``self.state.constant`` and are
not snapshotted; a nested object such as a target stream or a path generator
holds a child registry (``self.state.child``) that the parent walks. There is no
automatic attribute scan: a bare tensor attribute with a per-env first dimension
is refused at construction.

Simulator ``save_state()`` covers action history and temporal buffers. The
environment still saves physical poses and backend solver state separately,
restoring solver state after the physical reset. Global random-generator state
is not captured, so future random draws can differ from uninterrupted training.

Configuration migration
-----------------------

Regenerate old resolved configs from the updated experiment file. There is no
automatic migration or parallel compatibility path. Move numeric entries
from ``evaluation_components`` to ``metric_components`` and express each old
threshold as ``ThresholdFailure(metric, threshold, fail_above=...)``. Keep the
intended units, comparisons and scoring window. Numeric metric kernels take
no ``threshold`` argument.

Reapply any original CLI overrides when regenerating configs. Resume loads saved
configs, so editing an experiment file alone does not migrate an existing run.
Use a new experiment name and a weights-only warm start when changing benchmark
semantics, rather than comparing its scores against the old best checkpoint.
Pass ``--checkpoint`` with the new experiment name to load weights into freshly
built configs; do not resume the old run.
The evaluator does not infer migrations or automatically reset best scores.

Configuration errors, binding errors and failures inside metric or failure
components propagate. Evaluation restores the captured training state in
``finally`` without suppressing the original error. Aggregate plugins are the
exception: ``SmoothnessCalculator`` catches ``RuntimeError``, ``ValueError`` and
``IndexError`` from the jerk computation, logs a warning and returns the metrics
it had already computed. It also reports zero jerk metrics when no motion has
enough valid frames, and no smoothness metrics when the trajectory channel was
not collected. Treat a missing or zero smoothness entry as a diagnostic gap,
not as a perfect score.

Ordinary policy action selection is unchanged. MaskedMimic recipes explicitly
switch training evaluation from the earlier prior action to the supervised posterior.
These scores must not be compared to prior-based best scores: use a weights-only
warm start when adopting that change. Recreated configs select their action
source explicitly rather than inferring it from a historical checkpoint.
