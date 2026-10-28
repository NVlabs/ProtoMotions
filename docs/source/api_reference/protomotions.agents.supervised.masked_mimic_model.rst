protomotions.agents.supervised.masked_mimic_model module
========================================================

Latent decoding
---------------

The prior outputs its Gaussian mean and log-variance. The encoder outputs a
mean residual (added to the prior mean) and the posterior log-variance.
The decoder receives the sampled latent and its configured observations.
Inference runs only the prior and decoder; it never requires encoder inputs.

``MaskedMimicVAEConfig(latent_geometry=LatentGeometry.SPHERE)`` projects sampled prior and
posterior latents to unit length before decoding. The Gaussian parameters remain
unprojected for the KL regularizer; this is not a KL between spherical densities.
An exactly zero sample maps to the first unit basis vector. Gaussian sampling
remains the default. ``prior_mean_unit_norm_coeff`` optionally penalizes
``(||prior_mean|| - 1)^2`` for spherical models.

Decoder observation statistics are recorded once on the posterior pass during
training, not twice for the prior and posterior. Deterministic goal features must
be computed from deployable inputs; masking and token construction belong in the
configured prior modules. The model does not infer or reconstruct hidden goals.

.. automodule:: protomotions.agents.supervised.masked_mimic_model
   :members:
   :show-inheritance:
   :undoc-members:

Temporal objectives
-------------------

``temporal_losses`` lists ``TemporalLossConfig`` entries. Each compares a
declared model output (``prediction_key``, default ``key``) against the
rollout value of ``key`` at the previous step, or at the next step when
``forward=True``. Targets are held fixed during optimization. For the Metatron
recipe, use ``TemporalLossConfig("privileged_action", 0.05)``,
``TemporalLossConfig("solution_features", 0.001)``. These previous-step
terms do not compare against the action selected by an expert-driven rollout.
A next-step objective is also supported, but requires the model to produce
the configured prediction key; it is not enabled in this recipe. All
terms average squared feature errors over valid transitions, with no timestep
scaling, and are logged as ``model/temporal_{previous|next}_{prediction_key}``.

The model's ``build_training_targets`` hook constructs targets on time-by-env
tensors before flattening and shuffling. A reset after time t invalidates the
t-to-t+1 pair. Missing neighbors at buffer edges are excluded. Targets are
stop-gradient rollout values, not recomputed neighboring minibatch rows. Model
rollout state, including latent noise, is replayed from the collected batch.
