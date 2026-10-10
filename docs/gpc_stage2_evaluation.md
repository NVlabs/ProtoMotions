# GPC Stage 2 P0 evaluation

This evaluator measures the frozen Stage 1 / trainable Stage 2 combination without
changing the prior architecture, FSQ grouping, loss, optimizer, or training actor.
`GPCPriorEvaluator` runs through the existing training evaluator and Fabric logger;
`--use-wandb` enables the same scalar metrics in W&B. No extra W&B run is created
by a rank. Standalone inference saves local artifacts and prints metrics.

## Implemented scope

- `evaluate_prior_offline()` uses fixed expert states and frozen grouped-token
  labels. NLL is **unsmoothed natural-log probability per valid grouped token**.
  Top1, Top5, and per-position accuracy use the same mask and denominator.
- Free autoregressive sampling on **the same states**, with **no GT prefix**,
  produces separate histograms. Report generated/GT utilization, normalized
  empirical entropy, effective vocabulary, JS (nats, [0, ln 2]), sample counts,
  vocabulary size and FSQ group size. No dense one-hot tensors are allocated.
- All ranks SUM NLL/count statistics and group histograms **before** computing
  ratios, entropy, utilization or JS. A mean of rank-local entropies is incorrect.
- `evaluate_prior_rollout()` uses `model.generate()` -> frozen decoder -> Isaac
  Lab. It never requests GT tokens and never resets on reference `done` signals.
  Finite policy actions continue after a fall, allowing terminal recovery.
- `rollout/survival_rate` measures **terminal** upright state. Default criterion:
  robot anchor height above terrain >= 0.30 m AND local up-Z >= 0.25. This is a
  configurable P2 diagnostic, not the paper's root-height-only criterion.
  Any nonfinite action disqualifies that episode.
- `rollout/never_fell_rate` / legacy `upright_survival_horizon` track persistent
  failures (5 consecutive control steps). Initial non-upright states fail that
  diagnostic at t=0. Terminal survival and never-fell rate are distinct.

P1 remains separate: repeated-initial-state APD, normalized body/foot jerk,
mean acceleration, perturbations/recovery windows and latency. Reports store
`recovery_success_rate: null` with `recovery_status: not_implemented_p1`; this is
NA, never a fabricated zero or success rate. Existing action-delta/contact-speed
proxies are retained and must not be called physical jerk.

## Stable dashboard fields

`train/ce_loss` aliases the original supervised cross entropy (including configured
label smoothing). The training loss itself is unchanged.

With an explicit held-out validation library:

```text
eval/val_nll
eval/token_top1_acc
eval/token_top5_acc
eval/token_acc_pos_0 ...
eval/sample_entropy_norm
eval/gt_entropy_norm
eval/token_js_divergence
eval/token_utilization
eval/gt_token_utilization
eval/effective_vocab
eval/gt_effective_vocab
eval/sample_token_count
eval/gt_token_count
eval/vocab_size
eval/group_size
rollout/survival_rate
rollout/never_fell_rate
rollout/episode_count
```

All splits also have `eval_train/prior/*`, `eval_validation/prior/*` and
`eval_{raw,ema}_{train,validation}/*`. Without a validation library, train-fit
metrics are **not** labeled `eval/val_nll`. Rollout aliases use validation when
available, otherwise train; the JSON records `score_split`. Best-model selection
uses terminal survival; offline-only mode returns no selection score.
Legacy NLL/token/top5 names are kept as aliases; older teacher-forced action-MSE,
sequence-accuracy and mean-motion-perplexity diagnostics are no longer produced.

## Fixed protocol and isolation

Configuration comes from `GPCPriorEvaluatorConfig`; CLI overrides also apply to
old saved checkpoint configs:

```text
--gpc-prior-eval
--prior-eval-disable                 # disables periodic and load-triggered evaluation
--prior-eval-mode both|offline|rollout
--prior-eval-every 200               # epochs/iterations, NOT frames or optimizer updates
--prior-eval-num-envs 32             # initial motions PER RANK, capped by num-envs
--prior-eval-steps 1000              # control steps per autonomous episode
--prior-eval-teacher-steps 64        # expert collection steps
--prior-eval-seed 20260929
--prior-eval-top-p 0.9
--prior-eval-temperature 1.0
--prior-eval-min-height 0.30
--prior-eval-min-up-z 0.25
--prior-eval-weights raw|ema
--prior-eval-compare-raw-ema
--prior-validation-motion-file /absolute/heldout.pt
--prior-eval-output output/my_eval
```

Defaults sample expert labels every 4 steps (`teacher_metric_stride` in config),
one generated sequence per labeled state. Sample budgets depend on world size:
32 states/rank x 16 labeled steps x 8 ranks = 4096 states. Hold world size, budgets,
seed, simulator settings, checkpoint and data constant when comparing runs.
Each shared library must have at least one motion per rank. Validation motions
are deterministically partitioned across ranks; training shards use local IDs.

The full-motion train/validation split must be created upstream. The evaluator
rejects identical library files and exact source-clip-path overlap across ranks;
renamed copies and related variants still require a source-level split audit.
It does not silently split neighboring frames or substitute another dataset.

The first call builds a fixed expert-state/label cache, reused in subsequent
periodic evaluations and raw/EMA comparisons. The exported cache records the
actual frozen encoder SHA-256, grouping, library, motion IDs and seed. The pure
offline runner rejects incompatible encoder weights or grouping. A resumed
process deterministically rebuilds the cohort; retain the exported cache for
exact cross-checkpoint comparison without simulation.

Python/NumPy/Torch RNG, model mode, sampling settings, environment snapshot,
motion manager/library, reset noise and pushes are restored on exit. Collection,
offline sampling and rollout use separate RNG scopes, so enabling/disabling the
offline pass or reusing its cache cannot change the rollout random stream.
The repository's existing post-evaluation update-skip behavior is unchanged.
GPU physics may have nondeterministic numerical differences; do not promise
bitwise cross-device or cross-world-size rollout equality.

Raw/EMA source is reported explicitly. If an old checkpoint lacks historical
EMA, integrated evaluation warns and reports its actual loaded weights instead
of calling a newly initialized EMA history genuine EMA. Comparing both sources
requires historical EMA. The pure offline CLI rejects an unavailable requested
source; inference exports that contain only EMA cannot reconstruct raw weights.

## 1号机 smoke: offline + real physics

Validated host: `chenguanting@172.16.9.8`, repository `/data/chenguanting/ProtoMotions`.
Use an available GPU; the script's default is GPU 6.

```bash
cd /data/chenguanting/ProtoMotions
CUDA_VISIBLE_DEVICES=6 bash scripts/evaluation/gpc_stage2_smoke.sh
```

This evaluates the existing `gpc_prior_amass_8gpu/inference_last.ckpt` using raw
weights, the matching AMASS train shard and independent validation MotionLib.
Two environments per split, eight expert collection steps, 100 control steps
(2 seconds at dt=0.02): an integration smoke, **not a quality benchmark**.
Override `CHECKPOINT` / `OUTPUT` via environment variables or append CLI flags.
No long-running training is started.

Outputs: `epoch_*_rank_*.json` (globally aggregated scalars, rank-local cohorts,
protocol and weight source), rank-zero scalar CSV, rank-zero per-group top-k
CSV/PNG, and `offline_{split}_rank_*.pt` tensor caches. Histograms compare the same
IDs selected by combined GT+sample frequency, showing their sample counts.
Use a separate output directory for each protocol/run.

## Pure offline evaluation: no simulator

After exporting a cohort once:

```bash
cd /data/chenguanting/ProtoMotions
CUDA_VISIBLE_DEVICES=7 /data/chenguanting/.venvs/protomotions-isaaclab/bin/python \
  -m protomotions.agents.evaluators.gpc_prior_offline \
  --checkpoint results/gpc_prior_amass_8gpu/inference_last.ckpt \
  --cache output/gpc_p0_verified_20261010/offline_validation_rank_0.pt \
  --device cuda:0 --weights raw --seed 20260929 --top-p 0.9 --temperature 1.0 \
  --output output/gpc_p0_verified_20261010/offline_replay
```

Use `--device cpu` without a GPU. This loads the checkpoint's resolved model
config and frozen Stage 1 path, not an environment. It writes JSON/CSV and full
count tensors, including checkpoint/cache SHA-256. It evaluates one cache per
invocation; the integrated training evaluator supports distributed aggregation.

## Training-time online evaluation

Append to an existing **compatible** Stage 2 training/resume command:

```bash
--gpc-prior-eval --prior-eval-mode both --prior-eval-every 200 \
--prior-validation-motion-file /data/chenguanting/datasets/AMASS/p2_motionlib/validation/proto-astro_p2.pt \
--prior-eval-num-envs 8 --prior-eval-teacher-steps 64 --prior-eval-steps 500 \
--prior-eval-seed 20260929 --prior-eval-top-p 0.9 --prior-eval-temperature 1.0 \
--prior-eval-weights ema --use-wandb --wandb-project p2-gpc
```

The AMASS validation example is for AMASS training; for BONES use its own held-out
MotionLib. Same-name resume reloads saved general CLI flags (including W&B);
explicit `--prior-*` overrides are reapplied. Do not assume `--use-wandb` overrides
an old saved `use_wandb: false`. Use an already W&B-enabled run or configure a
separate named warm start deliberately. Keep checkpoint/data/Stage 1 paths and
training budgets as intended; this change does not restart existing processes.

At 8 ranks, `num-envs=8` produces 64 episodes per split. This P0 cohort uses
independent initial motions, not the repeated-pose APD protocol. Frequency is
in epochs; choose it deliberately from actual training cost rather than treating
200 epochs as 200 environment frames. No automatic hyperparameter adjustment.

## Tests

```bash
PYTHONPATH=/tmp/gpc-embedding-pytest:$PWD /data/chenguanting/.venvs/protomotions-isaaclab/bin/python \
  -m pytest -q protomotions/tests/test_gpc_prior_p0.py \
  protomotions/tests/test_gpc_prior_evaluator.py protomotions/tests/test_gpc_prior_config.py
OMP_NUM_THREADS=1 PYTHONPATH=/tmp/gpc-embedding-pytest:$PWD \
  /data/chenguanting/.venvs/protomotions-isaaclab/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 scripts/evaluation/gpc_stage2_ddp_smoke.py
```

The 8-process test uses CPU/Gloo and unequal sample counts. It verifies the real
collective path, not 8-GPU Isaac Lab execution. Full 8-GPU simulator/training
validation and online W&B upload are separate from these checks.
