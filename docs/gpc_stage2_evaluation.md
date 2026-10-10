# GPC Stage 2 evaluation: P0 / P1 / P2

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
- `rollout/never_fell_rate` tracks persistent
  failures (5 consecutive control steps). Initial non-upright states fail that
  diagnostic at t=0. Terminal survival and never-fell rate are distinct.

P1 adds repeated-initial-pose APD, physical acceleration/jerk, normalized
body/foot jerk, measured velocity perturbations, recovery events and clustered
confidence intervals. P2 adds synchronized complete-prior/decoder latency and
replay videos. See the precise conventions below; these are custom Astro P2
metrics, not claims of reproducing the paper's numerical tables.

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
--prior-eval-num-envs 32             # offline expert cohort PER RANK, capped by num-envs
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

## Host and outputs

Validated host: `chenguanting@172.16.9.8`, repository
`/data/chenguanting/ProtoMotions`. The complete-evaluation wrapper below uses
`gpc_prior_amass_8gpu/inference_last.ckpt`, its AMASS train shard and independent
validation MotionLib. Override `CHECKPOINT` / `OUTPUT` via environment variables
or append CLI flags. No long-running training is started by evaluation.

Outputs include `epoch_*_rank_*.json` (global scalars and rank-local provenance),
rank-zero scalar CSV, per-group top-k CSV/PNG, `offline_{split}_rank_*.pt` tensor
caches and physical trajectory files. Histogram panels show the same IDs chosen
by combined GT/sample frequency. Use a separate output directory per run.

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
count tensors, including checkpoint/cache SHA-256. It accepts multiple cache files and supports torchrun SUM reductions. See the
distributed example below; cache argument order determines the per-rank assignment.

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

Online rollout now uses a GLOBAL pose budget, independently of the offline
expert-state budget. All repeats for one pose stay on one rank, including when
there are fewer poses than ranks. The number of simulator environments controls
batching, not the total rollout episode count. Cadence is in epochs. Existing
processes must be restarted/resumed deliberately to load changed code.

## P1/P2 protocol and profiles

| Profile | Global poses | Repeats per pose | Steps | Conditions (root delta-v) |
| --- | ---: | ---: | ---: | --- |
| `quick` | 8 | 8 | 500 | 0 m/s |
| `full` | 15 | 32 | 1000 | 0, 0.5, 1.5 m/s at step 200 |
| `smoke` | 2 | 2 | 200 | 0, 0.5, 1.5 m/s at step 20 |

The smoke profile also sets 8 expert collection steps, 2 expert environments,
and a 2-second recovery deadline. Explicit CLI flags override profile values.
At control dt=0.02, full runs last 20 seconds per episode, with 480 episodes per
split and condition. Train plus validation, three conditions = 2880 episodes;
raw/EMA comparison doubles this. The full profile has not been run as part of
the integration smoke. Do not infer benchmark quality from a 4-second smoke.

- **Initial state / APD:** all repeats reset to the same motion's initial state,
  with reset noise disabled and body/joint/orientation agreement checked. Root
  APD is mean pairwise XY L2 averaged over control samples. Pose APD is mean
  pairwise L2 of the flattened selected-body XYZ vector, averaged over samples.
  Both use metres. Each episode's initial root XY and terrain height are removed
  to eliminate spawn offsets; orientation and subsequent global displacement are
  retained. A separate mean-body-distance diagnostic is also exported. Default
  body set is every robot rigid body; the actual body/foot lists are recorded.
- **Smoothness:** positions come from simulated rigid bodies, sampled every
  control step. Acceleration uses second forward differences (m/s²), physical
  jerk uses third forward differences (m/s³). Normalized jerk is dimensionless:
  `D^5 * sum(||third_difference / dt^3||^2 * dt) / path_length^2` in overlapping
  0.4-second windows (21 position samples at dt=0.02), then mean over windows and
  bodies. This is an explicitly specified discrete, path-length convention.
  Windows with path length <=1e-6 m are excluded instead of adding a large
  epsilon. Stationary trajectories have zero raw jerk but NA normalized jerk.
  Valid-window fraction is exported, and absent feet give NA foot metrics.
  Nonfinite episodes are excluded from APD/kinematic means and counted explicitly.
- **Recovery:** a fall requires the configured number of consecutive bad samples
  (default 5); an initially bad state counts at t=0. A stable recovery needs
  anchor height >=0.45 m, up-Z >=0.8, root linear speed <=0.5 m/s and angular
  speed <=1 rad/s, continuously for 0.5 seconds. The default deadline is 3 s.
  `fall_or_push` uses the first fall or verified push as the episode trigger;
  `fall` restricts it to falls. No trigger means NA, not success. A trigger too
  late to observe the entire deadline is censored and excluded from the recovery
  denominator. A persistent fall is one event until stable rearming. Terminal
  survival, never-fell rate and recovery success remain separate metrics.
- **Pushes:** an additive horizontal root velocity increment is read back from
  the simulator and checked before physics. It is applied after the current
  action computation, so the policy observes it on its next decision. Directions
  are seeded independently and reused across conditions and weight sources.
  0.5/1.5 m/s are P2-specific test settings, not the paper's 2.4/9.8 m/s.
- **Statistics:** ordinary metrics first average within each initial pose;
  overall mean/sample SD/95% percentile bootstrap CI use pose clusters. Recovery
  aggregates eligible-success counts, with the same cluster bootstrap. One
  valid pose gives no CI; JSON uses `null` for undefined values. Invalid episodes,
  trigger/eligible/censored counts and motion IDs remain auditable.
- **Latency:** synchronized wall time covers state TensorDict preparation,
  every autoregressive group and the frozen decoder. Report mean/P95/std and
  measured batch-size range, excluding simulation, file export and videos.
  Default warmup is 10 control steps, then one sample every 10 steps. Values are
  per batch and are NOT divided by the number of robots.
- **Videos:** optional MP4 skeleton replays use actual physical trajectories.
  Success/failure/high empirical token-rarity cases are selected when present;
  rarity does not claim a semantically rare skill. These are not RGB camera
  recordings. Rank zero exports its local examples, not an exhaustive global
  failure search.

Sources: [GPC Appendix C.3](https://arxiv.org/html/2606.29148v1) supplies the
repeated-pose evaluation motivation;
[Hogan and Sternad](https://pmc.ncbi.nlm.nih.gov/articles/PMC3470860/) supplies
the dimensional normalization. Coordinate, body, finite-difference, window and
recovery conventions above must be retained when comparing checkpoints.

Additional CLI controls (see `gpc_prior_cli.py` for the complete parser):

```text
--prior-eval-profile quick|full|smoke
--prior-eval-poses 15 --prior-eval-repeats 32
--prior-eval-push-speeds 0 0.5 1.5 --prior-eval-push-step 200
--prior-eval-jerk-window 0.4
--prior-eval-recovery-trigger fall_or_push
--prior-eval-recovery-hold 0.5 --prior-eval-recovery-timeout 3
--prior-eval-latency-warmup 10 --prior-eval-latency-stride 10
--prior-eval-bootstrap-samples 500 --prior-eval-video-cases 3
--prior-eval-no-trajectories
```

The scalar namespace includes `rollout/apd_root`, `rollout/apd_pose`,
`rollout/normalized_jerk`, `rollout/normalized_foot_jerk`,
`rollout/mean_acceleration`, `rollout/foot_jerk`,
`rollout/recovery_success_rate`, and `rollout/latency_ms` / `latency_p95_ms`.
Unqualified aliases are no-push results. Pushes have separate namespaces such
as `rollout/push_0p5mps/...`; every split and weight source is also retained.

## Complete evaluation on host 1

```bash
cd /data/chenguanting/ProtoMotions
# Real closed-loop smoke with two GPUs; choose GPUs free on the host.
CUDA_VISIBLE_DEVICES=6,7 NGPU=2 NUM_ENVS=2 PROFILE=smoke \
  OUTPUT=output/my_stage2_smoke bash scripts/evaluation/gpc_stage2_evaluate.sh
# Full protocol, same checkpoint/data defaults; expensive, launch deliberately.
CUDA_VISIBLE_DEVICES=6 NGPU=1 NUM_ENVS=32 PROFILE=full \
  OUTPUT=output/my_stage2_full bash scripts/evaluation/gpc_stage2_evaluate.sh \
  --prior-eval-video-cases 3
```

`--ngpu > 1` is restricted to headless full evaluation. Training continues to
use its existing Fabric/DDP launch. Reports contain global metrics on each rank,
rank-local cohorts, per-episode CSVs, histogram CSV/PNGs, tensor caches and,
unless disabled, physical trajectory `.pt` files. Exported trajectories can be
large for full profiles. Rank-zero report/media logging uses the existing W&B
training run when one is configured; ranks do not create extra runs.

Distributed pure-offline token evaluation (no Isaac Lab initialization):

```bash
CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=4 \
  /data/chenguanting/.venvs/protomotions-isaaclab/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m protomotions.agents.evaluators.gpc_prior_offline \
  --checkpoint results/gpc_prior_amass_8gpu/inference_last.ckpt \
  --cache output/my_stage2_smoke/offline_validation_rank_0.pt \
          output/my_stage2_smoke/offline_validation_rank_1.pt \
  --device cuda --weights raw --seed 20260929 \
  --output output/my_stage2_smoke/offline_distributed
```

Use the same ordered cache files, world size, seed and sampling settings to
reproduce generated-token histograms. NLL/accuracy pool sufficient statistics;
changing world size changes the sampling RNG assignment. Empty-work ranks still
participate in the reductions. Checkpoint/cache digests are exported.

Pure offline P1 physical replay (CPU; no checkpoint or simulator required):

```bash
/data/chenguanting/.venvs/protomotions-isaaclab/bin/python \
  -m protomotions.agents.evaluators.gpc_prior_replay \
  --trajectories 'output/my_stage2_smoke/epoch_0_eval_1_raw_validation_no_push_rank_*.trajectories.pt' \
  --output output/my_stage2_smoke/trajectory_replay
```

Combine all ranks for one split/weight/condition. The replay recomputes APD,
acceleration, jerk, survival and recovery using stored state flags and supports
`--jerk-window`. It cannot change standing thresholds retrospectively and does
not recompute latency. Preserve the original report alongside derived reports.

For standalone W&B recording, after a report exists:

```bash
/data/chenguanting/.venvs/protomotions-isaaclab/bin/python \
  -m protomotions.agents.evaluators.gpc_prior_reporting \
  --report output/my_stage2_smoke/epoch_0_eval_1_rank_0.json \
  --project p2-gpc --name stage2-evaluation --mode offline
```

`--mode online` uploads to the selected account/project when authorized. It
includes report/config paths, checkpoint/data paths, metrics, histograms and
replay videos. Offline mode was validated locally; cloud publication is a
separate action. W&B availability is unrelated to closed-loop online evaluation.

## Tests

```bash
PYTHONPATH=/tmp/gpc-embedding-pytest:$PWD /data/chenguanting/.venvs/protomotions-isaaclab/bin/python \
  -m pytest -q protomotions/tests/test_gpc_prior_p0.py \
  protomotions/tests/test_gpc_prior_trajectory.py \
  protomotions/tests/test_gpc_prior_evaluator.py protomotions/tests/test_gpc_prior_config.py
OMP_NUM_THREADS=1 PYTHONPATH=/tmp/gpc-embedding-pytest:$PWD \
  /data/chenguanting/.venvs/protomotions-isaaclab/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 scripts/evaluation/gpc_full_ddp_smoke.py
```

The 8-process test uses CPU/Gloo with unequal sample counts and empty-work
ranks. It verifies real collectives and clustered aggregation, not 8-GPU Isaac
Lab execution. Single-GPU physical smoke and two-GPU physical smoke are separate
checks; neither establishes full-benchmark policy quality.
