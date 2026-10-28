# Metatron

| Experiment | Purpose |
| --- | --- |
| `g1_teacher.py` | Five-body hybrid FSQ/PPO tracker for G1. |
| `soma_teacher.py` | Full-body FSQ/PPO tracker with SOMA23 motors and PD mapping; no domain randomization. |
| `g1_student.py` | Single-teacher spherical conditional-VAE distillation for G1. |
| `soma_student.py` | Basic single-teacher Transformer conditional-VAE distillation for SOMA23. |

Neither teacher uses AMP. Supply motion data retargeted to the selected robot.
Use a new experiment name and omit `--checkpoint` to train from scratch.

## Train a teacher

```bash
python protomotions/train_agent.py --robot-name g1 --simulator isaaclab \
  --experiment-path examples/experiments/metatron/g1_teacher.py \
  --motion-file /path/to/g1_motions.pt \
  --num-envs 4096 --batch-size 4096 --experiment-name metatron_g1_teacher
```

For SOMA, use `--robot-name soma23`, `soma_teacher.py`, and SOMA motion data.
Add `--create-config-only` to inspect configs without training.

## The G1 teacher (five-body hybrid)

`g1_teacher.py` is a privileged PPO tracker with five global body targets,
whole-body joint targets, maximal and reduced current state, and a 64-scalar FSQ
bottleneck. It uses quarter-effort BeyondMimic action scaling with the robot's
configured gains, and no AMP or reward normalization. Train it fresh rather than
loading weights from the smaller `mimic/fsq.py` model.

Its low-pose limits use a stateless, causal EMA approximation of reference anchor
height: 51 samples over the preceding second, with a 0.2 s time constant. Samples
are terrain-corrected, clamped to clip boundaries, and the recurrence is seeded
with the oldest height; no future poses or episode history enter the calculation.
The standard mimic control enables this through `reference_height_smoothing_seconds`
(zero by default, meaning disabled). Below 0.5 m, hands/torso XYZ error must stay
within 0.5 m and feet Z error within 0.3 m. Upright training uses maximum body
error; evaluation uses mean body error plus separate aligned-foot and foot-height
criteria. Raw metric diagnostics do not independently gate success. Both use
global tracking, not per-step reference realignment. Dataset balancing belongs in
the supplied data; the recipe contains no dataset-specific recovery shard IDs. A
concrete motion file selects fixed loading; a `slurmrank` filename pattern enables
live cycling.

Evaluation declares diagnostic `metric_components` separately from
`failure_components`. The teacher scores success with its three low-pose
tracking/foot rules; `gt_error` remains diagnostic. Build fresh
`resolved_configs.json` for this evaluator API instead of resuming an old config
bundle. Saved action scales are authoritative too: checkpoints built with the
earlier duplicate quarter multiplier retain that smaller scale — do not treat
those policies as action-compatible with the corrected fresh-training recipe. The
teacher binds `compute_action_rate_l2` for its unchanged raw action-rate reward;
regenerate configs that reference the removed `compute_action_smoothness_squared`
helper, as no compatibility alias is kept.

## Distill a student

```bash
python protomotions/train_agent.py --robot-name g1 --simulator isaaclab \
  --experiment-path examples/experiments/metatron/g1_student.py \
  --expert-model-path results/metatron_g1_teacher/last.ckpt \
  --motion-file /path/to/g1_motions.pt \
  --num-envs 2048 --batch-size 4096 --experiment-name metatron_g1_student
```

For SOMA, use `--robot-name soma23`, `soma_student.py`, a SOMA teacher, and
matching motions. Keep the teacher's resolved configs beside its checkpoint.
Students retain the teacher's action/physics contract; inference does not need
the teacher checkpoint. These are basic single-teacher recipes, not co-distillation.

## Pretrained policy

[Metatron v1](https://huggingface.co/nvidia/Metatron-v1/blob/main/G1/MODEL_CARD.md)
is the pretrained G1 SAMP-capable policy. Its architecture is stored beside
the weights, separately from these training recipes. **No pretrained Metatron
SOMA policy is included.** Training datasets are not bundled.

Run `python scripts/download_metatron.py` from the repository root to download
the pinned native G1 package, including configs and legal files. See the
[download guide](../../../data/pretrained_models/metatron/README.md).

See the [G1 student guide](../../../docs/source/user_guide/metatron_student.rst)
for the observation and model details. Existing runs retain their own experiment
snapshot and resolved configs; these renamed entry points are for fresh configs.
