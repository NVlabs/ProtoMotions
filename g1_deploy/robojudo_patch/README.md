# RoboJudo Patch for ProtoMotions BeyondMimic Deployment

This patch adds support for deploying ProtoMotions BeyondMimic (BM) tracker
policies on the Unitree G1 via the [RoboJudo](https://github.com/hansz8/robojudo/)
framework.

> **Note:** This is a temporary solution. We are working on properly integrating
> these changes as a PR into the upstream RoboJudo repository. This patch will
> be removed once that PR is merged.

## Prerequisites

- ProtoMotions with a trained G1 BM tracker policy exported to ONNX
- Python 3.10+ with MuJoCo and ONNX Runtime installed

## Setup

### 1. Clone RoboJudo at the compatible commit

The patch is built against RoboJudo commit
[`806850e`](https://github.com/hansz8/robojudo/tree/806850e46f2fef7f15e48ff8da867d8cb6f7db71).
Clone and check out this specific commit:

```bash
git clone https://github.com/hansz8/robojudo.git
cd robojudo
git checkout 806850e46f2fef7f15e48ff8da867d8cb6f7db71
```

### 2. Apply the patch

From the RoboJudo repo root:

```bash
git am /path/to/protomotions/g1_deploy/robojudo_patch/protomotions-bm-tracker.patch
```

If you prefer not to create git commits, use `git apply` instead:

```bash
git apply /path/to/protomotions/g1_deploy/robojudo_patch/protomotions-bm-tracker.patch
```

### 3. Install RoboJudo

Follow the RoboJudo installation instructions in their README.

## Usage

### Simulation (MuJoCo)

```bash
python scripts/run_pipeline.py -c g1_protomotions_bm_tracker \
    --onnx-path /path/to/unified_pipeline.onnx \
    --motion-path /path/to/motion.motion
```

### Real Robot

```bash
python scripts/run_pipeline.py -c g1_protomotions_bm_tracker_real \
    --onnx-path /path/to/unified_pipeline.onnx \
    --motion-path /path/to/motion.motion
```

## What the patch adds

- **BeyondMimic tracker policy** (`protomotions_bm_tracker_policy.py`): ONNX
  inference with action history feedback, anchor-body rotation references, and
  motion fade-in/fade-out support.
- **Virtual gantry**: spring-damper safety harness for real-robot deployment
  transitions.
- **Blend-in/blend-out**: smooth transitions between idle pose and policy
  control.
- **CLI enhancements**: `--onnx-path`, `--motion-path`, `--motion-index`,
  `--simulate-deploy`, `--prepare-seconds`, `--hold-seconds`.

See `docs/bm-tracker-changes.md` (in the patched RoboJudo repo) for a detailed
description of all changes.
