#!/usr/bin/env bash
set -euo pipefail
cd /data/chenguanting/ProtoMotions
PYTHON=${PYTHON:-/data/chenguanting/.venvs/protomotions-isaaclab/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export MUJOCO_GL=egl
export ACCEPT_EULA=Y
unset DISPLAY
"$PYTHON" protomotions/inference_agent.py \
  --checkpoint "${CHECKPOINT:-results/gpc_prior_amass_8gpu/inference_last.ckpt}" \
  --simulator isaaclab --headless --full-eval --num-envs 2 \
  --motion-file /data/chenguanting/datasets/AMASS/p2_motionlib/shards_8/amass_smpl_train_astro_p2_0.pt \
  --gpc-prior-eval \
  --prior-validation-motion-file /data/chenguanting/datasets/AMASS/p2_motionlib/validation/proto-astro_p2.pt \
  --prior-eval-num-envs 2 --prior-eval-steps 100 --prior-eval-teacher-steps 8 \
  --prior-eval-weights raw --prior-eval-seed 20260929 \
  --prior-eval-top-p 0.9 --prior-eval-temperature 1.0 \
  --prior-eval-output "${OUTPUT:-output/gpc_p0_verified_20261010}" "$@"
