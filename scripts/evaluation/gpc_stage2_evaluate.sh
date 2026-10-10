#!/usr/bin/env bash
set -euo pipefail
cd /data/chenguanting/ProtoMotions
EVAL_PYTHON=${EVAL_PYTHON:-/data/chenguanting/.venvs/protomotions-isaaclab/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export MUJOCO_GL=egl ACCEPT_EULA=Y NCCL_CUMEM_HOST_ENABLE=0
unset DISPLAY
"$EVAL_PYTHON" protomotions/inference_agent.py \
  --checkpoint "${CHECKPOINT:-results/gpc_prior_amass_8gpu/inference_last.ckpt}" \
  --simulator isaaclab --headless --full-eval \
  --ngpu "${NGPU:-1}" --num-envs "${NUM_ENVS:-32}" \
  --motion-file /data/chenguanting/datasets/AMASS/p2_motionlib/shards_8/amass_smpl_train_astro_p2_0.pt \
  --gpc-prior-eval --prior-eval-profile "${PROFILE:-quick}" \
  --prior-validation-motion-file /data/chenguanting/datasets/AMASS/p2_motionlib/validation/proto-astro_p2.pt \
  --prior-eval-weights raw --prior-eval-seed 20260929 \
  --prior-eval-output "${OUTPUT:-output/gpc_complete_eval}" "$@"
