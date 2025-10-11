#!/usr/bin/env bash

# Helper to refresh the Hugging Face dataset clone.
# One-time setup:
#   git clone https://huggingface.co/datasets/ludekcizinsky/zurihack <your_path>
# Usage after cloning:
#   From the ProtoMotions repo root run: bash pull_dataset_repo.sh
#   (Edit DATASET_REPO_ROOT below if you cloned to a different path.)

set -euo pipefail

# Path to the dataset git repo clone. Update if you keep it elsewhere.
DATASET_REPO_ROOT="<your_path>"

if [[ ! -d "${DATASET_REPO_ROOT}/.git" ]]; then
  echo "Dataset repository not found at ${DATASET_REPO_ROOT}."
  echo "Update DATASET_REPO_ROOT in pull_dataset_repo.sh to your clone location."
  exit 1
fi

echo "Pulling latest changes in ${DATASET_REPO_ROOT}..."
git -C "${DATASET_REPO_ROOT}" pull --ff-only
echo "Done."
