# Building the `protomotions:isaaclab` Docker Image

This guide covers how to build the ProtoMotions IsaacLab Docker image on both a **local workstation** (with Docker Hub access) and a **remote HPC cluster** (without Docker Hub access, using images already available on the cluster).

---

## Overview

The image is defined in [`Dockerfile.isaaclab`](../../Dockerfile.isaaclab) and installs:

| Layer | Package / Version |
|---|---|
| Base image | `nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04` |
| Python | 3.11 (system) |
| Package manager | `uv` (fast pip-compatible installer) |
| PyTorch | `torch==2.5.1+cu124`, `torchvision==0.20.1+cu124` |
| IsaacLab + IsaacSim | `isaaclab[isaacsim,all]==2.3.0` (from `pypi.nvidia.com`) |
| ProtoMotions deps | `requirements_isaaclab.txt` (pinned versions) |

ProtoMotions itself is **not** copied into the image — it is synced and installed at runtime by `train_slurm.py` to ensure fresh code on every job.

---

## Prerequisites

### Both Environments

- Docker ≥ 20.10 with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- GPU driver ≥ 550 (required for CUDA 12.4 containers)
- Git LFS installed (`git lfs install`)
- Network access to `pypi.nvidia.com` and `download.pytorch.org` during the build

Verify the NVIDIA runtime is available:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04 nvidia-smi
```

### Local Workstation (Additional)

- Docker Hub access is required only if pulling a different base image. The Dockerfile uses `nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04` which is hosted on NVIDIA's registry — **Docker Hub access is not required**.
- NVIDIA NGC account for pulling from `nvcr.io` (free): https://ngc.nvidia.com

Log in to the NVIDIA Container Registry:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```

### Remote HPC (Additional)

The base image `nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04` (tagged as `899566afc09e`) is already present on the HPC, so no pull is required. Confirm with:

```bash
docker images | grep "12.4.0-devel"
# nvcr.io/nvidia/cuda   12.4.0-devel-ubuntu22.04   899566afc09e   24 months ago   7.22GB
```

During the build, Docker still needs outbound network access to:
- `astral.sh` — download `uv` installer
- `download.pytorch.org` — torch wheels
- `pypi.nvidia.com` — IsaacLab / IsaacSim wheels
- `pypi.org` — ProtoMotions dependencies

If the HPC has a network proxy, set it before building (see [Proxy Configuration](#proxy-configuration) below).

---

## Step-by-Step Build Instructions

### Step 1 — Clone the Repository

```bash
git clone https://github.com/notAnyrobot/ProtoMotions.git
cd ProtoMotions

# Pull large binary files via Git LFS
git lfs fetch --all
git lfs checkout
```

### Step 2 — Verify the Dockerfile and Requirements

Review the two files that define the image:

- [`Dockerfile.isaaclab`](../../Dockerfile.isaaclab)
- [`requirements_isaaclab.txt`](../../requirements_isaaclab.txt)

Key pinned packages in `requirements_isaaclab.txt` and the reason they are pinned:

| Package | Pinned Version | Reason |
|---|---|---|
| `tensordict` | `0.9.0` | IsaacLab installs 0.11.0; ProtoMotions is validated against 0.9.0 |
| `wandb` | `0.15.12` | Stable logging version used in all ProtoMotions experiments |
| `sentry-sdk` | `1.38.0` | Matches wandb 0.15.12 dependency |
| `setuptools` | `69.5.1` | Avoids setuptools 70+ breaking changes for some deps |
| `rtree` | `1.2.0` | Required by terrain generation components |
| `hydra-core` | `1.3.2` | Config system version used by train_agent.py |

> **Note:** If you upgrade `tensordict` to `>=0.11.0` and experience runtime errors specific to ProtoMotions (not IsaacLab), that is the first version to investigate.

### Step 3 — Build the Image

Run from the root of the repository (where `Dockerfile.isaaclab` lives):

```bash
docker build \
    -f Dockerfile.isaaclab \
    -t protomotions:isaaclab \
    .
```

The build has four stages and estimated durations on a fast network:

| Stage | What happens | Estimated time |
|---|---|---|
| System deps | apt-get: cmake, git, python3.11, etc. | ~2 min |
| torch install | Downloads ~2 GB of cu124 wheels | ~5–10 min |
| isaaclab install | Downloads ~8 GB of IsaacSim wheels | ~15–30 min |
| ProtoMotions deps | ~30 packages from requirements_isaaclab.txt | ~2–5 min |

Total: **~25–50 minutes** depending on network speed and cache hits.

#### Build with a custom tag

```bash
docker build \
    -f Dockerfile.isaaclab \
    -t protomotions:isaaclab-v1.0 \
    .
```

#### Build without using cached layers (clean rebuild)

```bash
docker build \
    --no-cache \
    -f Dockerfile.isaaclab \
    -t protomotions:isaaclab \
    .
```

### Step 4 — Verify the Image

Run a quick smoke test inside the container:

```bash
docker run --rm --gpus all protomotions:isaaclab python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
import tensordict
print('TensorDict:', tensordict.__version__)
import lightning
print('Lightning:', lightning.__version__)
"
```

Expected output (device name will vary):

```
PyTorch: 2.5.1+cu124
CUDA available: True
CUDA device: NVIDIA GeForce RTX 4090
TensorDict: 0.9.0
Lightning: 2.x.x
```

Also verify IsaacLab is importable (headless mode):

```bash
docker run --rm --gpus all \
    -e OMNI_KIT_ACCEPT_EULA=YES \
    -e ACCEPT_EULA=Y \
    protomotions:isaaclab \
    python3 -c "import isaaclab; print('IsaacLab OK')"
```

---

## Remote HPC Specific Instructions

### Transferring the Image

If you build on your local workstation and want to transfer to HPC rather than rebuilding there:

**Option A — Save/load as tar archive (no registry needed):**

```bash
# On local workstation: save
docker save protomotions:isaaclab | gzip > protomotions_isaaclab.tar.gz

# Transfer to HPC (adjust hostname/path)
rsync -avz --progress protomotions_isaaclab.tar.gz user@hpc-login:/scratch/user/

# On HPC: load
docker load < /scratch/user/protomotions_isaaclab.tar.gz
docker images | grep protomotions
```

**Option B — Use a private registry (if your HPC has one):**

```bash
# On local workstation
docker tag protomotions:isaaclab registry.hpc.example.com/yourproject/protomotions:isaaclab
docker push registry.hpc.example.com/yourproject/protomotions:isaaclab

# On HPC
docker pull registry.hpc.example.com/yourproject/protomotions:isaaclab
docker tag registry.hpc.example.com/yourproject/protomotions:isaaclab protomotions:isaaclab
```

### Proxy Configuration

If the HPC build nodes require an HTTP proxy for outbound connections:

```bash
docker build \
    -f Dockerfile.isaaclab \
    -t protomotions:isaaclab \
    --build-arg HTTP_PROXY=http://proxy.hpc.example.com:3128 \
    --build-arg HTTPS_PROXY=http://proxy.hpc.example.com:3128 \
    --build-arg NO_PROXY=localhost,127.0.0.1 \
    .
```

Or set them in your shell before building:

```bash
export HTTP_PROXY=http://proxy.hpc.example.com:3128
export HTTPS_PROXY=http://proxy.hpc.example.com:3128
docker build -f Dockerfile.isaaclab -t protomotions:isaaclab .
```

### Running with SLURM

ProtoMotions uses `train_slurm.py` which handles syncing code into the container and launching training jobs. A typical SLURM submission using this image:

```bash
python protomotions/train_slurm.py \
    --docker-image protomotions:isaaclab \
    --robot-name g1 \
    --simulator isaaclab \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name my_experiment \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --num-envs 4096 \
    --batch-size 16384
```

The `train_slurm.py` script syncs the local ProtoMotions repo into the container at job start, so you do **not** need to rebuild the image when changing Python code — only rebuild when `Dockerfile.isaaclab` or `requirements_isaaclab.txt` change.

---

## Troubleshooting

### `uv: command not found` inside container

The `uv` installer places the binary in `/root/.local/bin` and the Dockerfile adds it to `PATH`. This only applies during the build — inside the running container, `uv` is not needed. Use `pip` or `python3 -m pip` for any runtime installs inside the container.

### `tensordict` version conflict

If you see errors like `AttributeError: module 'tensordict' has no attribute '...'`, the isaaclab installation may have upgraded tensordict past 0.9.0. Force the pin:

```bash
docker run --rm protomotions:isaaclab pip show tensordict
# If not 0.9.0, rebuild with --no-cache after verifying requirements_isaaclab.txt
```

### IsaacSim fails to initialize (`omni.kit` errors)

IsaacSim requires headless mode on servers without displays. Ensure these environment variables are set when running:

```bash
docker run --rm --gpus all \
    -e OMNI_KIT_ACCEPT_EULA=YES \
    -e ACCEPT_EULA=Y \
    -e DISPLAY="" \
    protomotions:isaaclab \
    python3 your_script.py
```

### CUDA version mismatch

The image uses CUDA 12.4. Your host GPU driver must support CUDA 12.4 or newer (driver ≥ 550.54). Check:

```bash
nvidia-smi | grep "CUDA Version"
# Should show: CUDA Version: 12.4 or higher
```

### `pypi.nvidia.com` not reachable during build

If your network blocks `pypi.nvidia.com`, you will need to either:
1. Build on a machine with access and transfer the image (see [Transferring the Image](#transferring-the-image))
2. Set up a local PyPI mirror/proxy that caches NVIDIA packages
3. Pre-download the wheels and use `--find-links` with a local directory

---

## Image Size Reference

| Stage | Approximate cumulative size |
|---|---|
| Base CUDA image | ~7.2 GB |
| + system packages + Python 3.11 | ~7.5 GB |
| + torch 2.5.1+cu124 | ~10 GB |
| + isaaclab[isaacsim,all]==2.3.0 | ~18–22 GB |
| + requirements_isaaclab.txt | ~18–22 GB (minimal delta) |

The final image will be approximately **18–22 GB**.
