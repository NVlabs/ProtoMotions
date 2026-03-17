# USD Convert

Scripts for converting robot MJCF assets to USD format for the IsaacLab simulator backend.

## Prerequisites

- Isaac Lab environment active (`source env_isaaclab/bin/activate`)
- Run all commands from the repository root

## Typical Workflow

### 1. Flatten the MJCF

Resolve `<default>` class inheritance, convert `<freejoint>` to `<joint type="free">`, and auto-name unnamed mesh geoms:

```bash
python usd_convert/flatten_mjcf.py \
    protomotions/data/assets/mjcf/g1_holo_compat.xml
```

Output: `g1_holo_compat_flat.xml` in the same directory.

### 2. Convert to USDA

Run the end-to-end pipeline (strips incompatible elements, invokes Isaac Lab converter, patches the output):

```bash
python usd_convert/convert_robot_mjcf_to_usda.py \
    protomotions/data/assets/mjcf/g1_holo_compat_flat.xml
```

Output lands in `protomotions/data/assets/usd/g1_holo_compat_flat/` by default, or use `--output-dir` to override.

### 3. Convert scene objects (optional)

Convert `.obj`/`.stl`/`.ply` mesh files to USD:

```bash
python usd_convert/convert_objects_to_usd.py path/to/meshes/
```

## Script Reference

| Script | Purpose |
|--------|---------|
| `flatten_mjcf.py` | Flatten MJCF defaults, normalize structure, verify with MuJoCo |
| `convert_robot_mjcf_to_usda.py` | End-to-end robot MJCF → USDA pipeline (calls the other scripts) |
| `convert_mjcf_to_usd.py` | Low-level Isaac Lab `MjcfConverter` wrapper (called by the pipeline) |
| `patch_usd_visual_meshes.py` | Add visual meshes the Isaac Sim converter dropped (called by the pipeline) |
| `convert_objects_to_usd.py` | Batch-convert scene object meshes (`.obj`/`.stl`/`.ply`) to USD |
