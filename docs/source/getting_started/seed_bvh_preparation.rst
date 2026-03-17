SEED BVH Data Preparation (SOMA Skeleton)
==========================================

This guide covers converting BONES-SEED BVH motion data (SOMA format, 77 joints)
into ProtoMotions format for the soma23 humanoid.

.. note::

   The BONES-SEED dataset contains **~142K motions** — significantly larger than AMASS
   (~15K motions). At this scale, both conversion and training require special handling:

   * **Conversion** should be parallelized across multiple CPUs (see
     :ref:`seed-bvh-scaling-up`).
   * **Training** should use sharded MotionLib chunks so each GPU only loads a subset
     of motions, avoiding GPU memory exhaustion (see :ref:`seed-bvh-training`).

Overview
--------

The `BONES-SEED <https://huggingface.co/datasets/bones-studio/seed>`_ dataset contains motion capture data
in BVH format using the 77-joint SOMA skeleton.
BONES-SEED provides two BVH variants: **SOMA Uniform** (standardized skeleton shared
across all motions) and **SOMA Proportional** (per-actor body proportions). Use
**SOMA Uniform** — it matches the single ``soma23_humanoid.xml`` MJCF model that
ProtoMotions uses.

The conversion pipeline:

1. Parses BVH files into local rotation matrices and root translations
2. Converts from the BVH bone-axis-aligned zero-pose to the standard T-pose using
   precomputed global rotation offsets
3. Subselects from 77 joints to the 23 actuated joints in the ``soma23_humanoid.xml`` MJCF
4. Applies Y-up to Z-up coordinate transforms and computes full rigid body states

.. code-block:: text

   BONES-SEED .bvh files (77 joints, 120 fps, Y-up)
        │
        ▼ (convert_soma23_bvh_to_proto.py)
   ProtoMotions .motion files (23 bodies, 30 fps, Z-up)
        │
        ▼ (motion_lib.py --motion-path)
   Packaged .pt MotionLib (one per chunk, or single file for small sets)

Prerequisites
-------------

1. **Download BONES-SEED**: Download the dataset from
   `Hugging Face <https://huggingface.co/datasets/bones-studio/seed>`_.
   After downloading, extract the SOMA Uniform tar archive:

   .. code-block:: bash

      cd bones-seed
      tar -xf soma_uniform.tar

   The expected directory structure is:

   .. code-block:: text

      bones-seed/
        └── soma_uniform/
            └── bvh/
                ├── <date_1>/
                │   ├── motion_001.bvh
                │   └── motion_002.bvh
                └── <date_2>/
                    └── ...

2. **T-pose offsets**: The file ``data/soma/standard_t_pose_global_offsets_rots.p``
   is included in the repository. It contains precomputed per-body global rotation
   offsets that convert the BVH's bone-axis-aligned zero-pose to the standard T-pose.

Quick Start: Small Subset
-------------------------

For a small number of BVH files (e.g., testing with a few motions), convert and
package into a single MotionLib file:

.. code-block:: bash

   # Convert
   python data/scripts/convert_soma23_bvh_to_proto.py \
       --input-dir /path/to/bones-seed/soma_uniform/bvh \
       --output-dir /path/to/output/motions \
       --input-fps 120 \
       --output-fps 30

   # Package into a single .pt
   python protomotions/components/motion_lib.py \
       --motion-path /path/to/output/motions/ \
       --output-file /path/to/seed_bvh_motions.pt

   # Verify visually
   python examples/motion_libs_visualizer.py \
       --motion_files /path/to/seed_bvh_motions.pt \
       --robot soma23 \
       --simulator isaacgym

**Key arguments:**

* ``--input-dir``: Root directory to search recursively for ``.bvh`` files
* ``--output-dir``: Directory to save ``.motion`` files (preserves subdirectory structure)
* ``--input-fps``: Source BVH frame rate (default: 120)
* ``--output-fps``: Target output frame rate (default: 30)
* ``--force-remake``: Overwrite existing ``.motion`` files
* ``--ignore-motion-filter``: Skip quality filtering (useful for debugging)

What the Converter Does
-----------------------

1. **Parse BVH**: Reads the BVH hierarchy (excluding the dummy ``Root`` joint) to get
   local rotation matrices ``(T, 77, 3, 3)`` and root translations ``(T, 3)`` in
   centimeters. Root translations are converted to meters.

2. **T-pose conversion**: The BVH zero-pose (all rotations = identity) places bones
   along their primary axis, which is NOT a natural T-pose. The ``change_tpose()``
   function re-expresses the local rotations in the standard T-pose convention using
   precomputed global rotation offsets.

3. **Body subselection**: The 77 SOMASkeleton joints are subselected to the 23 bodies
   in ``soma23_humanoid.xml``. The 54 dropped joints are leaf end-effectors (finger
   details, face joints, toe ends, etc.) without actuators.

4. **Coordinate transform + FK**: The Y-up local rotations are transformed to Z-up
   and run through the MJCF forward kinematics to produce world positions, rotations,
   velocities, DOF positions/velocities, and ground contact labels.

5. **Contact label estimation**: Heuristic ground contact labels are computed for
   all bodies based on height and velocity thresholds. These labels are used by
   certain reward functions and observations during training (e.g., contact-aware
   tracking rewards).

6. **Quality filtering** (enabled by default): Motions are rejected if they have
   extreme velocities, underground body parts, or unnaturally airborne segments.
   Use ``--ignore-motion-filter`` to disable.

.. _seed-bvh-scaling-up:

Scaling Up: Parallel Conversion
-------------------------------

For the full BONES-SEED dataset (~142K BVH files), single-process conversion can be slow. 
The converter supports **chunk-based parallelism** via ``--num-rank``
and ``--slurm-rank``: each rank processes a deterministic subset of files (assigned
via SHA-256 hash), so all ranks run independently.

Splitting into chunks also keeps each packaged ``.pt`` file at a manageable size.
Loading the entire dataset into a single file would require excessive memory.

**Example: 15 parallel workers with SLURM**

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=seed_bvh_to_proto
   #SBATCH --array=0-14
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=64G
   #SBATCH --time=8:00:00

   CHUNK=${SLURM_ARRAY_TASK_ID}
   CHUNK_DIR=/path/to/output/chunk_$(printf '%02d' $CHUNK)
   CHUNK_PT=/path/to/output/chunk_$(printf '%02d' $CHUNK).pt

   python data/scripts/convert_soma23_bvh_to_proto.py \
       --input-dir /path/to/bones-seed/soma_uniform/bvh \
       --output-dir $CHUNK_DIR \
       --input-fps 120 --output-fps 30 \
       --num-rank 15 --slurm-rank $CHUNK

   python protomotions/components/motion_lib.py \
       --motion-path $CHUNK_DIR/ \
       --output-file $CHUNK_PT

**Without SLURM (GNU parallel / shell loop):**

.. code-block:: bash

   for RANK in $(seq 0 14); do
       python data/scripts/convert_soma23_bvh_to_proto.py \
           --input-dir /path/to/bones-seed/soma_uniform/bvh \
           --output-dir /path/to/output/chunk_$(printf '%02d' $RANK) \
           --input-fps 120 --output-fps 30 \
           --num-rank 15 --slurm-rank $RANK &
   done
   wait

   for RANK in $(seq 0 14); do
       python protomotions/components/motion_lib.py \
           --motion-path /path/to/output/chunk_$(printf '%02d' $RANK)/ \
           --output-file /path/to/output/chunk_$(printf '%02d' $RANK).pt
   done

.. _seed-bvh-training:

Training with MotionLib
-----------------------

**Single-file MotionLib** (small datasets):

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name soma23 \
       --simulator isaacgym \
       --experiment-path examples/experiments/mimic/mlp.py \
       --experiment-name soma23_seed_bvh \
       --motion-file /path/to/seed_bvh_motions.pt \
       --num-envs 4096 \
       --batch-size 16384

**Sharded MotionLib** (full BONES-SEED, multi-GPU):

For the full dataset, use sharded chunks so each GPU only loads one chunk into memory.
Name your chunk files with the ``slurmrank`` placeholder pattern, e.g.
``chunk_slurmrank.pt``. At runtime, ``MotionLib`` (see
``protomotions/components/motion_lib.py:process_packaged_motion_file_name_multi_gpu``)
discovers all matching files (``chunk_00.pt``, ``chunk_01.pt``, ...) and assigns each
GPU rank a chunk via round-robin (``rank % num_chunks``).

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name soma23 \
       --simulator isaaclab \
       --experiment-path examples/experiments/mimic/mlp.py \
       --experiment-name soma23_seed_bvh_allchunks \
       --motion-file /path/to/output/chunk_slurmrank.pt \
       --ngpu 8 --nodes 3 \
       --num-envs 8192 \
       --batch-size 16384 \
       --training-max-steps 10000000000000 \
       --use-slurm --use-wandb

This launches 24 GPUs (3 nodes × 8 GPUs), each loading one of the 15 chunks
(wrapped around via ``rank % 15``). Each GPU trains on its own motion subset,
keeping per-GPU memory usage bounded.

Next Steps
----------

* :doc:`kimodo_preparation` - Prepare motions generated by Kimodo
* :doc:`../tutorials/workflows/amass_smpl` - SMPL training workflow
