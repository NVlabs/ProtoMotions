SEED G1 CSV Data Preparation
=============================

This guide covers converting BONES-SEED retargeted G1 CSV motion data into
ProtoMotions format for the Unitree G1 humanoid robot.

.. note::

   The BONES-SEED dataset contains **~142K motions** — significantly larger than AMASS
   (~15K motions). At this scale:

   * **Conversion** should be parallelized across multiple CPUs (see
     :ref:`seed-g1-scaling-up`).
   * **Training** should use sharded MotionLib chunks so each GPU only loads a subset
     of motions, avoiding GPU memory exhaustion (see :ref:`seed-g1-training`).

Overview
--------

The `BONES-SEED <https://huggingface.co/datasets/bones-studio/seed>`_ dataset includes motions that have
been retargeted to the Unitree G1 skeleton and exported as CSV files. The conversion
pipeline transforms these CSV files into ProtoMotions ``.motion`` format.

.. code-block:: text

   Retargeted G1 .csv files (joint angles in degrees, 120 fps)
        │
        ▼ (convert_g1_csv_to_proto.py)
   ProtoMotions .motion files (30 fps)
        │
        ▼ (motion_lib.py --motion-path)
   Packaged .pt MotionLib (one per chunk, or single file for small sets)

Prerequisites
-------------

1. **Download BONES-SEED**: Download the dataset from
   `Hugging Face <https://huggingface.co/datasets/bones-studio/seed>`_.
   After downloading, extract the G1 tar archive:

   .. code-block:: bash

      cd bones-seed
      tar -xf g1.tar

   The expected directory structure is:

   .. code-block:: text

      bones-seed/
        └── g1/
            └── csv/
                ├── <date_1>/
                │   ├── motion_001.csv
                │   └── motion_002.csv
                └── <date_2>/
                    └── ...

CSV Format
----------

Each CSV file has the following columns:

* ``Frame``: Frame index
* ``root_translateX/Y/Z``: Root position in centimeters
* ``root_rotateX/Y/Z``: Root orientation as extrinsic XYZ Euler angles in degrees
* ``<joint>_dof``: Joint angles in degrees, matching the G1 MJCF joint order

Quick Start: Small Subset
-------------------------

For a small number of CSV files (e.g., testing with a few motions):

.. code-block:: bash

   # Convert
   python data/scripts/convert_g1_csv_to_proto.py \
       --input-dir /path/to/bones-seed/g1/csv \
       --output-dir /path/to/output/motions \
       --input-fps 120 \
       --output-fps 30

   # Package into a single .pt
   python protomotions/components/motion_lib.py \
       --motion-path /path/to/output/motions/ \
       --output-file /path/to/seed_g1_motions.pt

   # Verify visually
   python examples/motion_libs_visualizer.py \
       --motion_files /path/to/seed_g1_motions.pt \
       --robot g1 \
       --simulator isaacgym

**Key arguments:**

* ``--input-dir``: Root directory to search recursively for ``.csv`` files
* ``--output-dir``: Directory to save ``.motion`` files
* ``--input-fps``: Source CSV frame rate (default: 30)
* ``--output-fps``: Target output frame rate (default: 30)
* ``--robot-type``: Robot type (default: ``g1``)
* ``--euler-order``: Euler angle convention for root rotation (default: ``xyz``)
* ``--apply-motion-filter``: Enable quality filtering
* ``--force-remake``: Overwrite existing files

The converter also computes heuristic ground contact labels for all bodies (based on
height and velocity thresholds). These labels are used by certain reward functions and
observations during training (e.g., contact-aware tracking rewards).

.. _seed-g1-scaling-up:

Scaling Up: Parallel Conversion
-------------------------------

For the full BONES-SEED dataset (~142K CSV files), the converter supports
**chunk-based parallelism** via ``--num-rank`` and ``--slurm-rank``. Each rank
processes a deterministic subset of files, so all ranks run independently.

Splitting into chunks also keeps each packaged ``.pt`` file at a manageable size.
Loading the entire dataset into a single file would require excessive GPU memory.

**SLURM array job example (24 chunks):**

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=seed_g1_csv_to_proto
   #SBATCH --array=0-23
   #SBATCH --cpus-per-task=24
   #SBATCH --mem=64G
   #SBATCH --time=8:00:00

   CHUNK=${SLURM_ARRAY_TASK_ID}
   CHUNK_DIR=/path/to/output/chunk_$(printf '%02d' $CHUNK)
   CHUNK_PT=/path/to/output/chunk_$(printf '%02d' $CHUNK).pt

   python data/scripts/convert_g1_csv_to_proto.py \
       --input-dir /path/to/bones-seed/g1/csv \
       --output-dir $CHUNK_DIR \
       --input-fps 120 --output-fps 30 \
       --apply-motion-filter \
       --robot-type g1 --euler-order xyz \
       --num-rank 24 --slurm-rank $CHUNK

   python protomotions/components/motion_lib.py \
       --motion-path $CHUNK_DIR/ \
       --output-file $CHUNK_PT

**Without SLURM (shell loop):**

.. code-block:: bash

   for RANK in $(seq 0 23); do
       python data/scripts/convert_g1_csv_to_proto.py \
           --input-dir /path/to/bones-seed/g1/csv \
           --output-dir /path/to/output/chunk_$(printf '%02d' $RANK) \
           --input-fps 120 --output-fps 30 \
           --apply-motion-filter \
           --robot-type g1 --euler-order xyz \
           --num-rank 24 --slurm-rank $RANK &
   done
   wait

   for RANK in $(seq 0 23); do
       python protomotions/components/motion_lib.py \
           --motion-path /path/to/output/chunk_$(printf '%02d' $RANK)/ \
           --output-file /path/to/output/chunk_$(printf '%02d' $RANK).pt
   done

.. _seed-g1-training:

Training with MotionLib
-----------------------

**Single-file MotionLib** (small datasets):

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name g1 \
       --simulator isaacgym \
       --experiment-path examples/experiments/mimic/mlp.py \
       --experiment-name g1_seed_csv \
       --motion-file /path/to/seed_g1_motions.pt \
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
       --robot-name g1 \
       --simulator isaaclab \
       --experiment-path examples/experiments/mimic/mlp.py \
       --experiment-name g1_seed_csv_allchunks \
       --motion-file /path/to/output/chunk_slurmrank.pt \
       --ngpu 8 --nodes 3 \
       --num-envs 8192 \
       --batch-size 16384 \
       --training-max-steps 10000000000000 \
       --use-slurm --use-wandb

This launches 24 GPUs (3 nodes × 8 GPUs), each loading one of the 24 chunks.
Each GPU trains on its own motion subset, keeping per-GPU memory usage bounded.

Next Steps
----------

* :doc:`seed_bvh_preparation` - Prepare SEED BVH data for the `SOMA <https://github.com/NVlabs/SOMA-X>`_ skeleton
* :doc:`kimodo_preparation` - Prepare motions generated by Kimodo
