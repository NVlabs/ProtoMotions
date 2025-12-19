PHUMA Data Preparation
======================

This guide covers converting raw PHUMA motion data into ProtoMotions format for training.

Overview
--------

The conversion pipeline transforms PHUMA ``.npy`` files (root translations, root orientations, DOF positions, and FPS) 
into ProtoMotions ``.motion`` files containing full rigid body states (positions, rotations, 
velocities, contacts). These are then packaged into a single ``.pt`` MotionLib file for 
efficient training.

.. code-block:: text

   PHUMA .npy files
        │
        ▼ (convert_phuma_to_proto.py)
   ProtoMotions .motion files
        │
        ▼ (motion_lib.py --motion-path)
   Packaged .pt MotionLib

Prerequisites
-------------

Downloading PHUMA Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the PHUMA dataset from `DAVIAN-Robotics/PHUMA <https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA>`_.

Perform the following steps to download the PHUMA dataset:

1. Clone the `PHUMA <https://github.com/DAVIAN-Robotics/PHUMA>`_ repo and follow the installation instructions in the PHUMA repo. We recommend using a clean conda environment.
2. run `bash setup_phuma.sh`

Errors related to huggingface may be solved by running 

.. code-block:: bash

   conda install -c conda-forge huggingface_hub

The dataset is downloaded into the ``data/PHUMA/data`` folder, contains the retargeted data for 2 different humanoids: ``g1`` and ``h1_2``.

Quick Start: Convenience Script
-------------------------------

For a one-click solution, use the provided convenience script that runs the full 
conversion and packaging pipeline:

.. code-block:: bash

   cd /path/to/ProtoMotions
   python data/scripts/convert_phuma_to_motionlib.py <phuma_root_dir> <output_dir> \
        --humanoid-type <humanoid_type> \
        --motion-config <config1.yaml> [--motion-config <config2.yaml> ...]

**Arguments:**

* ``phuma_root_dir``: Root directory containing PHUMA subfolders with ``g1`` and ``h1_2`` folders.
* ``output_dir``: Directory for the packaged ``.pt`` MotionLib files
* ``--motion-config``: YAML file(s) containing motion configurations. Each creates a separate ``.pt`` file. Can be specified multiple times.
* ``--humanoid-type``: ``g1`` or ``h1_2``. Default: ``g1``
* ``--force-remake``: Overwrite existing ``.motion`` files
* ``--device``: Device to use for packaging (``cpu`` or ``cuda``). Default: ``cpu``


**Example:**

.. code-block:: bash

    # Use g1 humanoid
    python data/scripts/convert_phuma_to_motionlib.py /path/to/PHUMA/data /path/to/output \
        --humanoid-type g1 \
        --motion-config data/yaml_files/g1_phuma_train.yaml \
        --motion-config data/yaml_files/g1_phuma_val.yaml \
        --motion-config data/yaml_files/g1_phuma_unseen_video.yaml

    # Use h1_2 humanoid
    python data/scripts/convert_phuma_to_motionlib.py /path/to/PHUMA/data /path/to/output \
        --humanoid-type h1_2 \
        --motion-config data/yaml_files/h1_2_phuma_train.yaml \
        --motion-config data/yaml_files/h1_2_phuma_val.yaml \
        --motion-config data/yaml_files/h1_2_phuma_unseen_video.yaml

The script runs all steps automatically and outputs the final MotionLib ``.pt`` files:

.. code-block:: text

    output/
       ├── g1_phuma_train.pt
       ├── g1_phuma_val.pt
       ├── g1_phuma_unseen_video.pt
       ├── h1_2_phuma_train.pt
       ├── h1_2_phuma_val.pt
       └── h1_2_phuma_unseen_video.pt

Step-by-Step Guide
------------------

Step 1: Convert PHUMA to .motion Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert PHUMA ``.npy`` files to ProtoMotions ``.motion`` format:

.. code-block:: bash

   # Convert the PHUMA data to ProtoMotions format for the g1 humanoid
   python data/scripts/convert_phuma_to_proto.py \
        /path/to/PHUMA/data \
        --humanoid-type g1 \
        --force-remake
        
   # Convert the PHUMA data to ProtoMotions format for the h1_2 humanoid
   python data/scripts/convert_phuma_to_proto.py \
        /path/to/PHUMA/data \
        --humanoid-type h1_2 \
        --force-remake

**Arguments:**

* ``phuma_root_dir``: Root directory containing PHUMA subfolders with ``g1`` and ``h1_2`` folders.
* ``--output-dir``: Directory for the ``.motion`` files. Default: ``phuma_root_dir/data/humanoid_type``.
* ``--humanoid-type``: ``g1`` or ``h1_2``. Default: ``g1``
* ``--force-remake``: Overwrite existing ``.motion`` files


What the Script Does:

1. Load the PHUMA data: Reads ``.npy`` files containing:

   - ``root_trans``: Root translations (T, 3)
   - ``root_ori``: Root orientations (T, 4) in (x, y, z, w) format
   - ``dof_pos``: DOF positions (T, 29)
   - ``fps``: Frames per second

2. Convert the PHUMA data to ProtoMotions format:

   - Convert the quaternion from PHUMA (xyzw) to MuJoCo (wxyz) format
   - Build qpos: [root_pos(3), root_quat_wxyz(4), dof_pos(29)] = (T, 36)
   - Convert dof_pos to tensor
   - Perform forward kinematics to get rigid body positions and rotations
   - Compute dof_vel using finite differences
   - Compute contact labels based on position and velocity
   - Return the motion data

3. Save the motion data to the output directory:

   - Saves the motion data to the output directory as ``.motion`` files

Step 2: Package the .motion Files into a .pt MotionLib File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Package the ``.motion`` files into a ``.pt`` MotionLib file:

.. note::
   The YAML config files contain relative paths (e.g., ``g1/animation/...``).
   ``motion_lib.py`` resolves these paths relative to the YAML file's directory.
   Therefore, you must copy the YAML files to the PHUMA data directory first.

.. code-block:: bash

   # Package the g1 humanoid motion files into a .pt MotionLib file for the training set
   cp data/yaml_files/g1_phuma_train.yaml /path/to/PHUMA/data/
   python protomotions/components/motion_lib.py \
        --motion-path /path/to/PHUMA/data/g1_phuma_train.yaml \
        --output-file /path/to/g1_phuma_train.pt \
        --device cpu

   # Package the g1 humanoid motion files into a .pt MotionLib file for the validation set
   cp data/yaml_files/g1_phuma_val.yaml /path/to/PHUMA/data/
   python protomotions/components/motion_lib.py \
        --motion-path /path/to/PHUMA/data/g1_phuma_val.yaml \
        --output-file /path/to/g1_phuma_val.pt \
        --device cpu

   # Package the g1 humanoid motion files into a .pt MotionLib file for the unseen video set
   cp data/yaml_files/g1_phuma_unseen_video.yaml /path/to/PHUMA/data/
   python protomotions/components/motion_lib.py \
        --motion-path /path/to/PHUMA/data/g1_phuma_unseen_video.yaml \
        --output-file /path/to/g1_phuma_unseen_video.pt \
        --device cpu

**Arguments:**

* ``--motion-path``: Path to YAML config file listing motions to package (or directory of ``.motion`` files)
* ``--output-file``: Output path for the packaged ``.pt`` file
* ``--device``: Device to use (``cpu`` or ``cuda``)


Step 3: Verify with Motion Visualizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before training, verify the converted motions look correct using the motion 
visualizer:

.. code-block:: bash

   python examples/motion_libs_visualizer.py \
       --motion_files /path/to/g1_phuma_train.pt \
       --robot g1 \
       --simulator isaaclab

**Controls:**

* **R**: Switch to next motion
* **1/2**: Increase/decrease playback speed
* **3/4**: Adjust smoothness threshold for highlighting

YAML Motion Config Format
-------------------------

YAML config files define which motions to include, with optional timing slices 
and sampling weights:

.. code-block:: yaml

   motions:
     - file: g1/animation/Ways_to_Stand_Winded_clip1_chunk_0000.motion
       fps: 30.0
       weight: 1.0
     - file: g1/animation/Ways_to_Sit_Buying_a_Chair_clip1_chunk_0000.motion
       fps: 30.0
       weight: 1.0
     - file: g1/animation/Ways_to_Open_a_Christmas_Gift_the_boss_clip1_chunk_0000.motion
       fps: 30.0
       weight: 1.0

Fields:

* ``file``: Path to motion file (relative to phuma_root_dir)
* ``fps``: Original motion capture FPS
* ``weight``: Sampling weight (higher = sampled more often during training)

.. tip::
   Pre-built YAML configs for standard PHUMA g1 and h1_2 train/val/unseen_video splits are 
   available in ``data/yaml_files/``:
   
   * ``g1_phuma_train.yaml``
   * ``g1_phuma_val.yaml``
   * ``g1_phuma_unseen_video.yaml``
   * ``h1_2_phuma_train.yaml``
   * ``h1_2_phuma_val.yaml``
   * ``h1_2_phuma_unseen_video.yaml``


G1 vs H1-2
----------

* **G1**: 29 joints (28 non-root).
* **H1-2**: 27 joints (26 non-root).


Training with MotionLib
-----------------------

Once packaged, use in training:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name g1 \
       --simulator isaacgym \
       --experiment-path examples/experiments/mimic/mlp.py \
       --experiment-name g1_phuma \
       --motion-file /path/to/g1_phuma_train.pt \
       --num-envs 4096 \
       --batch-size 16384
