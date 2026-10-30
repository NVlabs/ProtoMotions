AMASS Data Preparation
======================

This guide covers converting raw AMASS motion data into ProtoMotions format for training.

Overview
--------

The conversion pipeline transforms AMASS ``.npz`` files (axis-angle poses + root translations) 
into ProtoMotions ``.motion`` files containing full rigid body states (positions, rotations, 
velocities, contacts). These are then packaged into a single ``.pt`` MotionLib file for 
efficient training.

.. code-block:: text

   AMASS .npz files
        │
        ▼ (convert_amass_to_proto.py)
   ProtoMotions .motion files
        │
        ▼ (motion_lib.py --motion-path)
   Packaged .pt MotionLib

Prerequisites
-------------

**Download AMASS**: register and download from the `AMASS website <https://amass.is.tue.mpg.de/>`_.

No SMPL or SMPL-X body-model download is needed for this pipeline. The
converter takes the skeleton from the MJCF humanoids shipped with the
repository, ``protomotions/data/assets/mjcf/smpl_humanoid.xml`` and
``protomotions/data/assets/mjcf/smplx_humanoid.xml``, and never opens a
body-model ``.pkl`` file.

Quick Start: Convenience Script
-------------------------------

For a one-click solution, use the provided convenience script that runs the full 
conversion and packaging pipeline:

.. code-block:: bash

   python data/scripts/convert_amass_to_motionlib.py <amass_root_dir> <output_dir> \
       --motion-config <config1.yaml> [--motion-config <config2.yaml> ...]

**Arguments:**

* ``amass_root_dir``: Root directory containing AMASS subfolders with ``.npz`` files.
  Give it as an absolute path: the script rewrites each YAML ``file:`` entry to
  ``<amass_root_dir>/<file>`` and packages from a temporary YAML written in
  ``output_dir``, and ``motion_lib.py`` resolves a relative entry against that
  temporary YAML's directory, where the motions are not.
* ``output_dir``: Directory for the packaged ``.pt`` MotionLib files
* ``--motion-config``: YAML file(s) containing motion configurations. Each creates a separate ``.pt`` file. Can be specified multiple times.
* ``--humanoid-type``: ``smpl`` (24 joints) or ``smplx`` (52 joints). Default: ``smpl``
* ``--output-fps``: Target output FPS for motion files. Default: 30
* ``--force-remake``: Overwrite existing ``.motion`` files
* ``--device``: Device to use for packaging (``cpu`` or ``cuda``). Default: ``cpu``

**Example:**

.. code-block:: bash

   # Create train/test/validation splits from YAML configs
   python data/scripts/convert_amass_to_motionlib.py /path/to/amass_root /path/to/output \
       --motion-config data/yaml_files/amass_smpl_train.yaml \
       --motion-config data/yaml_files/amass_smpl_test.yaml \
       --motion-config data/yaml_files/amass_smpl_validation.yaml

   # Use SMPL-X humanoid
   python data/scripts/convert_amass_to_motionlib.py /path/to/amass_root /path/to/output \
       --humanoid-type smplx \
       --motion-config data/yaml_files/amass_smplx_train.yaml

The script runs all steps automatically and outputs the final MotionLib ``.pt`` files:

.. code-block:: text

   output/
     ├── amass_smpl_train.pt
     ├── amass_smpl_test.pt
     └── amass_smpl_validation.pt

Step-by-Step Guide
------------------

Step 1: Convert AMASS to .motion Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert AMASS ``.npz`` files to ProtoMotions ``.motion`` format:

.. code-block:: bash

   python data/scripts/convert_amass_to_proto.py \
       /path/to/amass_root \
       --humanoid-type smpl \
       --output-fps 30 \
       --motion-config data/yaml_files/amass_smpl_train.yaml \
       --motion-config data/yaml_files/amass_smpl_test.yaml \
       --motion-config data/yaml_files/amass_smpl_validation.yaml \
       --force-remake

**Arguments:**

* ``amass_root_dir``: Root directory containing AMASS subfolders
* ``--humanoid-type``: ``smpl`` (24 joints) or ``smplx`` (52 joints)
* ``--output-fps``: Target output FPS (default: 30)
* ``--force-remake``: Overwrite existing converted files
* ``--motion-config``: YAML files for motion selection and slicing. Can be specified multiple times.

The ``--motion-config`` YAML files define the start and end times for each motion. 
These time ranges specify physically plausible segments within each raw AMASS recording 
(e.g., excluding T-pose calibration frames or problematic sections).

**What the Script Does:**

1. **Load AMASS data**: Reads ``.npz`` files containing:
   
   * ``poses``: Joint rotations in axis-angle format
   * ``trans``: Root translations
   * ``mocap_framerate``: Original capture FPS

2. **Downsample**: picks the smallest divisor of the source FPS that is still
   >= the target FPS and keeps every (source / divisor)-th frame, so the output
   lands as close to the target as the source rate allows. For example, a 120
   FPS source with a 30 FPS target uses every 4th frame.

3. **Forward Kinematics**: Computes full rigid body state:
   
   * World positions and rotations for all bodies
   * Linear and angular velocities (via finite differences)
   * DOF positions and velocities

4. **Contact Detection**: Labels ground contacts using thresholds:
   
   * Velocity threshold: 0.15 m/s
   * Height threshold: 0.1 m

5. **Height Fixing**: Adjusts motion so feet don't penetrate the ground or float 
   above it. Shifts the entire motion vertically so the lowest joint during the 
   motion matches the toe offset (height of the toe from ground in T-pose): 
   0.015m for SMPL, 0.017m for SMPL-X.

Step 2: Package into MotionLib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Package individual ``.motion`` files into a single ``.pt`` file. The YAML config
determines which motions are included in each package.

``motion_lib.py`` resolves every ``file:`` entry relative to the directory that
contains the YAML, and it stops with ``FileNotFoundError`` if the first entry
does not exist there. The shipped split YAMLs list paths relative to the AMASS
root, for example ``CMU/45/45_01_poses.motion``, and Step 1 writes each
``.motion`` file next to its source ``.npz`` under that root. So copy the YAML
into the AMASS root first and point ``--motion-path`` at the copy. The
convenience script does the same job for you by writing a temporary YAML whose
entries are absolute paths.

.. code-block:: bash

   # Package training set
   cp data/yaml_files/amass_smpl_train.yaml /path/to/amass_root/
   python protomotions/components/motion_lib.py \
       --motion-path /path/to/amass_root/amass_smpl_train.yaml \
       --output-file /path/to/amass_smpl_train.pt \
       --device cpu

   # Package test set
   cp data/yaml_files/amass_smpl_test.yaml /path/to/amass_root/
   python protomotions/components/motion_lib.py \
       --motion-path /path/to/amass_root/amass_smpl_test.yaml \
       --output-file /path/to/amass_smpl_test.pt \
       --device cpu

**Arguments:**

* ``--motion-path``: Path to YAML config file listing motions to package (or directory of ``.motion`` files)
* ``--output-file``: Output path for the packaged ``.pt`` file
* ``--device``: Device to use (``cpu`` or ``cuda``)

The YAML config specifies which motions belong to each split (train/test/validation),
so you run this step once per config to create separate ``.pt`` files.

**Why package?** While you can load motions directly from a YAML file, packaging
pre-processes the dataset once and saves it as a single ``.pt`` file. This makes
subsequent loads much faster since the motions don't need to be re-processed each time.

**Output:**

.. code-block:: text

   Loading motions from yaml/npy file or Directory of motions which is slower
   Loaded 1234 motions with a total length of 12345.6s.
   Motion library saved to amass_train.pt

Step 3: Verify with Motion Visualizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before training, verify the converted motions look correct using the motion 
visualizer:

.. code-block:: bash

   python examples/motion_libs_visualizer.py \
       --motion_files /path/to/amass_smpl_train.pt \
       --robot smpl \
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
     - file: CMU/45/45_01_poses.motion
       fps: 120.0
       weight: 1.0
       sub_motions:
         - timings:
             start: 0.0
             end: 3.8
     - file: CMU/138/138_06_poses.motion
       fps: 120.0
       weight: 2.0  # Sample this motion twice as often
       sub_motions:
         - timings:
             start: 0.5
             end: 3.0

**Fields:**

* ``file``: Path to the motion file, relative to ``amass_root_dir``. ``motion_lib.py`` resolves it relative to the YAML's own directory, which is why Step 2 copies the YAML there
* ``fps``: Original motion capture FPS. Informational only: neither script reads it; the frame rate comes from the ``.npz`` file or from ``motion_fps_amassx.yaml``
* ``weight``: Sampling weight (higher = sampled more often during training)
* ``sub_motions``: The time slice to extract. Only the first entry is used; further entries are ignored without a warning
* ``timings.start``: Start time in seconds
* ``timings.end``: End time in seconds

.. tip::

   Pre-built YAML configs for standard AMASS train/test/validation splits are 
   available in ``data/yaml_files/``:
   
   * ``amass_smpl_train.yaml``
   * ``amass_smpl_test.yaml``
   * ``amass_smpl_validation.yaml``
   * ``amass_smplx_train.yaml``
   * ``amass_smplx_test.yaml``
   * ``amass_smplx_validation.yaml``

SMPL vs SMPL-X
--------------

* **SMPL**: 24 joints (23 non-root). Simpler body model.
* **SMPL-X**: 52 joints (51 non-root). Includes hands with fingers.

For SMPL-X, the frame rate is taken from ``data/yaml_files/motion_fps_amassx.yaml``
when the motion has an entry there, and that entry wins over whatever the ``.npz``
file carries; a motion with no entry falls back to the ``mocap_framerate`` stored
in the ``.npz`` file.

Training with MotionLib
-----------------------

Once packaged, use in training. The IsaacGym command uses
``mimic/mlp_no_contact.py`` rather than ``mimic/mlp.py`` because IsaacGym's GPU
pipeline reports one combined force per body, so it cannot run a recipe that
declares contact bodies; on IsaacLab use ``mimic/mlp.py``.

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name smpl \
       --simulator isaacgym \
       --experiment-path examples/experiments/mimic/mlp_no_contact.py \
       --experiment-name smpl_amass \
       --motion-file /path/to/amass_train.pt \
       --num-envs 4096 \
       --batch-size 16384

Multi-GPU Training with Sharded MotionLibs
------------------------------------------

For very large datasets, you can shard the motion library across GPUs. Name files 
with the pattern ``*_slurmrank.pt``:

.. code-block:: text

   amass_0.pt  # Loaded by rank 0
   amass_1.pt  # Loaded by rank 1
   amass_2.pt  # Loaded by rank 2
   amass_3.pt  # Loaded by rank 3

Use ``amass_slurmrank.pt`` as the motion file path, and each rank will automatically 
load its corresponding shard.

Troubleshooting
---------------

**Missing FPS in SMPL-X files**: Ensure ``data/yaml_files/motion_fps_amassx.yaml`` 
contains FPS entries for your motions.

**Out of memory**: Process subsets of motions or use ``--device cpu``.

**Motions look wrong**: Verify the humanoid type matches your AMASS download 
(SMPL vs SMPL-X).

Next Steps
----------

* :doc:`../tutorials/workflows/amass_smpl` - Full SMPL training workflow
* :doc:`../tutorials/workflows/retargeting_pyroki` - Retarget motions to robots
