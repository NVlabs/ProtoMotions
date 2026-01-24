Scalable Training with SLURM
============================

This guide covers running ProtoMotions training jobs on SLURM-managed HPC clusters.

What is SLURM?
--------------

`SLURM <https://slurm.schedmd.com/>`_ (Simple Linux Utility for Resource Management) is a 
widely-used job scheduler for high-performance computing clusters. It manages compute 
resources, queues jobs, and handles multi-node distributed workloads. Most academic and 
enterprise GPU clusters use SLURM for job scheduling.

Overview
--------

ProtoMotions provides ``train_slurm.py``, a launcher script that:

1. **Syncs your code** to the cluster (via rsync over SSH)
2. **Generates a SLURM batch script** with the correct job parameters
3. **Submits the job** to the cluster queue
4. **Handles auto-resume** via SLURM job arrays (jobs continue after timeouts)

The script is a **template** designed to be customized for your specific cluster setup.

Configuring for Your Cluster
----------------------------

Before using SLURM training, edit the configuration section at the top of 
``protomotions/train_slurm.py``:

.. code-block:: python

   # =============================================================================
   # CLUSTER CONFIGURATION - EDIT THIS SECTION FOR YOUR CLUSTER
   # =============================================================================

   # Login node hostname (e.g., "login.mycluster.edu")
   CLUSTER_LOGIN_NODE = "YOUR_CLUSTER_LOGIN_NODE"

   # Base directory for experiments on the cluster filesystem
   CLUSTER_BASE_DIR = "/path/to/your/experiments/directory"

   # Container images (Singularity .sif or Enroot .sqsh format)
   CONTAINER_IMAGES = {
       "isaacgym": "/path/to/containers/isaacgym.sqsh",
       "isaaclab": "/path/to/containers/isaaclab.sqsh",
       "newton": "/path/to/containers/newton.sqsh",
   }

   # Default SLURM account (your allocation/project)
   DEFAULT_SLURM_ACCOUNT = "your_account"

   # Default SLURM partitions
   DEFAULT_SLURM_PARTITION = "gpu"

   # Filesystem mounts for container
   CONTAINER_MOUNTS = "/scratch:/scratch:rw"

**Key settings to configure:**

* ``CLUSTER_LOGIN_NODE``: SSH hostname for your cluster's login node
* ``CLUSTER_BASE_DIR``: Directory where experiment code will be synced
* ``CONTAINER_IMAGES``: Paths to your container images (Singularity/Enroot)
* ``DEFAULT_SLURM_ACCOUNT``: Your SLURM allocation or project name
* ``CONTAINER_MOUNTS``: Filesystem paths to mount inside the container

Container Setup
~~~~~~~~~~~~~~~

You'll need containerized environments with ProtoMotions dependencies. Convert your 
Docker images to Singularity (``.sif``) or Enroot (``.sqsh``) format as required 
by your cluster.

Auto-Resume with Job Arrays
---------------------------

Long training runs often exceed cluster time limits (e.g., 4-hour walltime). 
ProtoMotions handles this automatically using two mechanisms:

**1. SLURM Job Arrays**

The launcher submits jobs as arrays (``--array=0-5%1``), meaning up to 5 sequential 
jobs will run. When a job times out, the next array task starts and resumes from 
the last checkpoint.

**2. AutoResume Callback**

When ``--use-slurm`` is enabled, training registers the ``AutoResumeCallbackSrun`` 
callback. This callback:

* Tracks elapsed training time
* Saves a checkpoint before the SLURM time limit (default: after 3.5 hours)
* Gracefully stops training so the next array job can resume

.. code-block:: python

   # From protomotions/agents/callbacks/slurm_autoresume_srun.py
   class AutoResumeCallbackSrun(Callback):
       def __init__(self, autoresume_after=12600):  # 3.5 hours in seconds
           self.autoresume_after = autoresume_after
       
       def _check_autoresume(self, agent):
           if time.time() - self.start_time >= self.autoresume_after:
               agent.save()           # Save checkpoint
               agent._should_stop = True  # Signal graceful stop

The default ``autoresume_after=12600`` (3.5 hours) works well with 4-hour job limits, 
providing buffer time for checkpoint saving.

Understanding Scaling Parameters
--------------------------------

The ``--num-envs`` and ``--batch-size`` parameters are specified **per GPU**. With 
multi-GPU and multi-node training, the effective totals scale accordingly:

.. code-block:: text

   Total GPUs = ngpu × nodes
   Effective num-envs = num-envs × Total GPUs
   Effective batch-size = batch-size × Total GPUs

**Example:**

With ``--ngpu=4 --nodes=2 --num-envs=4096 --batch-size=16384``:

* **Total GPUs**: 4 × 2 = 8 GPUs
* **Effective environments**: 4,096 × 8 = **32,768 parallel environments**
* **Effective batch size**: 16,384 × 8 = **131,072 samples per update**

This scaling is automatic—you specify per-GPU values and the distributed training 
handles aggregation across all processes.

Running a Training Job
----------------------

Once configured, launch training from your local machine:

.. code-block:: bash

   python protomotions/train_slurm.py \
       --robot-name=g1 \
       --simulator=isaaclab \
       --num-envs=4096 \
       --batch-size=16384 \
       --motion-file=/cluster/path/to/motions.pt \
       --experiment-path=examples/experiments/mimic/mlp_bm.py \
       --experiment-name=g1_motion_tracker \
       --user=myusername \
       --ngpu=4 \
       --nodes=1 \
       --slurm-time=4:00:00 \
       --use-wandb

**Key arguments:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``--robot-name``
     - Robot to train (e.g., ``g1``, ``smpl``, ``h1_2``)
   * - ``--simulator``
     - Physics backend (``isaacgym``, ``isaaclab``, ``newton``)
   * - ``--num-envs``
     - Parallel environments (scale with GPU memory)
   * - ``--batch-size``
     - PPO batch size (typically 2-4x num-envs)
   * - ``--motion-file``
     - Path to motion data **on the cluster**
   * - ``--experiment-path``
     - Experiment config file (relative to repo root)
   * - ``--experiment-name``
     - Unique name for this experiment
   * - ``--user``
     - Your cluster username
   * - ``--ngpu``
     - GPUs per node
   * - ``--nodes``
     - Number of compute nodes
   * - ``--slurm-time``
     - Job time limit (HH:MM:SS)
   * - ``--array-size``
     - Number of auto-resume attempts (default: 5)
   * - ``--use-wandb``
     - Enable Weights & Biases logging

Multi-Node Training
-------------------

For large-scale training across multiple nodes:

.. code-block:: bash

   python protomotions/train_slurm.py \
       --robot-name=smpl \
       --simulator=isaacgym \
       --num-envs=8192 \
       --batch-size=16384 \
       --motion-file=/cluster/path/to/amass_train.pt \
       --experiment-path=examples/experiments/mimic/mlp.py \
       --experiment-name=smpl_motion_tracker_4node \
       --user=myusername \
       --ngpu=8 \
       --nodes=4 \
       --slurm-time=4:00:00 \
       --use-wandb

ProtoMotions uses PyTorch Fabric for distributed training. Each node runs 
``--ngpu`` processes, and gradients are synchronized across all nodes.

Monitoring Jobs
---------------

After submission, the script prints monitoring commands:

.. code-block:: bash

   # Monitor live output
   ssh myusername@cluster 'tail -f /path/to/exp/slurm_output.log'
   
   # Check job status
   ssh myusername@cluster 'squeue -u myusername'
   
   # Cancel a job
   ssh myusername@cluster 'scancel <job_id>'

Next Steps
----------

* :doc:`configuration` - Configuration system details
* :doc:`experiments` - Creating custom experiments
* :doc:`../tutorials/workflows/domain_randomization` - Domain randomization for robust policies

