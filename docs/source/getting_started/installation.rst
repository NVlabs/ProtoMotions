Installation
============

ProtoMotions supports four simulation backends: IsaacGym, IsaacLab, Genesis, and Newton. 
You can install the simulation of your choice, and the simulation backend is selected via the configuration file.

**Tested Versions:**

.. raw:: html

   <p>
     <a href="https://github.com/newton-physics/newton/commit/8a2abf2"><img src="https://img.shields.io/badge/Newton-8a2abf2-brightgreen.svg" alt="Newton"></a>
     <a href="https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.0"><img src="https://img.shields.io/badge/IsaacLab-2.3.0-blue.svg" alt="IsaacLab"></a>
     <a href="https://developer.nvidia.com/isaac-gym"><img src="https://img.shields.io/badge/IsaacGym-Preview_4-blue.svg" alt="IsaacGym"></a>
     <a href="https://github.com/Genesis-Embodied-AI/Genesis"><img src="https://img.shields.io/badge/Genesis-untested-lightgrey.svg" alt="Genesis"></a>
   </p>

.. note::

   We recommend creating a **separate virtual environment** for each simulator to avoid dependency conflicts.
   We recommend using **conda** or **venv** for IsaacGym and Genesis, and **uv** for IsaacLab and Newton.

Prerequisites
-------------

After cloning the repository, fetch all files stored in git-lfs:

.. code-block:: bash

   git lfs fetch --all

Choose Your Simulator(s)
------------------------

IsaacGym
~~~~~~~~

IsaacGym requires **Python 3.8**.

1. Create a conda environment:

   .. code-block:: bash

      conda create -n isaacgym python=3.8
      conda activate isaacgym

2. Download IsaacGym Preview 4:

   .. code-block:: bash

      wget https://developer.nvidia.com/isaac-gym-preview-4
      tar -xvzf isaac-gym-preview-4

3. Install IsaacGym Python API:

   .. code-block:: bash

      pip install -e isaacgym/python

4. Install ProtoMotions and dependencies:

   .. code-block:: bash

      pip install -e /path/to/protomotions
      pip install -r /path/to/protomotions/requirements_isaacgym.txt

IsaacLab
~~~~~~~~

We recommend using **uv** for IsaacLab installation. IsaacLab 2.x requires **Python 3.11**.
For full installation details, see the `IsaacLab Pip Installation Guide <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html>`__.

1. Create a virtual environment with uv:

   .. code-block:: bash

      uv venv --python 3.11 env_isaaclab
      source env_isaaclab/bin/activate

2. Install PyTorch and IsaacLab:

   .. code-block:: bash

      uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
      uv pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com

3. Install ProtoMotions and dependencies:

   .. code-block:: bash

      uv pip install -e /path/to/protomotions
      uv pip install -r /path/to/protomotions/requirements_isaaclab.txt

Genesis (Experimental)
~~~~~~~~~~~~~~~~~~~~~~

Genesis requires **Python 3.10**.

1. Create a conda environment:

   .. code-block:: bash

      conda create -n genesis python=3.10
      conda activate genesis

2. Install `Genesis <https://genesis-world.readthedocs.io/en/latest/index.html>`_

3. Install ProtoMotions and dependencies:

   .. code-block:: bash

      pip install -e /path/to/protomotions
      pip install -r /path/to/protomotions/requirements_genesis.txt

Newton
~~~~~~~~~~~~~

Newton (currently in beta) is a GPU-accelerated physics simulator built on NVIDIA Warp. We recommend using **uv** for installation.
For full installation details, see the `Newton Installation Guide <https://newton-physics.github.io/newton/guide/installation.html>`__.

**Requirements**: Python 3.10+, NVIDIA GPU (compute capability >= 5.0), driver 545+

1. Clone Newton and create a virtual environment:

   .. code-block:: bash

      git clone git@github.com:newton-physics/newton.git
      cd newton
      uv venv
      source .venv/bin/activate

2. Install Newton dependencies:

   .. code-block:: bash

      uv pip install mujoco --pre -f https://py.mujoco.org/
      uv pip install warp-lang --pre -U -f https://pypi.nvidia.com/warp-lang/
      uv pip install git+https://github.com/google-deepmind/mujoco_warp.git@main
      uv pip install -e .[examples]

3. Install ProtoMotions and dependencies:

   .. code-block:: bash

      uv pip install -e /path/to/protomotions
      uv pip install -r /path/to/protomotions/requirements_newton.txt

Troubleshooting
---------------

IsaacGym Issues
~~~~~~~~~~~~~~~

**libpython Error**

If you encounter ``libpython`` related errors, you need to set the ``LD_LIBRARY_PATH`` to your conda environment:

.. code-block:: bash

   # First, check your conda environment path
   conda info -e
   
   # Then set LD_LIBRARY_PATH (replace with your actual conda env path)
   export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib:$LD_LIBRARY_PATH
   
   # For example:
   export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH

To make this permanent, add the export command to your ``~/.bashrc`` or ``~/.zshrc``:

.. code-block:: bash

   echo 'export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc

**Memory Issues**

If you run into memory issues during training:

.. code-block:: bash

   # Reduce number of environments in your training command
   --num-envs 1024

Next Steps
----------

After installation, proceed to the :doc:`quickstart` guide to train your first agent or run pre-trained models.

