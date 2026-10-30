Pre-trained Models
==================

Each shipped checkpoint has a model card beside it with the training recipe,
input and output contract, artifact roles, intended use, and limitations.

NVIDIA-provided pretrained models use the **NVIDIA Open Model License** unless
their card explicitly specifies otherwise; ProtoMotions source code remains
Apache-2.0. Each model directory includes a full license and attribution notice.
Keep these files with redistributed checkpoints and exported models. See the
`model licensing policy <https://github.com/NVlabs/ProtoMotions/blob/main/legal/MODEL-LICENSING.md>`__
for details. Training datasets and robot assets retain their separate terms.

Unless its model card says otherwise, a policy should be used on its training
simulator. The recommended checkpoints were trained or fine-tuned in IsaacLab.
The G1 deployment tracker is the only current policy trained with the full
transfer-oriented domain-randomization recipe, so it is expected to transfer
to compatible simulators and hardware; this is not a guarantee for every
backend or version.

.. list-table::
   :header-rows: 1
   :widths: 24 31 25 20

   * - Model
     - Purpose
     - Runtime expectation
     - Model card
   * - SOMA BONES-SEED GPC prior
     - Discrete GPC prior for SOMA motion
     - IsaacLab only
     - `Read card <https://github.com/NVlabs/ProtoMotions/blob/main/data/pretrained_models/gpc_prior/soma_bones/MODEL_CARD.md>`__
   * - G1 BONES-SEED deployment tracker
     - Domain-randomized motion tracking and deployment
     - Trained in IsaacLab; expected to transfer
     - `Read card <https://github.com/NVlabs/ProtoMotions/blob/main/data/pretrained_models/motion_tracker/g1-bones-deploy/MODEL_CARD.md>`__
   * - Metatron v1 G1 (MaskedMimic)
     - Motion, recovery and scene-conditioned control
     - Trained in IsaacLab; expected to transfer
     - `Read card <https://huggingface.co/nvidia/Metatron-v1/blob/main/G1/MODEL_CARD.md>`__
   * - SOMA BONES-SEED tracker
     - Continuous-action SOMA motion tracking
     - IsaacLab fine-tune only (``inference_last_lab.ckpt``)
     - `Read card <https://github.com/NVlabs/ProtoMotions/blob/main/data/pretrained_models/motion_tracker/soma-bones/MODEL_CARD.md>`__
   * - SOMA BONES-SEED FSQ tracker
     - Discrete FSQ tracker used by GPC
     - IsaacLab only
     - `Read card <https://github.com/NVlabs/ProtoMotions/blob/main/data/pretrained_models/motion_tracker/soma_bones_fsq/MODEL_CARD.md>`__

Download Metatron v1 G1
-----------------------

The native package lives on `Hugging Face <https://huggingface.co/nvidia/Metatron-v1/tree/main/G1>`__,
not in Git LFS. From the repository root, after installing ProtoMotions and the
MuJoCo requirements:

.. code-block:: bash

   python scripts/download_metatron.py
   python protomotions/inference_agent.py --simulator mujoco --num-envs 1 \
       --checkpoint data/pretrained_models/metatron/G1/inference_last.ckpt \
       --inference-config examples/inference/metatron.py

The standard-library helper downloads about 460 MB, pins an immutable Hub
revision, verifies sizes and SHA256 digests, and includes the model card,
license, notice and all configuration sidecars. No account or token is required.
Verified files are reused; an existing mismatched file is refused, not overwritten.
Use ``--output-dir`` for another location and adjust ``--checkpoint`` accordingly.
``--configs-only`` fetches the small sidecars for configuration tests, without
weights; it is not sufficient for inference. Keep the package files together.

The playback config starts the robot idle; pressing ``X`` in the viewer cycles
four goal modes (idle, walk to a target location, follow a velocity command,
reach with the right hand), and the scene stream is published empty by the saved
configuration's ``scene_boxes`` control. Its generic ONNX re-export takes the
same eight ``scene_boxes_*`` inputs from the host, with a host contract that
differs from the browser graphs' (no simulator root pose; the current action and
the eight-slot history as separate inputs). The separate ``demo/model.onnx`` and
``racer/network.onnx`` interfaces remain available in the same Hub repository;
use each with its paired application. No pretrained Metatron SOMA policy is
included.
