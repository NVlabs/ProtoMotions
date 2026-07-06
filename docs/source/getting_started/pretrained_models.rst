Pre-trained Models
==================

Each shipped checkpoint has a model card beside it with the training recipe,
input and output contract, artifact roles, intended use, and limitations.

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
   * - SMPL MaskedMimic
     - Sparse and masked future-motion control
     - IsaacLab only
     - `Read card <https://github.com/NVlabs/ProtoMotions/blob/main/data/pretrained_models/masked_mimic/smpl/MODEL_CARD.md>`__
   * - G1 BONES-SEED deployment tracker
     - Domain-randomized motion tracking and deployment
     - Trained in IsaacLab; expected to transfer
     - `Read card <https://github.com/NVlabs/ProtoMotions/blob/main/data/pretrained_models/motion_tracker/g1-bones-deploy/MODEL_CARD.md>`__
   * - SMPL AMASS flat tracker
     - General SMPL motion tracking on flat ground
     - IsaacLab only
     - `Read card <https://github.com/NVlabs/ProtoMotions/blob/main/data/pretrained_models/motion_tracker/smpl/MODEL_CARD.md>`__
   * - SMPL AMASS terrain tracker
     - General SMPL motion tracking on procedural terrain
     - IsaacLab only
     - `Read card <https://github.com/NVlabs/ProtoMotions/blob/main/data/pretrained_models/motion_tracker/smpl-terrains/MODEL_CARD.md>`__
   * - SOMA BONES-SEED tracker
     - Continuous-action SOMA motion tracking
     - IsaacLab only
     - `Read card <https://github.com/NVlabs/ProtoMotions/blob/main/data/pretrained_models/motion_tracker/soma-bones/MODEL_CARD.md>`__
   * - SOMA BONES-SEED FSQ tracker
     - Discrete FSQ tracker used by GPC
     - IsaacLab only
     - `Read card <https://github.com/NVlabs/ProtoMotions/blob/main/data/pretrained_models/motion_tracker/soma_bones_fsq/MODEL_CARD.md>`__
   * - SOMA GPC prior
     - Releasing soon
     - Not yet released
     - Model card will be added with the release
