Configuration System
====================

An experiment is a Python file that builds dataclass configs. It connects a
robot and simulator to environment components and an agent. Start from
``examples/experiments/mimic/mlp.py`` for motion tracking, or
``examples/experiments/steering/mlp.py`` for a task without reference motions.

**The experiment builds a new run; the saved config defines an existing run.**
Editing an experiment does not change a resumed run or a saved config bundle.

Experiment File Structure
-------------------------

Each experiment file (e.g., ``examples/experiments/mimic/mlp.py``) defines a complete
training configuration through a set of functions:

.. code-block:: python

   # Required functions
   def terrain_config(args) -> TerrainConfig:
       """Build terrain configuration."""
       return TerrainConfig()

   def scene_lib_config(args) -> SceneLibConfig:
       """Build scene library configuration."""
       return SceneLibConfig(scene_file=args.scenes_file)

   def motion_lib_config(args) -> MotionLibConfig:
       """Build motion library configuration."""
       return MotionLibConfig(motion_file=args.motion_file)

   def env_config(robot_cfg, args) -> EnvConfig:
       """Wire controls, observations, rewards, terminations, and actions."""
       ...

   # Optional functions
   def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
       """Customize robot and simulator settings."""
       simulator_cfg.domain_randomization = None

   def agent_config(robot_cfg, env_cfg, args) -> PPOAgentConfig:
       """Build the model and algorithm; required when training an agent."""
       ...

   def apply_inference_overrides(configs: ExperimentConfigs, args) -> None:
       """Adjust the inference view; runs when a checkpoint is loaded."""
       configs.simulator.domain_randomization = None

This shows the interface, not a complete runnable experiment. See
``examples/experiments/format.py`` for the hook signatures. Scene and motion
factories return configs even when their file fields are ``None``.

Config Building Flow
~~~~~~~~~~~~~~~~~~~~

When you run ``train_agent.py``, configs are built in this order:

1. Load robot config from factory (``--robot-name``)
2. Load simulator config (``--simulator``)
3. Call ``configure_robot_and_simulator()`` for customization
4. Apply the ``robot.*`` and ``simulator.*`` CLI overrides, so every builder
   that follows sees the overridden robot
5. Build ``terrain_config()``, ``scene_lib_config()``, ``motion_lib_config()``
6. Build ``env_config()``
7. Build ``agent_config()``
8. Apply the remaining CLI overrides (``terrain``, ``scene_lib``,
   ``motion_lib``, ``env`` and ``agent``)
9. Save the training configs to ``resolved_configs.json``

If a builder step changes a value that one of the early overrides set, the
build stops and names that step.

There is one saved bundle. ``inference_agent.py`` loads it and derives its
inference view at load, in this order: framework defaults, then ``--overrides``,
then the recipe's ``apply_inference_overrides()`` hook, then the hook from
``--inference-config path/to/file.py`` if you pass one. The recipe hook is
imported from the ``experiment_config.py`` saved beside the checkpoint, so that
file has to stay next to ``resolved_configs.json``; the file you gave as
``--experiment-path`` is not read again. An ``--inference-config`` file defines
``apply_inference_overrides(configs, args)`` with that same two-parameter
signature; the older eight-parameter form is refused. Because the overrides are
applied before the hooks run, a hook cannot be overridden from the command line:
a hook that changes a value you also gave in ``--overrides`` stops inference
with a message naming that override.

A missing file or an unknown name is refused by name, never silently defaulted.
Silently substituting a default for a missing artifact or a mistyped name has
repeatedly turned a typo or an absent file into a run that looked correct but
used the wrong configuration. To keep a field's default, leave it out of
``--overrides``: an empty value such as ``env.max_episode_length=`` does not
reset the field, it is refused as a type mismatch (on a string field it would
be stored as an empty string). A ``--inference-config`` path that does not
exist, or an ``--overrides`` key naming a field the configs do not define,
stops the run and names the file or the field instead of continuing on a
substituted default. An experiment that defines no ``apply_inference_overrides``
simply gets the framework's inference defaults.

Evaluation rules belong in ``agent_cfg.evaluator``: ``metric_components`` name
measurements and ``failure_components`` name threshold rules over per-trial
reductions (any frame, sum, mean, min, max or last).
They are independent of training episode terminations. See :doc:`evaluation`
for examples, saved-config migration and checkpoint-score behavior.

Robot Configurations
--------------------

Robot configs in ``protomotions/robot_configs/`` define the robot. Abridged
from the G1 config:

.. code-block:: python

   @dataclass
   class G1RobotConfig(RobotConfig):
       semantic_forward_axis_xy: Tuple[float, float] = (1.0, 0.0)

       # Map common names to robot-specific body names
       common_naming_to_robot_body_names: Dict[str, List[str]] = field(
           default_factory=lambda: {
               "all_left_foot_bodies": ["left_ankle_roll_link"],
               "all_right_foot_bodies": ["right_ankle_roll_link"],
               "head_body_name": ["head"],
               "torso_body_name": ["torso_link"],
           }
       )
       
       # Asset configuration (IsaacLab derives USD from MJCF at scene build)
       asset: RobotAssetConfig = field(default_factory=lambda: RobotAssetConfig(
           asset_file_name="mjcf/g1_holo_compat.xml",
       ))
       
       # PD gains and limits per joint (regex patterns), taken from the
       # motor classes
       control: ControlConfig = field(default_factory=lambda: ControlConfig(
           control_type=ControlType.BUILT_IN_PD,
           override_control_info={
               ".*_hip_(pitch|yaw)_joint": G1_7520_14.control_info(),
               ".*_knee_joint": G1_7520_22.control_info(),
           }
       ))
       
       # 50 Hz policy. Each backend's physics rate is derived from this
       # (isaacgym 100, isaaclab/newton 200, mujoco 1000 fps), so there is
       # no decimation field to set.
       control_hz: int = 50

Using Configurations
--------------------

Marker Body Mass and Inertia
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before importing a robot, the shared loading code resolves its MJCF masses.
A zero-mass body (typically a fixed tracking marker) receives **0.001 kg (1 g)**
and diagonal inertia of **4e-8 kg m²** at its local origin. These inertias
correspond to a 1 g solid sphere with a 1 cm radius; the visual marker geometry
does not need to change. This convention avoids the extremely small inertia
of a visualization-sized sphere.

Source MJCF files are unchanged. The loader prepares a temporary asset with
explicit marker inertias, which IsaacGym, IsaacLab, Newton and MuJoCo
all consume. IsaacLab performs this step before MJCF-to-USD conversion and
invalidates older conversion caches. PhysX articulation links cannot have zero
mass; leaving a marker massless can silently give it the simulator's default
mass instead.

Ordinary physical bodies retain their authored or geometry-derived mass.
Disabling mass randomization does not make a body massless: it uses the MJCF
values after normalization. When enabled, body-mass randomization overrides
the selected bodies and scales their inertia from the positive source properties.

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name g1 \
       --simulator isaacgym \
       --experiment-path examples/experiments/mimic/mlp_no_contact.py \
       --experiment-name my_experiment \
       --motion-file path/to/motions.pt \
       --num-envs 4096 \
       --batch-size 16384

On IsaacGym use the ``_no_contact`` mimic recipes: IsaacGym's GPU pipeline
reports one combined force per body, so it cannot run a recipe that declares
contact bodies. On IsaacLab the original ``mimic/mlp.py`` runs as is.

CLI Overrides
~~~~~~~~~~~~~

Override nested config values with ``--overrides``:

.. code-block:: bash

   --overrides "agent.num_mini_epochs=4" "env.max_episode_length=500"

   # Change a reward weight; weights live in the component's RewardMeta
   --overrides "env.reward_components.pow_rew.meta.weight=2e-6"

   # Disable simulator domain randomization
   --overrides "simulator.domain_randomization=None"

Paths start with ``robot``, ``simulator``, ``terrain``, ``scene_lib``,
``motion_lib``, ``env``, or ``agent``. Nested dictionaries use the same dot
notation as dataclass fields. Values use Python literals (``True``, ``False``,
``None``). Overrides change existing fields; define new components in Python.

A reward weight cannot be set to ``0.0``: ``RewardMeta`` refuses a zero weight.
To drop a reward term, delete its entry from ``reward_components`` in a copy of
the experiment file, as ``mimic/mlp_no_contact.py`` does for
``contact_match_rew``, and train under a new experiment name.

Saved Configurations
--------------------

All configurations are saved for reproducibility:

.. code-block:: text

   results/<experiment_name>/
   ├── config.yaml              # CLI arguments + wandb_id
   ├── resolved_configs.json    # The config bundle (tagged JSON) - source of truth
   └── experiment_config.py     # Experiment copy; inference reads its hook here

**resolved_configs.json** is the source of truth. It is a tagged JSON tree with
every value resolved -- each dataclass under its class name, enums, tensors and
module-level functions tagged, a component as its kernel name plus bindings and
parameters -- and a ``format_version`` header. Loading rebuilds every object
through its real constructor, so ``__post_init__`` runs and validation happens by
construction; a stale or hand-edited bundle fails naming the field, the class and
the file with the regenerate command. There is no pickle and no YAML sidecar: the
bundle is plain text you can read and ``git diff``. Keep ``experiment_config.py``
beside it: inference imports the recipe's ``apply_inference_overrides`` hook
from that copy and refuses to start without it.

.. warning::

   ``--overrides`` affects fresh training, warm starts, and inference, but not
   resumed training. To change a training setup, use a new experiment name and
   a weights-only warm start.

Resume Behavior
~~~~~~~~~~~~~~~

.. warning::

   **Resume uses exact saved configs.** CLI overrides are ignored during resume.
   
   If you need to change configs, start a new experiment with ``--experiment-name``.

.. list-table:: Which configuration is used?
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - Config source
     - Training state
   * - Fresh start
     - Experiment file + CLI overrides
     - New weights and optimizer
   * - Resume (existing run checkpoint)
     - ``resolved_configs.json`` beside that checkpoint
     - Restored from the checkpoint
   * - Warm start (new name + ``--checkpoint``)
     - Experiment file + CLI overrides
     - Load weights; start new training state
   * - Inference
     - ``resolved_configs.json`` + derived inference view + runtime overrides
     - No optimization

Use Old Checkpoints with New Configs or Code Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate configs without training (useful for migrating old checkpoints):

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name g1 --simulator isaacgym \
       --experiment-path examples/experiments/mimic/mlp_no_contact.py \
       --experiment-name migrated_experiment \
       --motion-file /path/to/motion.pt \
       --num-envs 4096 --batch-size 16384 \
       --create-config-only

Point ``--experiment-path`` at the recipe the checkpoint was trained with. This
example uses the ``_no_contact`` mimic recipe because it targets IsaacGym,
whose GPU pipeline reports one combined force per body and so cannot run a
recipe that declares contact bodies.

When code or config classes change, regenerate the resolved configs for public
artifacts instead of adding hidden inference-time migration hooks.
Regenerate the config bundle. This regenerates configuration,
not weights: verify that architecture, observation layout, normalization and
action semantics still match the checkpoint. Changing those contracts may
require a new model or retraining, not just replacement config files.

Component Configuration (Observations, Rewards, Terminations)
--------------------------------------------------------------

All components use ``MdpComponent`` to bind pure tensor kernels to context paths:

.. code-block:: python

   from protomotions.envs.mdp_component import MdpComponent
   from protomotions.envs.rewards.meta import RewardMeta
   from protomotions.envs.rewards.regularization import compute_action_smoothness
   from protomotions.envs.rewards.tracking import compute_gt_rew

   reward_components = {
       "gt_rew": MdpComponent.from_kernel(
           compute_gt_rew,
           meta=RewardMeta(weight=0.5),  # how the term enters the reward sum
           coefficient=-25.0,            # replaces the kernel's constant
       ),
       "action_smoothness": MdpComponent.from_kernel(
           compute_action_smoothness,
           meta=RewardMeta(weight=-0.02),
       ),
   }

``from_kernel`` reads the kernel's signature: a parameter whose default is a
context path (``EnvContext.current.rigid_body_pos``) becomes a binding, a plain
default becomes a constant, and a keyword argument replaces either. A reward's
weight is never a kernel parameter and never a ``static_params`` entry: it lives
in ``RewardMeta``, passed as ``meta=``, together with the optional clamps and
the grace-period switch. A reward component without a ``RewardMeta`` is refused
when the rewards are combined.

**Key features:**

* **Type-safe bindings**: IDE autocomplete for context paths (``EnvContext.current.rigid_body_pos``)
* **Explicit dependencies**: Bindings show exactly what data each kernel needs
* **Pure kernels**: Functions take tensors, return tensors - easy to test

**Context paths** provide dual access:

* **Class access** (for config): ``EnvContext.current.rigid_body_pos`` → FieldPath object
* **Runtime access** (the flat context map): ``ctx["current.rigid_body_pos"]`` → Tensor value

Agent/Model Configuration
-------------------------

Network architectures are composed through configs:

.. code-block:: python

   from protomotions.agents.ppo.config import PPOActorConfig
   from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig

   actor_config = PPOActorConfig(
       num_out=robot_config.number_of_actions,
       mu_key="action_mean",
       actor_logstd=-2.9,
       in_keys=["max_coords_obs", "mimic_target_poses"],
       mu_model=MLPWithConcatConfig(
           in_keys=["max_coords_obs", "mimic_target_poses"],
           out_keys=["action_mean"],
           num_out=robot_config.number_of_actions,
           normalize_obs=True,
           layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
           output_activation="tanh",
       ),
   )

The ``in_keys``/``out_keys`` system connects observations to network inputs,
and connects different network layers/modules. 
TensorDict is used to handle the data flow, which also make ONNX export easier.

PPO exploration
~~~~~~~~~~~~~~~

``PPOActorConfig.std_mode`` chooses the Gaussian exploration parameter:

* ``StdMode.FIXED`` (default): one fixed log-standard-deviation vector.
* ``StdMode.GLOBAL``: one learned vector, shared by all states and motions.
* ``StdMode.PER_MOTION``: one learned vector per motion, for motion-tracking tasks.

Learned values use the existing PPO likelihood and entropy losses. After each
actor update they are projected into ``[min_logstd, max_logstd]``; the defaults
correspond to standard deviations of 0.001 and 0.5. Fixed values are not clamped.

For per-motion exploration, training setup counts the selected motion shards and
creates the same dense table on every rank. It supplies ``global_motion_id`` at
rollout time and stores that ID with the sample. Do not add it to experiment
observation bindings or the mean network's inputs. Repeated IDs accumulate
gradients into the same row; normal DDP averages gradients across ranks. Unused
rows have zero current gradient, but optimizer momentum can still update a row
visited in an earlier batch. Identical shard paths share rows; duplicate clips
in different shards do not.

Full checkpoints include the table, optimizer state and motion index. Resume
requires the same shard paths, order and motion names. A weights-only warm start
from fixed/global std broadcasts the checkpoint vector across the new table;
optimizer and curriculum state start fresh. ``motion_logstd.pt`` is an inspection
sidecar overwritten on each save, not a resume source. Preserve checkpoints or
sidecar snapshots to compare individual motions over time.

Deterministic inference and PPO export use ``forward_deployable``. They do not require
motion IDs or access to the training shards. Regenerate resolved configs when
changing exploration mode; resume uses the saved config rather than rebuilding it.

Next Steps
----------

* :doc:`developer_tips` - Quality of life tips
* See ``examples/experiments/`` for more examples
