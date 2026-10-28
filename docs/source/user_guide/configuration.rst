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

   def apply_inference_overrides(
       robot_cfg, simulator_cfg, env_cfg, agent_cfg,
       terrain_cfg, motion_lib_cfg, scene_lib_cfg, args,
   ):
       """Modify copies before saving the inference config bundle."""
       simulator_cfg.domain_randomization = None

This shows the interface, not a complete runnable experiment. See
``examples/experiments/format.py`` for the hook signatures. Scene and motion
factories return configs even when their file fields are ``None``.

Config Building Flow
~~~~~~~~~~~~~~~~~~~~

When you run ``train_agent.py``, configs are built in this order:

1. Load robot config from factory (``--robot-name``)
2. Load simulator config (``--simulator``)
3. Call ``configure_robot_and_simulator()`` for customization
4. Build ``terrain_config()``, ``scene_lib_config()``, ``motion_lib_config()``
5. Build ``env_config()``
6. Build ``agent_config()``
7. Apply CLI overrides (``--overrides``)
8. Save the training configs to ``resolved_configs.json``

There is one saved bundle. ``inference_agent.py`` loads it and derives its
inference view at load: framework defaults, then ``--overrides``, then the
experiment's ``apply_inference_overrides()`` hook. It does not automatically
re-import the training experiment. For an additional runtime hook, pass
``--inference-config path/to/file.py`` with the same
``apply_inference_overrides()`` signature. Runtime ``--overrides`` apply after
that hook.

A value is taken from a fallback only when it is explicitly blank; a missing
file or an unknown name is refused by name, never silently defaulted. Silently
substituting a default for a missing artifact or a mistyped name has repeatedly
turned a typo or an absent file into a run that looked correct but used the
wrong configuration. So an empty override value or an unset optional hook takes
its documented default, while a ``--inference-config`` path that does not exist,
or an ``--overrides`` key naming a field the configs do not define, stops the
run and names the file or the field instead of continuing on a substituted
default.

Evaluation rules belong in ``agent_cfg.evaluator``: ``metric_components`` name
measurements and ``failure_components`` name threshold rules over per-trial
reductions (any frame, sum, mean, min, max or last).
They are independent of training episode terminations. See :doc:`evaluation`
for examples, saved-config migration and checkpoint-score behavior.

Robot Configurations
--------------------

Robot configs in ``protomotions/robot_configs/`` define the robot:

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
           asset_file_name="mjcf/g1_bm_no_mesh_box_feet.xml",
       ))
       
       # PD control parameters per joint (regex patterns)
       control: ControlConfig = field(default_factory=lambda: ControlConfig(
           override_control_info={
               ".*_hip_(pitch|yaw)_joint": ControlInfo(
                   stiffness=40.0, damping=8.0, effort_limit=88,
               ),
               ".*_knee_joint": ControlInfo(
                   stiffness=99.0, damping=19.8, effort_limit=139,
               ),
           }
       ))
       
       # Per-simulator physics settings
       simulation_params: SimulatorParams = field(default_factory=lambda: SimulatorParams(
           isaacgym=IsaacGymSimParams(fps=100, decimation=2),
           newton=NewtonSimParams(fps=200, decimation=4),
       ))

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
       --experiment-path examples/experiments/mimic/mlp.py \
       --experiment-name my_experiment \
       --motion-file path/to/motions.pt \
       --num-envs 4096 \
       --batch-size 16384

CLI Overrides
~~~~~~~~~~~~~

Override nested config values with ``--overrides``:

.. code-block:: bash

   --overrides "agent.num_mini_epochs=4" "env.max_episode_length=500"

   # Override reward weights
   --overrides "env.reward_components.contact_match_rew.static_params.weight=0.0"

   # Disable simulator domain randomization
   --overrides "simulator.domain_randomization=None"

Paths start with ``robot``, ``simulator``, ``terrain``, ``scene_lib``,
``motion_lib``, ``env``, or ``agent``. Nested dictionaries use the same dot
notation as dataclass fields. Values use Python literals (``True``, ``False``,
``None``). Overrides change existing fields; define new components in Python.

Saved Configurations
--------------------

All configurations are saved for reproducibility:

.. code-block:: text

   results/<experiment_name>/
   ├── config.yaml              # CLI arguments + wandb_id
   ├── resolved_configs.json    # The config bundle (tagged JSON) - source of truth
   └── experiment_config.py     # Copy of experiment file

**resolved_configs.json** is the source of truth. It is a tagged JSON tree with
every value resolved -- each dataclass under its class name, enums, tensors and
module-level functions tagged, a component as its kernel name plus bindings and
parameters -- and a ``format_version`` header. Loading rebuilds every object
through its real constructor, so ``__post_init__`` runs and validation happens by
construction; a stale or hand-edited bundle fails naming the field, the class and
the file with the regenerate command. There is no pickle and no YAML sidecar: the
bundle is plain text you can read and ``git diff``.

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
       --experiment-path examples/experiments/mimic/mlp.py \
       --experiment-name migrated_experiment \
       --motion-file /path/to/motion.pt \
       --num-envs 4096 --batch-size 16384 \
       --create-config-only

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

   from protomotions.envs.context_views import EnvContext
   from protomotions.envs.mdp_component import MdpComponent
   from protomotions.envs.rewards.regularization import compute_action_smoothness
   from protomotions.envs.rewards.tracking import compute_gt_rew

   reward_components = {
       "gt_rew": MdpComponent(
           compute_func=compute_gt_rew,                      # Pure tensor function
           dynamic_vars={                                  # Map params to context paths
               "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
               "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
           },
           static_params={"weight": 0.5, "coefficient": -100.0},  # Static parameters
       ),
       "action_smoothness": MdpComponent(
           compute_func=compute_action_smoothness,
           dynamic_vars={
               "current_processed_action": EnvContext.current_processed_action,
               "previous_processed_action": EnvContext.previous_processed_action,
           },
           static_params={"weight": -0.02},
       ),
   }

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
