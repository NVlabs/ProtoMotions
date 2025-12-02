Configuration System
====================

.. note::

   ProtoMotions3 uses **Python-based dataclass configs**, 
   changing from Hydra or YAML inheritance in previous versions.
   This design provides IDE autocomplete, type checking, easier debugging, and keeps
   all config logic in one place (the experiment file).

Design
-----------------

Why Python dataclasses over Hydra/YAML?

* **IDE support**: Full autocomplete and type hints
* **No inheritance complexity**: No manual tracking which YAML overrides what
* **Readable**: Config logic is explicit Python code

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

   def env_config(robot_cfg, args) -> MimicEnvConfig:
       """Build environment configuration with rewards."""
       return MimicEnvConfig(
           max_episode_length=1000,
           reward_config={...},
           ...
       )

   # Optional functions
   def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
       """Customize robot and simulator settings."""
       robot_cfg.update_fields(contact_bodies=[...])

   def agent_config(robot_cfg, env_cfg, args) -> PPOAgentConfig:
       """Build agent/network configuration."""
       return PPOAgentConfig(
           model=PPOModelConfig(...),
           batch_size=args.batch_size,
           ...
       )

   def apply_inference_overrides(robot_cfg, simulator_cfg, env_cfg, agent_cfg, args):
       """Apply evaluation-time overrides (e.g., disable early termination)."""
       env_cfg.mimic_early_termination = None

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
8. Save all to ``resolved_configs.pt``

Robot Configurations
--------------------

Robot configs in ``protomotions/robot_configs/`` define the robot:

.. code-block:: python

   @dataclass
   class G1RobotConfig(RobotConfig):
       # Map common names to robot-specific body names
       common_naming_to_robot_body_names: Dict[str, List[str]] = field(
           default_factory=lambda: {
               "all_left_foot_bodies": ["left_ankle_roll_link"],
               "all_right_foot_bodies": ["right_ankle_roll_link"],
               "head_body_name": ["head"],
               "torso_body_name": ["torso_link"],
           }
       )
       
       # Asset configuration
       asset: RobotAssetConfig = field(default_factory=lambda: RobotAssetConfig(
           asset_file_name="mjcf/g1_bm_no_mesh_box_feet.xml",
           usd_asset_file_name="usd/g1_bm/g1_bm.usda",
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
   --overrides "env.reward_config.contact_match_rew.weight=0.0"

   # Disable domain randomization (NOTE: "True" not "true")
   --overrides "robot.asset.self_collisions=True"

Saved Configurations
--------------------

All configurations are saved for reproducibility:

.. code-block:: text

   results/<experiment_name>/
   ├── config.yaml              # CLI arguments + wandb_id
   ├── resolved_configs.pt      # Full config objects (pickled) - primary
   ├── resolved_configs.yaml    # Human-readable (best-effort)
   ├── experiment_config.py     # Copy of experiment file
   └── resolved_configs_inference.pt  # Configs with eval overrides

**resolved_configs.pt** is the primary source of truth. It uses pickle to handle
complex types (Union, nested dataclasses, torch.Tensor) that YAML cannot represent.

.. warning::

   **Do NOT modify resolved_configs.yaml files.** They are generated for human 
   readability only—the actual source of truth is the ``.pt`` file.

   To modify configurations:

   * **Small changes:** Use ``--overrides`` on the command line
   * **Large changes:** Use ``--create-config-only`` to generate new configs, 
     then copy the newly generated ``.pt`` file to your checkpoint directory

Resume Behavior
~~~~~~~~~~~~~~~

.. warning::

   **Resume uses exact saved configs.** CLI overrides are ignored during resume.
   
   If you need to change configs, start a new experiment with ``--experiment-name``.

Training modes:

1. **Fresh start**: New experiment name → build configs from experiment file
2. **Resume**: Same experiment name with existing checkpoint → load from ``resolved_configs.pt``
3. **Warm start**: ``--checkpoint <path>`` with new experiment name → new configs, old weights

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

For simple code changes which breaks old models,
you can also add to backward compatibility fixes in 
``apply_backward_compatibility_fixes()`` in ``protomotions.utils.inference_utils.py``.

Reward Configuration
--------------------

Rewards are configured as a dictionary of ``RewardComponentConfig``:

.. code-block:: python

   from protomotions.envs.base_env.config import RewardComponentConfig
   from protomotions.envs.utils.rewards import mean_squared_error_exp

   reward_config = {
       "gt_rew": RewardComponentConfig(
           function=mean_squared_error_exp,
           variables={
               "x": "current_state.rigid_body_pos",
               "ref_x": "ref_state.rigid_body_pos",
               "coefficient": "-100.0",
           },
           weight=0.5,
       ),
       "action_smoothness": RewardComponentConfig(
           function=norm,
           variables={"x": "current_actions - previous_actions"},
           weight=-0.02,
       ),
   }

**Variable references** use string expressions evaluated at runtime:

* ``current_state.rigid_body_pos`` - Current robot state
* ``ref_state.rigid_body_pos`` - Reference motion state
* ``current_actions - previous_actions`` - Action difference

This design lets you add new reward terms without modifying core code.


Agent/Model Configuration
-------------------------

Network architectures are composed through configs:

.. code-block:: python

   from protomotions.agents.ppo.config import PPOActorConfig, PPOModelConfig
   from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig

   actor_config = PPOActorConfig(
       num_out=robot_config.kinematic_info.num_dofs,
       actor_logstd=-2.9,
       in_keys=["max_coords_obs", "mimic_target_poses"],
       mu_model=MLPWithConcatConfig(
           in_keys=["max_coords_obs", "mimic_target_poses"],
           normalize_obs=True,
           layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
           output_activation="tanh",
       ),
   )

The ``in_keys``/``out_keys`` system connects observations to network inputs,
and connects different network layers/modules. 
TensorDict is used to handle the data flow, which also make ONNX export easier.

Debugging Tips
--------------

* **Ask AI assistants**: To understand the meaning of config fields,
  you can ask your favorite AI coding assistant.

Next Steps
----------

* :doc:`developer_tips` - Quality of life tips
* See ``examples/experiments/`` for more examples
