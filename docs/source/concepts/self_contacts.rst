Self-contact cost
=================

Use this cost to discourage unwanted contact between robot parts.
Choose which bodies to penalize and which other bodies they should not touch.

The BM tracker in ``examples/experiments/mimic/mlp_bm_l2c2.py`` scores
non-foot/non-hand bodies against all robot links. For G1:

.. code-block:: python

   from protomotions.envs.mdp_component import MdpComponent
   from protomotions.envs.rewards.meta import RewardMeta
   from protomotions.envs.rewards.regularization import compute_self_contacts

   allowed = {
       "left_ankle_roll_link", "right_ankle_roll_link",
       "left_wrist_yaw_link", "right_wrist_yaw_link",
       "left_rubber_hand", "right_rubber_hand",  # Noncolliding hand markers.
   }
   names = robot_cfg.kinematic_info.body_names
   bodies = [name for name in names if name not in allowed]
   partners = [f"robot/{name}" for name in names]

   # In configure_robot_and_simulator():
   robot_cfg.update_fields(
       contact_bodies=bodies, contact_partners=partners,
   )

   # In env_config(), after constructing cfg:
   cfg.reward_components["self_contact"] = MdpComponent.from_kernel(
       compute_self_contacts,
       meta=RewardMeta(weight=-0.1),
       # The build refuses a selection the simulator does not measure.
       required_keys={"contact_body_index": bodies, "contact_partner_index": partners},
       body_names=bodies,
       partner_names=partners,
       force_threshold=1.0,  # Normal force in newtons.
   )

The simulator measures these contacts; the reward selects which ones to use.
A missing requested measurement raises an error at setup. Ground and object
contacts do not contribute. Each selected body adds at most one penalty per
step, even if several other links touch it. A hand touching the torso penalizes
the torso; two knees touching penalizes both knees. Hands and feet are exempt
from scoring, not removed as partners. G1 measures 27 rows and 33 columns.

This follows the scored-body selection in `BeyondMimic`_ and `Holosoma`_.
`SONIC`_ additionally exempts elbows; this BM example does not. Those upstream
terms use net-contact history, whereas this cost intentionally uses only current
robot-to-robot pairs, not terrain or objects.

Train the example with:

.. code-block:: bash

   python protomotions/train_agent.py --robot-name g1 --simulator isaaclab \
       --experiment-path examples/experiments/mimic/mlp_bm_l2c2.py \
       --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
       --num-envs 4096 --batch-size 4096 --experiment-name g1_bm_self_contact

The robot asset must allow self-collisions; the reward does not enable them.
Links without collision geometry cannot trigger the cost, and PhysX filters
contacts between directly connected parent/child links.
For simulator support, see :doc:`contact_matrix`.

.. _BeyondMimic: https://github.com/HybridRobotics/whole_body_tracking/blob/9db915838b17785c6fec90c91fe80b95b147e1bd/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py#L220-L232
.. _Holosoma: https://github.com/amazon-far/holosoma/blob/06945a06fecc9791cacc204db8d988c5b516c7c4/src/holosoma/holosoma/config_values/wbt/g1/reward.py#L50-L60
.. _SONIC: https://github.com/NVlabs/GR00T-WholeBodyControl/blob/45c03facbf848e0c743091b7d9c17952096ff908/gear_sonic/config/manager_env/rewards/terms/undesired_contacts.yaml
