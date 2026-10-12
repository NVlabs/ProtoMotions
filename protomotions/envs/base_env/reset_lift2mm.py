"""Diagnostic variant: lift each reference-pose robot reset by exactly 2 mm."""
from protomotions.envs.base_env.env import BaseEnv


class ResetLift2mmEnv(BaseEnv):
    def compute_ref_reset_state(self, env_ids, motion_ids, motion_times, sample_flat=False):
        states, objects = super().compute_ref_reset_state(
            env_ids, motion_ids, motion_times, sample_flat
        )
        # Translate the robot only after the usual reference pose and terrain
        # placement have been computed. Preserve XY, orientation and velocities.
        # Do not change the reference motion or respawn_root_offset.
        states.root_pos[:, 2] += 0.002
        return states, objects
