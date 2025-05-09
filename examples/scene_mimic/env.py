from protomotions.envs.mimic.env import Mimic
from protomotions.utils.scene_lib import (
    Scene,
    SceneObject,
    ObjectOptions,
    SceneLib,
)
import torch


class SceneMimic(Mimic):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config, device, *args, **kwargs)

    def get_motion_requires_scene(self, motion_ids):
        # For this example, we assume all motions require a scene
        return torch.ones(motion_ids.shape[0], device=self.device, dtype=torch.bool)

    def create_terrain_and_scene_lib(self):
        super().create_terrain_and_scene_lib()

        chair_options = ObjectOptions(
            density=1000,
            fix_base_link=True,
            angular_damping=0.01,
            linear_damping=0.01,
            max_angular_velocity=100.0,
            default_dof_drive_mode="DOF_MODE_NONE",
            vhacd_enabled=True,
            override_com=True,
            override_inertia=True,
            vhacd_params={
                "max_convex_hulls": 10,
                "max_num_vertices_per_ch": 64,
                "resolution": 300000,
            },
        )
        # Create scene with chair
        chair = SceneObject(
            object_path="examples/data/armchair.usda",
            options=chair_options,
            translation=(0.0, 0.9, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )
        scene = Scene(objects=[chair])

        # Create SceneLib instance
        self.scene_lib = SceneLib(num_envs=self.num_envs, device=self.device)

        # Create scenes
        self.scene_lib.create_scenes([scene], self.terrain)
