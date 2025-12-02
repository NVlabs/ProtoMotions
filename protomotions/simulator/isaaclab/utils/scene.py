# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Optional
from protomotions.components.terrains.terrain import Terrain
from protomotions.robot_configs.base import RobotConfig
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from protomotions.simulator.isaaclab.utils.usd_utils import TrimeshTerrainImporter
from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig
from protomotions.robot_configs.base import ControlType


@configclass
class TrimeshTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = TrimeshTerrainImporter

    terrain_type: str = "trimesh"
    terrain_vertices: list = None
    terrain_faces: list = None


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    def __init__(
        self,
        config: IsaacLabSimulatorConfig,
        robot_config: RobotConfig,
        terrain: Optional[Terrain] = None,
        scene_cfgs=None,
        pretty=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        activate_contact_sensors = robot_config.contact_bodies is not None

        # lights
        if True:  # pretty:
            # This is way prettier, but also slower to render
            self.light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                ),
            )
        else:
            self.light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(
                    intensity=3000.0, color=(0.75, 0.75, 0.75)
                ),
            )

        num_objects_per_scene = 0
        if scene_cfgs is not None:
            num_objects_per_scene = len(scene_cfgs)
            for obj_idx, obj_configs in enumerate(scene_cfgs):
                spawn_cfg = sim_utils.MultiAssetSpawnerCfg(
                    activate_contact_sensors=activate_contact_sensors,
                    assets_cfg=obj_configs,
                    random_choice=False,
                )
                # Rigid Object
                object = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Object_{obj_idx}",
                    spawn=spawn_cfg,
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                setattr(self, f"object_{obj_idx}", object)

                # Object contact sensors are used to detect collisions between objects.
                object_contact_paths = ["/World/ground/terrain/mesh"]
                for i in range(num_objects_per_scene):
                    if i != obj_idx:
                        object_contact_paths.append(f"/World/envs/env_.*/Object_{i}")
                if activate_contact_sensors:
                    object_sensor_cfg = ContactSensorCfg(
                        prim_path=f"/World/envs/env_.*/Object_{obj_idx}",
                        # debug_vis=True,
                        filter_prim_paths_expr=object_contact_paths,
                        history_length=config.sim.decimation,
                    )
                    setattr(self, f"object_{obj_idx}_contact_sensor", object_sensor_cfg)

        actuators = {}
        ActuatorConfig = (
            ImplicitActuatorCfg
            if robot_config.control.control_type == ControlType.BUILT_IN_PD
            else IdealPDActuatorCfg
        )
        for dof_name, control_info in robot_config.control.control_info.items():
            stiffness = control_info.stiffness
            damping = control_info.damping
            if robot_config.control.control_type != ControlType.BUILT_IN_PD:
                stiffness = 0.0
                damping = 0.0
            actuators[dof_name] = ActuatorConfig(
                joint_names_expr=[dof_name],
                # Only include non-None values in the kwargs
                **{
                    key: value
                    for key, value in {
                        "stiffness": stiffness,
                        "damping": damping,
                        "armature": control_info.armature,
                        "effort_limit_sim": control_info.effort_limit,
                        "velocity_limit_sim": control_info.velocity_limit,
                        "friction": control_info.friction,
                    }.items()
                    if value is not None
                },
            )

        # articulation
        self.robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{robot_config.asset.asset_root}/{robot_config.asset.usd_asset_file_name}",
                activate_contact_sensors=activate_contact_sensors,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=robot_config.asset.disable_gravity,
                    retain_accelerations=False,
                    linear_damping=robot_config.asset.linear_damping,
                    angular_damping=robot_config.asset.angular_damping,
                    max_linear_velocity=robot_config.asset.max_linear_velocity,
                    max_angular_velocity=robot_config.asset.max_angular_velocity,
                    max_depenetration_velocity=config.sim.physx.max_depenetration_velocity,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=robot_config.asset.self_collisions,
                    solver_position_iteration_count=config.sim.physx.num_position_iterations,
                    solver_velocity_iteration_count=config.sim.physx.num_velocity_iterations,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=config.sim.physx.contact_offset,
                    rest_offset=config.sim.physx.rest_offset,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.9, 0.9), metallic=0.5
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, robot_config.default_root_height),
                joint_pos={".*": 0.0},
                joint_vel={".*": 0.0},
            ),
            actuators=actuators,
        )

        # Apply disable_gravity setting for all robot types if specified
        if (
            hasattr(robot_config.asset, "disable_gravity")
            and robot_config.asset.disable_gravity
        ):
            # Only modify disable_gravity field, keeping all other settings
            new_rigid_props = self.robot.spawn.rigid_props.replace(disable_gravity=True)
            self.robot.spawn = self.robot.spawn.replace(rigid_props=new_rigid_props)

        if activate_contact_sensors:
            sensing_filter = ["/World/ground/terrain/mesh"]
            for obj_idx in range(num_objects_per_scene):
                sensing_filter.append(f"/World/envs/env_.*/Object_{obj_idx}")
            for body_name in robot_config.contact_bodies:
                contact_sensor_cfg = ContactSensorCfg(
                    prim_path=f"{robot_config.asset.usd_bodies_root_prim_path}{body_name}",
                    filter_prim_paths_expr=sensing_filter,
                    history_length=config.sim.decimation,
                )
                setattr(self, f"contact_sensor_{body_name}", contact_sensor_cfg)

        if terrain is not None:
            terrain_physics_material = sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode=terrain.sim_config.combine_mode.value,
                restitution_combine_mode=terrain.sim_config.combine_mode.value,
                static_friction=terrain.sim_config.static_friction,
                dynamic_friction=terrain.sim_config.dynamic_friction,
                restitution=terrain.sim_config.restitution,
            )
            terrain_visual_material = sim_utils.MdlFileCfg(
                mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                project_uvw=True,
            )

            vertices = terrain.vertices
            height_offset = terrain.sim_config.height_offset
            vertices[..., 2] += height_offset

            self.terrain = TrimeshTerrainImporterCfg(
                prim_path="/World/ground",
                # Pass the mesh data instead of the mesh object
                terrain_vertices=vertices.tolist(),
                terrain_faces=terrain.triangles,
                collision_group=-1,
                visual_material=terrain_visual_material,
                physics_material=terrain_physics_material,
            )
        else:
            self.terrain = None
