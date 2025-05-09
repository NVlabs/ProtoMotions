from protomotions.envs.base_env.env_utils.terrains.flat_terrain import FlatTerrain
from protomotions.simulator.base_simulator.config import RobotConfig
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from protomotions.simulator.isaaclab.utils.usd_utils import (
    TrimeshTerrainImporter,
)
from protomotions.simulator.isaaclab.utils.robots import (
    SMPL_CFG,
    SMPLX_CFG,
    H1_CFG,
)


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
        config,
        robot_config: RobotConfig,
        terrain,
        scene_cfgs=None,
        pretty=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        robot_type = robot_config.asset.robot_type
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

        # articulation
        if robot_type == "smpl_humanoid":
            self.robot: ArticulationCfg = SMPL_CFG.replace(
                prim_path="/World/envs/env_.*/Robot"
            )
            self.contact_sensor: ContactSensorCfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/bodies/.*",
                filter_prim_paths_expr=[f"/World/objects/object_{i}" for i in range(0)],
            )
        elif robot_type == "smplx_humanoid":
            self.robot: ArticulationCfg = SMPLX_CFG.replace(
                prim_path="/World/envs/env_.*/Robot"
            )
            self.contact_sensor: ContactSensorCfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/bodies/.*",
                filter_prim_paths_expr=[f"/World/objects/object_{i}" for i in range(0)],
            )
        elif robot_type in ["h1", "g1"]:
            init_state = ArticulationCfg.InitialStateCfg(
                pos=tuple(robot_config.init_state.pos),
                joint_pos={
                    joint_name: joint_angle for joint_name, joint_angle in
                    robot_config.init_state.default_joint_angles.items()
                },
                joint_vel={".*": 0.0},
            )

            # ImplicitActuatorCfg IdealPDActuatorCfg
            actuators = {
                robot_config.dof_names[i]: IdealPDActuatorCfg(
                    joint_names_expr=[robot_config.dof_names[i]],
                    effort_limit=robot_config.dof_effort_limits[i],
                    velocity_limit=robot_config.dof_vel_limits[i],
                    stiffness=0,
                    damping=0,
                    armature=robot_config.dof_armatures[i],
                    friction=robot_config.dof_joint_frictions[i],
                ) for i in range(len(robot_config.dof_names))
            }

            if robot_type == "h1":
                self.robot: ArticulationCfg = H1_CFG.replace(
                    prim_path="/World/envs/env_.*/Robot", init_state=init_state, actuators=actuators
                )
            elif robot_type == "g1":
                self.robot: ArticulationCfg = G1_CFG.replace(
                    prim_path="/World/envs/env_.*/Robot", init_state=init_state, actuators=actuators
                )
            self.contact_sensor: ContactSensorCfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/.*",
                filter_prim_paths_expr=[f"/World/objects/object_{i}" for i in range(0)],
            )
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")

        if scene_cfgs is not None:
            for obj_idx, obj_configs in enumerate(scene_cfgs):
                spawn_cfg = sim_utils.MultiAssetSpawnerCfg(
                    activate_contact_sensors=True,
                    assets_cfg=obj_configs,
                    random_choice=False,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0
                    ),
                )
                # Rigid Object
                object = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Object_{obj_idx}",
                    spawn=spawn_cfg,
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                setattr(self, f"object_{obj_idx}", object)


        terrain_physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=config.plane.static_friction,
            dynamic_friction=config.plane.dynamic_friction,
            restitution=config.plane.restitution,
        )
        terrain_visual_material = sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        )
        if isinstance(terrain, FlatTerrain):
            # When using a flat terrain, we spawn the built-in plane.
            # This is faster and more memory efficient than spawning a trimesh terrain.
            # The IsaacLab plane spans the entire environment.
            self.terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                visual_material=terrain_visual_material,
                physics_material=terrain_physics_material,
                debug_vis=False,
            )
        else:
            self.terrain = TrimeshTerrainImporterCfg(
                prim_path="/World/ground",
                # Pass the mesh data instead of the mesh object
                terrain_vertices=terrain.vertices,
                terrain_faces=terrain.triangles,
                collision_group=-1,
                visual_material=terrain_visual_material,
                physics_material=terrain_physics_material,
            )
