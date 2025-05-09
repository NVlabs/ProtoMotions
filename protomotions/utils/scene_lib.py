import logging
import random
import copy
from dataclasses import dataclass, field, MISSING
from typing import List, Optional, Tuple, Dict
from isaac_utils import torch_utils
import torch
import trimesh
import os

from protomotions.envs.base_env.env_utils.object_utils import (
    as_mesh,
    compute_bounding_box,
)


from protomotions.envs.base_env.env_utils.terrains.terrain import Terrain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ObjectMotion:
    """
    Contains motion data for an object.
    Frames is a list of dictionaries. Each frame is expected to have keys:
      'translation': tuple of 3 floats,
      'rotation': tuple of 4 floats.
    fps: Frames per second for the motion (default is 30.0).
    """
    frames: List[dict] = field(default_factory=list)
    fps: float = None


@dataclass
class ObjectOptions:
    """
    Contains options for configuring object properties in the simulator.
    """
    fix_base_link: bool = field(default=MISSING)
    vhacd_enabled: bool = None
    vhacd_params: Dict = field(default_factory=lambda: {
        "resolution": None,
        "max_convex_hulls": None,
        "max_num_vertices_per_ch": None,
    })
    density: float = None
    angular_damping: float = None
    linear_damping: float = None
    max_angular_velocity: float = None
    default_dof_drive_mode: str = None
    override_com: bool = None
    override_inertia: bool = None

    def to_dict(self) -> Dict:
        """Convert options to a dictionary, excluding None values."""
        options_dict = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                options_dict[field_name] = field_value
        return options_dict


@dataclass
class SceneObject:
    """
    Represents an object inside a scene.
    Defaults to translation (0,0,0) and rotation (0,0,0,1) if not provided.
    """
    object_path: str = field(default=MISSING)
    translation: Tuple[float, float, float] = field(default=MISSING)
    rotation: Tuple[float, float, float, float] = field(default=MISSING)
    motion: Optional[ObjectMotion] = None
    options: ObjectOptions = field(default_factory=ObjectOptions)


@dataclass
class Scene:
    """
    Represents a scene consisting of one or more SceneObjects.
    An offset (x, y) indicates the scene's location.
    """
    objects: List[SceneObject] = field(default_factory=list)
    offset: Tuple[float, float] = (0.0, 0.0)

    def add_object(self, scene_object: SceneObject):
        """Add an object to the scene."""
        self.objects.append(scene_object)


@dataclass
class ObjectState:
    """
    Represents the state of an object, including translations and rotations as torch tensors.
    """
    translations: torch.Tensor
    rotations: torch.Tensor


@dataclass
class SpawnInfo:
    """
    Contains information about how to spawn an object in the scene.
    """
    id: int
    object_path: str
    object_options: ObjectOptions
    object_dims: Tuple[float, float, float, float, float, float] = None
    is_first_instance: bool = True
    first_instance_id: int = None


class SceneLib:
    """
    A simplified scene library.

    - The user instantiates SceneLib with a config dictionary (which contains num_envs and scene offset parameters) and a device string.
    - Scenes are provided as a list of Scene dataclasses via create_scenes(). If fewer scenes than num_envs are provided,
      scenes are replicated (either sequentially or randomly).
    - Object motions (a list of frames per SceneObject) are combined into unified tensors stored within the SceneLib instance.
    - The get_object_pose method interpolates the pose of an object at a specified time using delta time (dt).

    Note: SceneObjects no longer have explicit IDs; their order defines their unique index.
    """
    def __init__(self, num_envs: int,device: str = "cpu"):
        """
        Args:
            config: Dictionary containing keys:
                - num_envs: int, number of environments.
            device: Device identifier for torch tensors.
        """
        self.device = device
        self.num_envs = num_envs
        self.scenes: List[Scene] = []
        self.num_objects_per_scene = None
        self._total_spawned_scenes = 0
        self._scene_offsets = []
        self._object_spawn_list = []
        self._object_path_to_id = {}

        # Placeholders for aggregated motion data
        self.object_translations = None
        self.object_rotations = None
        self.motion_lengths = None  # In seconds, per object
        self.motion_starts = None   # Starting index in the unified tensor for each object
        self.motion_dts = None      # Delta time for each object's motion
        self.motion_num_frames = None  # Number of frames in each object's motion

    def create_scenes(self, scenes: List[Scene], terrain: Terrain, assign_method: str = "sequential", replicate_method: str = "random") -> List[Scene]:
        """
        Create and assign scenes to environments using configurations from the Terrain object.

        If fewer scenes are provided than num_envs, replicate scenes using the given replicate_method.

        Args:
            scenes: List of Scene instances.
            terrain: Terrain object providing configuration parameters.
            assign_method: "sequential" or "random" to order the provided scenes.
            replicate_method: "sequential" or "random" to replicate scenes if needed.

        Returns:
            A list of Scene instances (one per environment) with assigned offsets.
        """
        if not scenes:
            logger.error("No scenes provided.")
            raise ValueError("No scenes provided.")

        # Clear previous scene data
        self._scene_offsets = []
        self._object_spawn_list = []
        self._object_path_to_id = {}

        num_scenes = len(scenes)
        # Order scenes according to assign_method
        if assign_method == "sequential":
            assigned_scenes = list(scenes)
        elif assign_method == "random":
            assigned_scenes = list(scenes)
            random.shuffle(assigned_scenes)
        else:
            logger.error("Unknown assign method: %s", assign_method)
            raise ValueError("Assign method must be either 'sequential' or 'random'.")

        # Replicate scenes if needed
        if num_scenes < self.num_envs:
            if replicate_method == "sequential":
                for i in range(self.num_envs - num_scenes):
                    scene = copy.deepcopy(assigned_scenes[i % num_scenes])
                    assigned_scenes.append(scene)
            elif replicate_method == "random":
                for _ in range(self.num_envs - num_scenes):
                    scene = copy.deepcopy(random.choice(assigned_scenes))
                    assigned_scenes.append(scene)
            else:
                logger.error("Unknown replicate method: %s", replicate_method)
                raise ValueError("Replicate method must be either 'sequential' or 'random'.")

        # Use Terrain configurations to compute scene offsets and validate spawn locations
        # Compute offsets using terrain properties
        for idx, scene in enumerate(assigned_scenes):
            x_offset = ((idx % terrain.num_scenes_per_column + 1) * terrain.spacing_between_scenes + terrain.border * terrain.horizontal_scale)
            y_offset = ((idx // terrain.num_scenes_per_column + 1) * terrain.spacing_between_scenes + terrain.scene_y_offset)
            scene.offset = (x_offset, y_offset)
            
            # Compute integer grid location
            scene_x = int(x_offset / terrain.horizontal_scale)
            scene_y = int(y_offset / terrain.horizontal_scale)
            locations = torch.tensor([[scene_x, scene_y]], device=terrain.device)
            assert terrain.is_valid_spawn_location(locations).cpu().item(), f"Scene {idx} is not a valid spawn location."
            terrain.mark_scene_location(scene_x, scene_y)
            logger.info("Assigned scene id %s to offset (%.2f, %.2f)", idx, x_offset, y_offset)
            self._scene_offsets.append((x_offset, y_offset))

            # Ensure each scene has the same number of objects
            object_counts = [len(scene.objects) for scene in assigned_scenes]
            if len(set(object_counts)) != 1:
                logger.error("All scenes must have the same number of objects. Found counts: %s", object_counts)
                raise ValueError("Scenes have inconsistent number of objects: " + str(object_counts))
            self.num_objects_per_scene = object_counts[0]

        # Create object spawn list from the first scene (canonical objects)
        for idx, obj in enumerate(assigned_scenes[0].objects):
            min_x, max_x, min_y, max_y, min_z, max_z = self.compute_object_dims(obj.object_path)
            
            spawn_info = SpawnInfo(
                id=idx,  # First instance gets its index as ID
                object_path=obj.object_path,
                object_options=obj.options,
                object_dims=(min_x, max_x, min_y, max_y, min_z, max_z),
                is_first_instance=True,
                first_instance_id=idx
            )
            self._object_spawn_list.append(spawn_info)
            self._object_path_to_id[obj.object_path] = idx

        # Track first instances for replicated scenes
        first_instances = {}  # Map object paths to their first instance IDs
        for idx, obj in enumerate(assigned_scenes[0].objects):
            first_instances[obj.object_path] = idx

        # Process remaining scenes to mark non-first instances
        for scene in assigned_scenes[1:]:
            for obj in scene.objects:
                if obj.object_path in first_instances:
                    first_instance_id = first_instances[obj.object_path]
                    spawn_info = SpawnInfo(
                        id=first_instance_id,  # Use the ID of the first instance
                        object_path=obj.object_path,
                        object_options=obj.options,
                        object_dims=self._object_spawn_list[first_instance_id].object_dims,
                        is_first_instance=False,
                        first_instance_id=first_instance_id
                    )
                    self._object_spawn_list.append(spawn_info)

        self._total_spawned_scenes = len(assigned_scenes)
        self.scenes = assigned_scenes
        # Automatically combine object motions so the user does not have to call it
        self.combine_object_motions()
        return assigned_scenes
    
    def compute_object_dims(self, object_path):
        obj_path = object_path.replace(".urdf", ".obj").replace(".usda", ".obj").replace(".usd", ".obj")
        stl_path = object_path.replace(".urdf", ".stl").replace(".usda", ".stl").replace(".usd", ".stl")
        ply_path = object_path.replace(".urdf", ".ply").replace(".usda", ".ply").replace(".usd", ".ply")

        if (
            os.path.exists(obj_path)
            or os.path.exists(stl_path)
            or os.path.exists(ply_path)
        ):
            if os.path.exists(obj_path):
                mesh_path = obj_path
            elif os.path.exists(stl_path):
                mesh_path = stl_path
            else:
                mesh_path = ply_path
            mesh = as_mesh(trimesh.load_mesh(mesh_path))
            w_x, w_y, w_z, m_x, m_y, m_z = compute_bounding_box(mesh)
        elif object_path.endswith(".urdf"):
            import xml.etree.ElementTree as ET

            tree = ET.parse(object_path)
            root = tree.getroot()
            link = root.find("link")
            has_size = False
            if link is not None:
                collision = link.find("collision")
                if collision is not None:
                    geometry = collision.find("geometry")
                    if geometry is not None:
                        box = geometry.find("box")
                        if box is not None:
                            size = box.get("size")

                            w_x, w_y, w_z = map(float, size.split())
                            m_x = -w_x / 2
                            m_y = -w_y / 2
                            m_z = -w_z / 2
                            has_size = True
            assert (
                has_size
            ), f"URDF {object_path} must provide size parameters."
        else:
            raise FileNotFoundError(
                f"Object file not found: {obj_path}, {stl_path}, or valid URDF"
            )

        min_x = m_x
        max_x = min_x + w_x
        min_y = m_y
        max_y = min_y + w_y
        min_z = m_z
        max_z = min_z + w_z

        return min_x, max_x, min_y, max_y, min_z, max_z

    def combine_object_motions(self):
        """
        Combine motion data from all SceneObjects in all scenes into unified tensors.

        For each SceneObject:
          - If motion frames are provided, each frame (a dict with 'translation' and 'rotation') is used.
          - Otherwise, a default static frame (using the object's translation and rotation) is added.

        The following tensors are created and stored in SceneLib:
            - self.object_translations: (total_motion_length, 3)
            - self.object_rotations: (total_motion_length, 4)
            - self.motion_lengths: (num_objects,) tensor of motion lengths in seconds
            - self.motion_starts: (num_objects,) tensor of starting indices for each object
            - self.motion_dts: (num_objects,) tensor with delta time per object
            - self.motion_num_frames: (num_objects,) tensor with the number of frames per object
        """
        translations_list = []
        rotations_list = []
        motion_lengths_list = []
        motion_dts_list = []
        motion_num_frames_list = []
        motion_starts = {}  # Map object index -> starting index
        current_start = 0

        all_objects = []
        for scene in self.scenes:
            for obj in scene.objects:
                all_objects.append(obj)

        for idx, obj in enumerate(all_objects):
            motion_starts[idx] = current_start
            if obj.motion and obj.motion.frames:
                frames = obj.motion.frames
                try:
                    fps = obj.motion.fps
                    dt = 1.0 / fps
                except Exception as e:
                    logger.warning("Failed to read fps for object at index %d. Defaulting to 30 fps.", idx)
                    fps = 30.0
                    dt = 1.0 / fps

                num_frames = len(frames)
                motion_length = num_frames * dt
                motion_lengths_list.append(motion_length)
                motion_dts_list.append(dt)
                motion_num_frames_list.append(num_frames)

                for frame in frames:
                    t = frame.get("translation", obj.translation)
                    r = frame.get("rotation", obj.rotation)
                    translations_list.append(torch.tensor(t, dtype=torch.float, device=self.device))
                    rotations_list.append(torch.tensor(r, dtype=torch.float, device=self.device))
                current_start += num_frames
            else:
                dt = 1.0 / 30.0
                motion_lengths_list.append(dt)
                motion_dts_list.append(dt)
                motion_num_frames_list.append(1)
                t = obj.translation
                r = obj.rotation
                translations_list.append(torch.tensor(t, dtype=torch.float, device=self.device))
                rotations_list.append(torch.tensor(r, dtype=torch.float, device=self.device))
                current_start += 1

        total_motion_length = current_start
        if translations_list:
            self.object_translations = torch.stack(translations_list)
            self.object_rotations = torch.stack(rotations_list)
        else:
            self.object_translations = torch.empty((0, 3), device=self.device)
            self.object_rotations = torch.empty((0, 4), device=self.device)
        self.motion_lengths = torch.tensor(motion_lengths_list, dtype=torch.float, device=self.device)
        num_objects = len(all_objects)
        self.motion_starts = torch.tensor([motion_starts[i] for i in range(num_objects)], dtype=torch.long, device=self.device)
        self.motion_dts = torch.tensor(motion_dts_list, dtype=torch.float, device=self.device)
        self.motion_num_frames = torch.tensor(motion_num_frames_list, dtype=torch.long, device=self.device)

        logger.info("Combined motion data for %d objects with total frames: %d", num_objects, total_motion_length)

    def _calc_frame_blend(
        self,
        time: torch.Tensor,
        length: torch.Tensor,
        num_frames: torch.Tensor,
        dt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate frame indices and blend factor for interpolation.

        Args:
            time (torch.Tensor): Current time.
            length (torch.Tensor): Length of the motion sequence in seconds.
            num_frames (torch.Tensor): Number of frames in the motion sequence.
            dt (torch.Tensor): Time step between frames.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Frame index 0, frame index 1, and blend factor.
        """
        phase = time / length
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def get_object_pose(self, object_indices: torch.Tensor, time: torch.Tensor) -> ObjectState:
        """
        Get the interpolated poses for multiple objects at given times.

        Args:
            object_indices (torch.Tensor): 1D tensor of object indices with length equal to num_envs.
            time (torch.Tensor): 1D tensor of times at which to interpolate poses, length equal to num_envs.

        Returns:
            ObjectState: A batched ObjectState with translations (num_envs, 3) and rotations (num_envs, 4).
        """
        if self.motion_starts is None:
            logger.error("Motion data not combined. Call combine_object_motions() first.")
            raise ValueError("Motion data not combined.")

        dt = self.motion_dts[object_indices]
        length = self.motion_lengths[object_indices]
        num_frames = self.motion_num_frames[object_indices]
        start_idx = self.motion_starts[object_indices]

        t_tensor = time.to(dtype=torch.float, device=self.device)
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(t_tensor, length, num_frames, dt)
        idx0 = start_idx + frame_idx0
        idx1 = start_idx + frame_idx1

        translation0 = self.object_translations[idx0]
        translation1 = self.object_translations[idx1]
        translation = (1 - blend.unsqueeze(-1)) * translation0 + blend.unsqueeze(-1) * translation1

        rotation0 = self.object_rotations[idx0]
        rotation1 = self.object_rotations[idx1]
        rotations_list = []
        for i in range(blend.shape[0]):
            # For each element, compute slerp after unsqueezing to add batch dimension
            rot = torch_utils.slerp(rotation0[i].unsqueeze(0), rotation1[i].unsqueeze(0), blend[i].unsqueeze(0).unsqueeze(-1))
            rotations_list.append(rot.squeeze(0))
        rotation = torch.stack(rotations_list, dim=0)

        return ObjectState(translations=translation, rotations=rotation)

    @property
    def total_spawned_scenes(self) -> int:
        """Returns the total number of scenes that were spawned."""
        return self._total_spawned_scenes

    @property
    def scene_offsets(self) -> List[Tuple[float, float]]:
        """Returns the list of scene offsets."""
        return self._scene_offsets

    @property
    def object_spawn_list(self) -> List[SpawnInfo]:
        """Returns the list of canonical objects with their properties."""
        return self._object_spawn_list

    @property
    def object_path_to_id(self) -> Dict[str, int]:
        """Returns the mapping from object paths to their spawn list indices."""
        return self._object_path_to_id


# ----------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    import torch
    
    # Define a dummy Terrain for example usage
    class DummyTerrain:
        def __init__(self):
            self.num_scenes_per_column = 2
            self.spacing_between_scenes = 5.0
            self.border = 2.0
            self.horizontal_scale = 1.0
            self.scene_y_offset = 0.0
            self.device = "cpu"

        def is_valid_spawn_location(self, locations):
            return torch.tensor(True)

        def mark_scene_location(self, x, y):
            pass

    scene_lib = SceneLib(num_envs=4, device="cpu")

    # Create SceneObjects with options
    obj1 = SceneObject(
        object_path="cup.urdf",
        translation=(1.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        motion=ObjectMotion(
            frames=[
                {"translation": (1.0, 0.0, 0.0), "rotation": (0.0, 0.0, 0.0, 1.0)},
                {"translation": (1.5, 0.0, 0.0), "rotation": (0.0, 0.0, 0.0, 1.0)}
            ],
            fps=30.0
        ),
        options=ObjectOptions(
            vhacd_enabled=True,
            vhacd_params={
                "resolution": 50000,
                "max_convex_hulls": 128,
                "max_num_vertices_per_ch": 64
            },
            fix_base_link=True
        )
    )

    obj2 = SceneObject(
        object_path="obstacle.urdf",
        options=ObjectOptions(
            vhacd_enabled=True,
            vhacd_params={"resolution": 50000},
            fix_base_link=True
        )
    )
    scene1 = Scene(id=1, objects=[obj1, obj2])

    obj3 = SceneObject(
        object_path="chair.urdf",
        translation=(2.0, 2.0, 0.0),
        options=ObjectOptions(
            vhacd_enabled=True,
            vhacd_params={"resolution": 50000},
            fix_base_link=True
        )
    )
    obj4 = SceneObject(
        object_path="table.urdf",
        translation=(2.5, 2.0, 0.0),
        options=ObjectOptions(
            vhacd_enabled=True,
            vhacd_params={"resolution": 50000},
            fix_base_link=True
        )
    )
    scene2 = Scene(id=2, objects=[obj3, obj4])

    scenes = [scene1, scene2]

    terrain = DummyTerrain()
    assigned_scenes = scene_lib.create_scenes(scenes, terrain, replicate_method="random")
    for idx, scene in enumerate(assigned_scenes):
        logger.info("Environment %d assigned Scene with objects %s with offset %s", idx, scene.objects, scene.offset)

    # get_object_pose now returns an ObjectState and combine_object_motions is automatically called in create_scenes()
    time = 1. / 30 * 0.5
    pose_obj0 = scene_lib.get_object_pose(object_indices=torch.tensor([0]), time=torch.tensor([time]))
    logger.info("Pose for object at index 0 at time %s:\nTranslations: %s\nRotations: %s", time, pose_obj0.translations, pose_obj0.rotations)
