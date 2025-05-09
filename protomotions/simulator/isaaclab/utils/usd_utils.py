import trimesh
import isaaclab.sim as sim_utils
from isaaclab.terrains.utils import create_prim_from_mesh
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from protomotions.envs.base_interface.isaaclab_utils.robots import (
        TrimeshTerrainImporterCfg,
    )
else:
    TrimeshTerrainImporterCfg = object


class TrimeshTerrainImporter:
    def __init__(self, cfg: TrimeshTerrainImporterCfg):
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # generate the terrain
        # Create the mesh here, after the config has been validated
        mesh = trimesh.Trimesh(vertices=cfg.terrain_vertices, faces=cfg.terrain_faces)
        self.import_mesh("terrain", mesh)

    def import_mesh(self, key: str, mesh: trimesh.Trimesh):
        """Import a mesh into the simulator.

        The mesh is imported into the simulator under the prim path ``cfg.prim_path/{key}``. The created path
        contains the mesh as a :class:`pxr.UsdGeom` instance along with visual or physics material prims.

        Args:
            key: The key to store the mesh.
            mesh: The mesh to import.

        Raises:
            ValueError: If a terrain with the same key already exists.
        """

        # get the mesh
        mesh_prim_path = self.cfg.prim_path + f"/{key}"
        # import the mesh
        create_prim_from_mesh(
            mesh_prim_path,
            mesh,
            visual_material=self.cfg.visual_material,
            physics_material=self.cfg.physics_material,
        )
