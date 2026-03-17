# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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
"""MuJoCo CPU-only simulator implementation."""

import atexit
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np
import torch

log = logging.getLogger(__name__)

from protomotions.simulator.base_simulator.config import (
    MarkerState,
    ProjectileConfig,
    SimBodyOrdering,
)
from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    ResetState,
    RobotState,
    RootOnlyState,
    StateConversion,
)
from protomotions.simulator.mujoco.config import MujocoSimulatorConfig


def _to_torch_f32(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to float32 torch tensor."""
    return torch.from_numpy(arr.astype(np.float32))


class MujocoSimulator(Simulator):
    """MuJoCo CPU-only simulator backend.

    Key characteristics:
    - CPU-only execution (device must be torch.device("cpu"))
    - Single environment only (num_envs=1)
    - No scene/object support
    - Flat ground plane from MuJoCo default floor
    - Quaternion format: wxyz (w_last=False)
    """

    config: MujocoSimulatorConfig

    def __init__(
        self,
        config: MujocoSimulatorConfig,
        robot_config,
        terrain,
        device: torch.device,
        scene_lib,
    ) -> None:
        """Initialize MuJoCo simulator shell."""
        assert device.type == "cpu", "MuJoCo simulator only supports CPU device"
        assert config.num_envs == 1, "MuJoCo simulator only supports num_envs=1"
        assert scene_lib.num_scenes() == 0, "MuJoCo simulator does not support scenes"

        super().__init__(
            config=config,
            robot_config=robot_config,
            scene_lib=scene_lib,
            terrain=terrain,
            device=device,
        )

        # MuJoCo-specific state
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.viewer = None
        self._viewer_initialized = False

        # Cached control parameters
        self._kp = None  # [num_dofs] stiffness in common DOF order
        self._kd = None  # [num_dofs] damping in common DOF order
        self._effort_limits = None  # [num_dofs] torque limits in common DOF order
        self._last_applied_torques = None  # Track for _get_simulator_dof_forces
        self._dof_to_actuator = None  # Mapping: sim DOF index -> actuator index

        # Explicit PD mode state (used when use_implicit_pd=False)
        self._kp_sim = None  # PD gains reordered to sim DOF order
        self._kd_sim = None
        self._effort_limits_sim = None
        self._pd_targets_sim = None  # Current PD targets in sim DOF order

        # Body/DOF indexing
        self._num_actuated_dofs = self.robot_config.number_of_actions
        self._has_free_joint = not self.robot_config.asset.fix_base_link

        # Action EMA filter: a_applied = alpha * a_policy + (1 - alpha) * a_prev
        # Set alpha=1.0 to disable filtering, lower values = more smoothing
        self._action_filter_alpha = 1.0
        self._prev_pd_targets = None  # Initialized on first call

        # Debug counter
        self._step_count = 0

    @staticmethod
    def _load_mjcf_stripped(
        asset_path: str,
        projectile_config: Optional[ProjectileConfig] = None,
    ) -> mujoco.MjModel:
        """Load MJCF from file, patching it for standalone MuJoCo use.

        Modifications applied:
        1. Remove <sensor> elements (IMU, etc.) that may reference missing sites
        2. Add scene setup: visual settings, skybox, checkerboard ground, lighting
        3. Add projectile free-joint box bodies (if projectile_config provided)
        """
        tree = ET.parse(asset_path)
        root = tree.getroot()

        # 1. Remove <sensor> elements
        for sensor_elem in root.findall("sensor"):
            root.remove(sensor_elem)
            log.info("  Stripped <sensor> element from MJCF")

        # 2. Add scene setup (visual, textures, ground, lighting)
        MujocoSimulator._add_visual_settings(root)
        MujocoSimulator._add_scene_assets(root)
        MujocoSimulator._add_ground_and_light(root)

        # 3. Add projectile bodies
        if projectile_config is not None:
            MujocoSimulator._add_projectile_bodies(
                root,
                projectile_config.num_projectiles,
                projectile_config.get_sizes(),
                projectile_config.density,
                projectile_config.hide_z,
            )

        # Write cleaned XML to a temp file in the same directory
        # (preserves relative paths for meshdir, includes, etc.)
        asset_dir = os.path.dirname(os.path.abspath(asset_path))
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", dir=asset_dir, delete=False
        ) as tmp:
            tmp.write(ET.tostring(root, encoding="unicode"))
            tmp_path = tmp.name

        try:
            model = mujoco.MjModel.from_xml_path(tmp_path)
        finally:
            os.unlink(tmp_path)

        return model

    @staticmethod
    def _add_projectile_bodies(
        root: ET.Element,
        num_projectiles: int,
        sizes: list,
        density: float,
        hide_z: float,
    ) -> None:
        """Add free-joint box bodies for projectiles to the MJCF worldbody."""
        worldbody = root.find("worldbody")
        if worldbody is None:
            return

        for i in range(num_projectiles):
            s = str(sizes[i])
            body = ET.SubElement(worldbody, "body")
            body.set("name", f"projectile_{i}")
            body.set("pos", f"0 0 {hide_z}")
            joint = ET.SubElement(body, "joint")
            joint.set("type", "free")
            geom = ET.SubElement(body, "geom")
            geom.set("type", "box")
            geom.set("size", f"{s} {s} {s}")
            geom.set("density", str(density))
            geom.set("rgba", "0.8 0.1 0.1 1")

        log.info("  Added %d projectile bodies to MJCF", num_projectiles)

    @staticmethod
    def _add_visual_settings(root: ET.Element) -> None:
        """Add visual settings (headlight, haze, camera angles) if not present."""
        if root.find("visual") is None:
            visual = ET.SubElement(root, "visual")
            headlight = ET.SubElement(visual, "headlight")
            headlight.set("diffuse", "0.6 0.6 0.6")
            headlight.set("ambient", "0.4 0.4 0.4")
            headlight.set("specular", "0.0 0.0 0.0")
            rgba = ET.SubElement(visual, "rgba")
            rgba.set("haze", "0.15 0.25 0.35 1")
            global_elem = ET.SubElement(visual, "global")
            global_elem.set("azimuth", "-130")
            global_elem.set("elevation", "-20")
            print("  Added visual settings to MJCF")

    @staticmethod
    def _add_scene_assets(root: ET.Element) -> None:
        """Add skybox and ground textures/materials if not present."""
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")

        # Skybox (gradient blue-to-black)
        if asset.find("texture[@type='skybox']") is None:
            skybox = ET.SubElement(asset, "texture")
            skybox.set("type", "skybox")
            skybox.set("builtin", "gradient")
            skybox.set("rgb1", "0.3 0.5 0.7")
            skybox.set("rgb2", "0 0 0")
            skybox.set("width", "512")
            skybox.set("height", "3072")

        # Checkerboard ground texture
        if asset.find("texture[@name='groundplane']") is None:
            ground_tex = ET.SubElement(asset, "texture")
            ground_tex.set("type", "2d")
            ground_tex.set("name", "groundplane")
            ground_tex.set("builtin", "checker")
            ground_tex.set("mark", "edge")
            ground_tex.set("rgb1", "0.2 0.3 0.4")
            ground_tex.set("rgb2", "0.1 0.2 0.3")
            ground_tex.set("markrgb", "0.8 0.8 0.8")
            ground_tex.set("width", "300")
            ground_tex.set("height", "300")

        # Ground material
        if asset.find("material[@name='groundplane']") is None:
            ground_mat = ET.SubElement(asset, "material")
            ground_mat.set("name", "groundplane")
            ground_mat.set("texture", "groundplane")
            ground_mat.set("texuniform", "true")
            ground_mat.set("texrepeat", "5 5")
            ground_mat.set("reflectance", "0.2")

        print("  Added scene textures/materials to MJCF")

    @staticmethod
    def _add_ground_and_light(root: ET.Element) -> None:
        """Add ground plane and directional light to worldbody if not present."""
        worldbody = root.find("worldbody")
        if worldbody is None:
            return

        # Check if a ground/floor geom already exists
        has_ground = False
        for geom in worldbody.findall("geom"):
            name = geom.get("name", "").lower()
            geom_type = geom.get("type", "").lower()
            if "floor" in name or "ground" in name or geom_type == "plane":
                has_ground = True
                break

        if not has_ground:
            ground = ET.SubElement(worldbody, "geom")
            ground.set("name", "floor")
            ground.set("type", "plane")
            ground.set("size", "0 0 0.05")
            ground.set("material", "groundplane")
            print("  Added ground plane to MJCF")

        # Check if a light already exists
        has_light = len(worldbody.findall("light")) > 0
        if not has_light:
            light = ET.SubElement(worldbody, "light")
            light.set("pos", "2 0 5.0")
            light.set("dir", "0 0 -1")
            light.set("diffuse", "0.4 0.4 0.4")
            light.set("specular", "0.1 0.1 0.1")
            light.set("directional", "true")
            print("  Added directional light to MJCF")

    def _create_simulation(self) -> None:
        """Create the MuJoCo simulation environment."""
        # Load MJCF model
        asset_root = self.robot_config.asset.asset_root
        asset_file = self.robot_config.asset.asset_file_name
        asset_path = os.path.join(asset_root, asset_file)

        # Pre-create projectile config so we can inject bodies into MJCF
        self._proj_config = ProjectileConfig()

        log.info("Loading MuJoCo model from: %s", asset_path)
        self.model = self._load_mjcf_stripped(asset_path, self._proj_config)
        self.data = mujoco.MjData(self.model)

        # Set timestep
        self.model.opt.timestep = 1.0 / self.config.sim.fps
        print(
            f"MuJoCo timestep: {self.model.opt.timestep:.4f}s ({self.config.sim.fps}Hz)"
        )

        # Zero passive forces from MJCF (we handle PD control via actuators)
        self._zero_passive_forces()

        # Override armature and frictionloss from robot config
        self._override_joint_properties()

        # Build actuator-to-DOF mapping
        self._build_actuator_mapping()

        # Cache PD control parameters
        self._setup_control_parameters()

        # Configure actuators based on PD mode
        # Fall back to implicit PD if config doesn't have use_implicit_pd
        # (can happen when switching simulators via update_simulator_config_for_test)
        use_implicit_pd = getattr(self.config, "use_implicit_pd", True)
        if use_implicit_pd:
            # Convert motor actuators to position actuators (MuJoCo handles PD internally)
            self._configure_actuators_for_pd()
        else:
            # Keep motor actuators -- we compute PD torques explicitly each substep
            print("  PD mode: EXPLICIT (manual PD at each substep, motor actuators)")
            self._configure_explicit_pd()

        # Initialize torque tracking
        self._last_applied_torques = np.zeros(self._num_actuated_dofs, dtype=np.float32)

        # Run initial forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Compute projectile qpos/qvel indices
        # Robot: free joint (7 qpos, 6 qvel) + actuated DOFs
        # Projectile i: free joint at qpos[start + i*7 : start + i*7 + 7]
        self._proj_qpos_start = (
            7 + self._num_actuated_dofs
            if self._has_free_joint
            else self._num_actuated_dofs
        )
        self._proj_qvel_start = (
            6 + self._num_actuated_dofs
            if self._has_free_joint
            else self._num_actuated_dofs
        )

        # Count robot bodies (exclude world body 0 and projectile bodies)
        self._num_robot_bodies = self._num_bodies  # from robot config

        # Initialize viewer if not headless
        if not self.headless:
            self._init_viewer()

        log.info(
            "MuJoCo simulator initialized: "
            "%d bodies, %d qpos, %d qvel, %d actuators, "
            "%d projectiles (qpos_start=%d, qvel_start=%d)",
            self.model.nbody,
            self.model.nq,
            self.model.nv,
            self.model.nu,
            self._proj_config.num_projectiles,
            self._proj_qpos_start,
            self._proj_qvel_start,
        )

    def _zero_passive_forces(self) -> None:
        """Zero out passive stiffness/damping from MJCF.

        We manage PD control ourselves, so passive forces would double-count.
        """
        self.model.jnt_stiffness[:] = 0.0
        self.model.dof_damping[:] = 0.0
        print(
            f"  Zeroed passive forces: "
            f"{self.model.njnt} joints stiffness, "
            f"{self.model.nv} DOFs damping"
        )

    def _override_joint_properties(self) -> None:
        """Override armature and frictionloss from robot config.

        The MJCF has default values (e.g. armature=0.03 for all joints) that
        may differ from the robot config's per-joint values. Newton and IsaacGym
        override these; we must do the same.

        frictionloss (Coulomb joint friction) is zeroed because IsaacGym's PhysX
        does not model it, so policies trained in IsaacGym don't expect it.
        """
        control_info = self.robot_config.control.control_info
        dof_start = 6 if self._has_free_joint else 0

        for i in range(self.model.njnt):
            jnt_type = self.model.jnt_type[i]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                continue

            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            dof_addr = self.model.jnt_dofadr[i]
            dof_idx = dof_addr - dof_start

            if jnt_name in control_info:
                info = control_info[jnt_name]
                old_armature = self.model.dof_armature[dof_addr]

                # Override armature from robot config
                if info.armature is not None:
                    self.model.dof_armature[dof_addr] = info.armature

                # # Zero frictionloss (IsaacGym doesn't model this)
                # self.model.dof_frictionloss[dof_addr] = 0.0

                print(
                    f"  Joint '{jnt_name}' DOF[{dof_idx}]: "
                    f"armature {old_armature:.6f} -> {self.model.dof_armature[dof_addr]:.6f}, "
                    f"frictionloss -> 0.0"
                )

    def _build_actuator_mapping(self) -> None:
        """Build mapping from sim DOF index to MuJoCo actuator index.

        MuJoCo's data.ctrl is indexed by actuator order (from <actuator> section),
        NOT by DOF order. This mapping ensures torques go to the right actuator.
        """
        dof_start = 6 if self._has_free_joint else 0  # skip free joint DOFs in qvel

        # Map: for each actuator, find which DOF it controls
        actuator_to_dof = {}
        for act_idx in range(self.model.nu):
            jnt_id = self.model.actuator_trnid[act_idx, 0]
            dof_addr = self.model.jnt_dofadr[jnt_id]
            dof_idx = dof_addr - dof_start  # relative to actuated DOFs
            actuator_to_dof[act_idx] = dof_idx
            act_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_idx)
                or f"act{act_idx}"
            )
            jnt_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
                or f"jnt{jnt_id}"
            )
            print(
                f"  Actuator[{act_idx}] '{act_name}' -> joint '{jnt_name}' -> DOF[{dof_idx}]"
            )

        # Invert: for each DOF index, which actuator index controls it
        self._dof_to_actuator = np.zeros(self._num_actuated_dofs, dtype=np.int32)
        for act_idx, dof_idx in actuator_to_dof.items():
            if 0 <= dof_idx < self._num_actuated_dofs:
                self._dof_to_actuator[dof_idx] = act_idx

        print(
            f"  DOF-to-actuator mapping built ({self._num_actuated_dofs} DOFs -> {self.model.nu} actuators)"
        )

    def _setup_control_parameters(self) -> None:
        """Cache PD control gains and effort limits from robot config (common DOF order)."""
        control_info = self.robot_config.control.control_info

        kp_list = []
        kd_list = []
        effort_list = []

        for dof_name in self._dof_names:
            info = control_info[dof_name]
            kp_list.append(info.stiffness)
            kd_list.append(info.damping)
            effort_list.append(
                info.effort_limit if info.effort_limit is not None else 1000.0
            )

        self._kp = np.array(kp_list, dtype=np.float64)
        self._kd = np.array(kd_list, dtype=np.float64)
        self._effort_limits = np.array(effort_list, dtype=np.float64)

    def _configure_actuators_for_pd(self) -> None:
        """Convert motor actuators to position (PD) actuators in the compiled model.

        MuJoCo position actuators compute PD torques implicitly at every substep:
            force = kp * (ctrl - q) - kd * qd

        This is achieved by setting:
            gainprm[0] = kp
            biasprm = [0, -kp, -kd]
            biastype = mjBIAS_AFFINE (1)

        After this, data.ctrl[i] = target_position for actuator i.
        MuJoCo handles the PD at every substep internally.
        """
        from protomotions.robot_configs.base import ControlType

        if self.control_type != ControlType.BUILT_IN_PD:
            print(
                "  Actuator config: TORQUE/PROPORTIONAL mode, keeping motor actuators"
            )
            return

        # Get PD gains in sim DOF order (matching actuator order via mapping)
        dof_start = 6 if self._has_free_joint else 0

        for act_idx in range(self.model.nu):
            jnt_id = self.model.actuator_trnid[act_idx, 0]
            dof_addr = self.model.jnt_dofadr[jnt_id]
            dof_idx = dof_addr - dof_start  # index into common DOF arrays

            # Find which common DOF this actuator controls
            # dof_idx is in sim DOF order; find corresponding common DOF
            # Use dof_convert_to_common if available, otherwise direct mapping
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if jnt_name in self.robot_config.control.control_info:
                info = self.robot_config.control.control_info[jnt_name]
                kp = info.stiffness
                kd = info.damping
                effort = info.effort_limit if info.effort_limit is not None else 1000.0
            else:
                # Fallback to common DOF order arrays
                kp = self._kp[dof_idx]
                kd = self._kd[dof_idx]
                effort = self._effort_limits[dof_idx]

            # Configure as position actuator:
            # force = gainprm[0] * ctrl + biasprm[0] + biasprm[1] * q + biasprm[2] * qd
            #       = kp * ctrl + 0 + (-kp) * q + (-kd) * qd
            #       = kp * (ctrl - q) - kd * qd
            self.model.actuator_gainprm[act_idx, 0] = kp
            self.model.actuator_biastype[act_idx] = 1  # mjBIAS_AFFINE
            self.model.actuator_biasprm[act_idx, 0] = 0.0
            self.model.actuator_biasprm[act_idx, 1] = -kp
            self.model.actuator_biasprm[act_idx, 2] = -kd

            # Set force limits on actuator
            self.model.actuator_forcerange[act_idx, 0] = -effort
            self.model.actuator_forcerange[act_idx, 1] = effort
            self.model.actuator_ctrllimited[act_idx] = (
                0  # Don't limit ctrl (it's position)
            )
            self.model.actuator_forcelimited[act_idx] = 1  # Limit output force

            act_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_idx)
                or f"act{act_idx}"
            )
            print(
                f"  Actuator '{act_name}' -> PD: "
                f"kp={kp:.2f}, kd={kd:.2f}, effort_limit={effort:.0f}"
            )

        print("  Configured all actuators as position (PD) controllers")

    def _configure_explicit_pd(self) -> None:
        """Set up explicit PD mode: keep motor actuators, cache sim-order gains.

        In explicit mode, we compute PD torques manually at each physics substep
        and write them to data.ctrl as raw torques. This matches how RoboJuDo
        and real hardware PD loops work.
        """
        control_info = self.robot_config.control.control_info

        for act_idx in range(self.model.nu):
            jnt_id = self.model.actuator_trnid[act_idx, 0]
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if jnt_name in control_info:
                effort = control_info[jnt_name].effort_limit or 1000.0
                self.model.actuator_forcerange[act_idx, 0] = -effort
                self.model.actuator_forcerange[act_idx, 1] = effort
                self.model.actuator_forcelimited[act_idx] = 1

    def _cache_sim_order_pd_gains(self) -> None:
        """Cache PD gains reordered to sim DOF order for explicit PD mode."""
        dof_to_sim = self.data_conversion.dof_convert_to_sim.numpy()
        self._kp_sim = self._kp[dof_to_sim]
        self._kd_sim = self._kd[dof_to_sim]
        self._effort_limits_sim = self._effort_limits[dof_to_sim]

    def _recompute_explicit_pd(self) -> None:
        """Compute PD torques from current state and write to data.ctrl.

        Called at each physics substep in explicit PD mode.
        """
        if self._pd_targets_sim is None:
            return

        if self._has_free_joint:
            q = self.data.qpos[7:]
            qd = self.data.qvel[6:]
        else:
            q = self.data.qpos[:]
            qd = self.data.qvel[:]

        torques = self._kp_sim * (self._pd_targets_sim - q) - self._kd_sim * qd
        torques = np.clip(torques, -self._effort_limits_sim, self._effort_limits_sim)
        self._apply_torques_to_ctrl(torques.astype(np.float32))

    def _init_viewer(self) -> None:
        """Initialize passive viewer for visualization."""
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self._mujoco_key_callback,
        )
        self._viewer_initialized = True
        # Ensure viewer is closed on exit to prevent hangs
        atexit.register(self._close_viewer)

        # Set up camera tracking
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10.0
        self.viewer.cam.azimuth = 180.0
        self.viewer.cam.trackbodyid = 1  # Track pelvis (first non-world body)
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

        log.info("MuJoCo passive viewer launched (tracking body 1)")

    def _close_viewer(self) -> None:
        """Close the viewer if it's still running."""
        if self.viewer is not None and self._viewer_initialized:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
            self._viewer_initialized = False

    def shutdown(self) -> None:
        """Clean shutdown of the MuJoCo simulator and viewer."""
        self._close_viewer()

    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        """Extract body and DOF names from MuJoCo model.

        Excludes projectile bodies (names starting with 'projectile_').
        """
        # Extract body names (skip body 0 which is "world" and projectile bodies)
        body_names = []
        for i in range(1, self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and not name.startswith("projectile_"):
                body_names.append(name)

        # Extract DOF names from joints (exclude projectile free joints)
        dof_names = []
        for i in range(self.model.njnt):
            jnt_type = self.model.jnt_type[i]
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)

            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            elif jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                dof_names.append(jnt_name)
            elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
                dof_names.append(jnt_name)
            elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                dof_names.extend([f"{jnt_name}_x", f"{jnt_name}_y", f"{jnt_name}_z"])

        return SimBodyOrdering(body_names=body_names, dof_names=dof_names)

    def _set_simulator_env_state(
        self,
        new_states: ResetState,
        new_object_states: Optional[ObjectState] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Set simulator state (qpos/qvel) and recompute FK.

        For MuJoCo (single env), we force XY spawn at origin so the robot
        stays near the ground plane center and camera tracking works well.
        """
        root_pos = new_states.root_pos[0].cpu().numpy().copy()
        root_rot = new_states.root_rot[0].cpu().numpy()
        root_vel = new_states.root_vel[0].cpu().numpy()
        root_ang_vel = new_states.root_ang_vel[0].cpu().numpy()
        dof_pos = new_states.dof_pos[0].cpu().numpy()
        dof_vel = new_states.dof_vel[0].cpu().numpy()

        # Force XY to origin for single env (keep Z height from motion)
        root_pos[0] = 0.0
        root_pos[1] = 0.0

        if self._has_free_joint:
            self.data.qpos[0:3] = root_pos
            self.data.qpos[3:7] = root_rot  # wxyz
            self.data.qpos[7 : 7 + self._num_actuated_dofs] = dof_pos

            self.data.qvel[0:3] = root_vel
            self.data.qvel[3:6] = root_ang_vel
            self.data.qvel[6 : 6 + self._num_actuated_dofs] = dof_vel
        else:
            self.data.qpos[: self._num_actuated_dofs] = dof_pos
            self.data.qvel[: self._num_actuated_dofs] = dof_vel

        # Clear forces
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        # Recompute forward kinematics
        mujoco.mj_forward(self.model, self.data)

    def _apply_torques_to_ctrl(self, torques_sim_order: np.ndarray) -> None:
        """Write torques (in sim DOF order) to data.ctrl (in actuator order)."""
        self.data.ctrl[self._dof_to_actuator] = torques_sim_order
        self._last_applied_torques = torques_sim_order.copy()

    def _physics_step(self) -> None:
        """Execute physics step with decimation.

        Two PD modes for BUILT_IN_PD (selected via config.use_implicit_pd):
          - Implicit: MuJoCo position actuators handle PD internally each substep.
          - Explicit: We recompute PD torques at each substep (1kHz), matching
            RoboJuDo and real hardware PD loops.

        For TORQUE/PROPORTIONAL modes, torques are applied once and held constant.
        """
        from protomotions.robot_configs.base import ControlType

        # Apply control (base class calls _apply_simulator_pd_targets
        # or _apply_simulator_torques which write to data.ctrl)
        self._apply_control()

        use_implicit_pd = getattr(self.config, "use_implicit_pd", True)
        use_explicit_substep_pd = (
            not use_implicit_pd
            and self.control_type == ControlType.BUILT_IN_PD
        )

        if use_explicit_substep_pd:
            # Explicit PD: recompute torques from current state at each substep
            for _ in range(self.decimation):
                self._recompute_explicit_pd()
                # print("Recomputed explicit PD torques")
                mujoco.mj_step(self.model, self.data)
        else:
            # Implicit PD (position actuators) or TORQUE/PROPORTIONAL mode
            for _ in range(self.decimation):
                mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Periodic state monitoring
        if self._step_count % 100 == 1:
            self._print_state_debug()

        # Sync viewer if active
        if self.viewer is not None and self._viewer_initialized:
            self.viewer.sync()

    def _print_state_debug(self) -> None:
        """Print state summary for debugging."""
        if self._has_free_joint:
            root_pos = self.data.qpos[0:3]
            root_quat = self.data.qpos[3:7]
            root_vel = self.data.qvel[0:3]
            dof_pos = self.data.qpos[7 : 7 + self._num_actuated_dofs]
            dof_vel = self.data.qvel[6 : 6 + self._num_actuated_dofs]
        else:
            root_pos = np.zeros(3)
            root_quat = np.array([1, 0, 0, 0])
            root_vel = np.zeros(3)
            dof_pos = self.data.qpos[: self._num_actuated_dofs]
            dof_vel = self.data.qvel[: self._num_actuated_dofs]

        has_nan = np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel))
        max_pos = np.max(np.abs(dof_pos)) if len(dof_pos) > 0 else 0
        max_vel = np.max(np.abs(dof_vel)) if len(dof_vel) > 0 else 0
        max_ctrl = np.max(np.abs(self.data.ctrl)) if self.model.nu > 0 else 0

        status = "NaN!" if has_nan else "OK"
        print(
            f"  [Step {self._step_count}] {status} | "
            f"root_pos=[{root_pos[0]:.2f}, {root_pos[1]:.2f}, {root_pos[2]:.2f}] "
            f"root_quat=[{root_quat[0]:.2f}, {root_quat[1]:.2f}, {root_quat[2]:.2f}, {root_quat[3]:.2f}] "
            f"root_vel=[{root_vel[0]:.2f}, {root_vel[1]:.2f}, {root_vel[2]:.2f}] | "
            f"max_dof_pos={max_pos:.3f} max_dof_vel={max_vel:.3f} max_ctrl={max_ctrl:.1f} "
            f"ncon={self.data.ncon}"
        )

    def _apply_root_velocity_impulse(
        self,
        linear_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Apply velocity impulse to root by adding to current velocities."""
        if not self._has_free_joint:
            return

        lin_vel = linear_velocity[0].cpu().numpy()
        ang_vel = angular_velocity[0].cpu().numpy()

        self.data.qvel[0:3] += lin_vel
        self.data.qvel[3:6] += ang_vel

    def _get_simulator_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RootOnlyState:
        """Get root state from qpos/qvel."""
        if self._has_free_joint:
            root_pos = self.data.qpos[0:3].copy()
            root_rot = self.data.qpos[3:7].copy()
            root_vel = self.data.qvel[0:3].copy()
            root_ang_vel = self.data.qvel[3:6].copy()
        else:
            root_pos = np.zeros(3, dtype=np.float32)
            root_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            root_vel = np.zeros(3, dtype=np.float32)
            root_ang_vel = np.zeros(3, dtype=np.float32)

        return RootOnlyState(
            root_pos=_to_torch_f32(root_pos).unsqueeze(0),
            root_rot=_to_torch_f32(root_rot).unsqueeze(0),
            root_vel=_to_torch_f32(root_vel).unsqueeze(0),
            root_ang_vel=_to_torch_f32(root_ang_vel).unsqueeze(0),
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_bodies_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Get rigid body states from MuJoCo FK results (robot bodies only)."""
        nb = self._num_robot_bodies
        body_pos = self.data.xpos[1 : 1 + nb, :].copy()
        body_rot = self.data.xquat[1 : 1 + nb, :].copy()

        # cvel is [ang_vel(3), lin_vel(3)]
        body_ang_vel = self.data.cvel[1 : 1 + nb, 0:3].copy()
        body_vel = self.data.cvel[1 : 1 + nb, 3:6].copy()

        return RobotState(
            rigid_body_pos=_to_torch_f32(body_pos).unsqueeze(0),
            rigid_body_rot=_to_torch_f32(body_rot).unsqueeze(0),
            rigid_body_vel=_to_torch_f32(body_vel).unsqueeze(0),
            rigid_body_ang_vel=_to_torch_f32(body_ang_vel).unsqueeze(0),
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Get DOF positions and velocities (actuated DOFs only)."""
        if self._has_free_joint:
            dof_pos = self.data.qpos[7 : 7 + self._num_actuated_dofs].copy()
            dof_vel = self.data.qvel[6 : 6 + self._num_actuated_dofs].copy()
        else:
            dof_pos = self.data.qpos[: self._num_actuated_dofs].copy()
            dof_vel = self.data.qvel[: self._num_actuated_dofs].copy()

        return RobotState(
            dof_pos=_to_torch_f32(dof_pos).unsqueeze(0),
            dof_vel=_to_torch_f32(dof_vel).unsqueeze(0),
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_forces(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Get applied DOF forces (track from last control application)."""
        dof_forces = self._last_applied_torques.copy()

        return RobotState(
            dof_forces=_to_torch_f32(dof_forces).unsqueeze(0),
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_limits_for_verification(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get DOF limits from MuJoCo model."""
        start_idx = 1 if self._has_free_joint else 0
        jnt_range = self.model.jnt_range[start_idx:, :]

        lower_limits = _to_torch_f32(jnt_range[:, 0].copy())
        upper_limits = _to_torch_f32(jnt_range[:, 1].copy())

        return lower_limits, upper_limits

    def _get_simulator_bodies_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Extract contact forces from MuJoCo contact buffer (robot bodies only)."""
        nb = self._num_robot_bodies
        contact_forces = np.zeros((nb, 3), dtype=np.float32)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            contact_force_local = c_array[0:3]

            frame = contact.frame.reshape(3, 3)
            contact_force_world = frame.T @ contact_force_local

            # Only accumulate for robot bodies (body_id 1..nb)
            if 0 < body1 <= nb:
                contact_forces[body1 - 1] += contact_force_world
            if 0 < body2 <= nb:
                contact_forces[body2 - 1] -= contact_force_world

        return RobotState(
            rigid_body_contact_forces=_to_torch_f32(contact_forces).unsqueeze(0),
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """Return empty object state (no scene support)."""
        return ObjectState(state_conversion=StateConversion.SIMULATOR)

    def _get_simulator_object_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """Return empty object contact buffer (no scene support)."""
        return ObjectState(state_conversion=StateConversion.SIMULATOR)

    def _apply_simulator_pd_targets(self, pd_targets: torch.Tensor) -> None:
        """Apply PD position targets.

        pd_targets are in sim DOF ordering (already converted by base class).

        Two modes:
          - Implicit (use_implicit_pd=True): Write position targets to data.ctrl.
            MuJoCo position actuators compute PD internally each substep.
          - Explicit (use_implicit_pd=False): Cache targets. PD torques are
            recomputed at each substep in _physics_step via _recompute_explicit_pd.
        """
        targets = pd_targets[0].detach().cpu().numpy()

        # Apply EMA filter
        alpha = self._action_filter_alpha
        if alpha < 1.0:
            if self._prev_pd_targets is None:
                self._prev_pd_targets = targets.copy()
            targets = alpha * targets + (1.0 - alpha) * self._prev_pd_targets
            self._prev_pd_targets = targets.copy()

        if getattr(self.config, "use_implicit_pd", True):
            # Implicit: write position targets to ctrl (MuJoCo handles PD)
            self.data.ctrl[self._dof_to_actuator] = targets
        else:
            # Explicit: cache targets, torques computed per-substep in _physics_step
            self._pd_targets_sim = targets.astype(np.float64)
            if self._kp_sim is None:
                self._cache_sim_order_pd_gains()
            # Initial PD computation so data.ctrl has reasonable values
            self._recompute_explicit_pd()

    def _apply_simulator_torques(self, torques: torch.Tensor) -> None:
        """Apply torques directly (for ControlType.TORQUE or PROPORTIONAL).

        Torques are in sim DOF ordering (already converted by base class).
        """
        torques_np = torques[0].detach().cpu().numpy().astype(np.float32)
        self._apply_torques_to_ctrl(torques_np)

    # ===== Projectile Implementation =====
    def _get_projectile_positions_rotations(self) -> tuple:
        """Return projectile (positions, rotations_xyzw) from MuJoCo qpos.

        MuJoCo stores wxyz quaternions — convert to xyzw.
        """
        n_proj = self._proj_config.num_projectiles
        pos = torch.zeros(1, n_proj, 3)
        rot = torch.zeros(1, n_proj, 4)
        for pid in range(n_proj):
            qp = self._proj_qpos_start + pid * 7
            pos[0, pid] = torch.from_numpy(self.data.qpos[qp : qp + 3].copy()).float()
            rot_wxyz = torch.from_numpy(self.data.qpos[qp + 3 : qp + 7].copy()).float()
            rot[0, pid] = torch.cat([rot_wxyz[1:4], rot_wxyz[0:1]])
        return pos, rot

    def _create_projectiles(self, config: ProjectileConfig) -> None:
        """Projectile bodies are already injected into MJCF during model load.

        Cache geom indices and disable collisions so hidden projectiles
        (placed below the ground plane) don't get ejected upward by the
        MuJoCo contact solver.
        """
        self._proj_geom_ids = []
        for i in range(config.num_projectiles):
            body_name = f"projectile_{i}"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                continue
            for gid in range(self.model.ngeom):
                if self.model.geom_bodyid[gid] == body_id:
                    self._proj_geom_ids.append(gid)
                    break
        self._disable_projectile_collisions()

    def _disable_projectile_collisions(self) -> None:
        """Disable collisions for all projectile geoms."""
        for gid in self._proj_geom_ids:
            self.model.geom_contype[gid] = 0
            self.model.geom_conaffinity[gid] = 0

    def _enable_projectile_collision(self, proj_idx: int) -> None:
        """Enable collisions for a single projectile geom."""
        if proj_idx < len(self._proj_geom_ids):
            gid = self._proj_geom_ids[proj_idx]
            self.model.geom_contype[gid] = 1
            self.model.geom_conaffinity[gid] = 1

    def _set_projectile_root_states(
        self,
        proj_indices: torch.Tensor,
        positions: torch.Tensor,
        rotations_xyzw: torch.Tensor,
        velocities: torch.Tensor,
        ang_velocities: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Set projectile state by writing directly to qpos/qvel arrays.

        Collision filtering: projectiles above hide_z get collisions enabled,
        those at hide_z get collisions disabled so the contact solver doesn't
        eject them through the ground plane.
        """
        # MuJoCo uses wxyz quaternion format
        rot_wxyz = rotations_xyzw[:, [3, 0, 1, 2]]
        hide_z = self._proj_config.hide_z

        for i in range(len(env_ids)):
            pid = proj_indices[i].item()
            qp = self._proj_qpos_start + pid * 7
            qv = self._proj_qvel_start + pid * 6
            self.data.qpos[qp : qp + 3] = positions[i].cpu().numpy()
            self.data.qpos[qp + 3 : qp + 7] = rot_wxyz[i].cpu().numpy()
            self.data.qvel[qv : qv + 3] = velocities[i].cpu().numpy()
            self.data.qvel[qv + 3 : qv + 6] = ang_velocities[i].cpu().numpy()

            # Toggle collisions based on whether this is a hide or throw
            z_pos = positions[i, 2].item()
            if z_pos <= hide_z + 0.1:
                if pid < len(self._proj_geom_ids):
                    gid = self._proj_geom_ids[pid]
                    self.model.geom_contype[gid] = 0
                    self.model.geom_conaffinity[gid] = 0
            else:
                self._enable_projectile_collision(pid)

        mujoco.mj_forward(self.model, self.data)

    def _mujoco_key_callback(self, keycode: int) -> None:
        """Handle keyboard events from MuJoCo passive viewer."""
        if keycode == ord("J") or keycode == ord("j"):
            self._throw_projectile()
        elif keycode == ord("R") or keycode == ord("r"):
            self._requested_reset()
        elif keycode == ord("L") or keycode == ord("l"):
            self._toggle_video_record()
        elif keycode == ord("M") or keycode == ord("m"):
            self._toggle_markers()

    def _write_viewport_to_file(self, file_name: str) -> None:
        """Render current view to file."""
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        renderer.update_scene(self.data)
        pixels = renderer.render()

        import matplotlib.pyplot as plt

        plt.imsave(file_name, pixels)
        renderer.close()

    def _init_camera(self) -> None:
        """Initialize camera position and orientation."""
        # Camera already set up in _init_viewer
        pass

    def _update_simulator_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """Update visualization markers (no-op for now)."""
        pass

    def render(self) -> None:
        """Render current simulation state."""
        if not self.headless and self.viewer is not None and self._viewer_initialized:
            if not self.viewer.is_running():
                self._viewer_initialized = False
                return

        super().render()
