import typer
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import torch
from scipy.spatial.transform import Rotation as sRot
import uuid

import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf
from dm_control.viewer import user_input
from loop_rate_limiters import RateLimiter

from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from smpl_sim.smpllib.smpl_joint_names import (
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)

import mink
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree

from tqdm import tqdm


@dataclass
class KeyCallback:
    pause: bool = False
    first_pose_only: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_SPACE:
            self.pause = not self.pause
        elif key == user_input.KEY_ENTER:
            self.first_pose_only = not self.first_pose_only
            print(f"First pose only: {self.first_pose_only}")


_HERE = Path(__file__).parent

_HAND_NAMES = ["Index", "Middle", "Pinky", "Ring", "Thumb", "Wrist"]
_IMPORTANT_NAMES = ["Shoulder", "Knee", "Toe", "Elbow", "Head"]

_H1_KEYPOINT_TO_JOINT = {
    # We provide higher weight to the "end of graph nodes" as they are more important for recovering the overall motion
    "Head": {"name": "head", "weight": 3.0},
    "Pelvis": {"name": "pelvis", "weight": 1.0},
    "L_Hip": {"name": "left_hip_yaw_link", "weight": 1.0},
    "R_Hip": {"name": "right_hip_yaw_link", "weight": 1.0},
    "L_Knee": {"name": "left_knee_link", "weight": 1.0},
    "R_Knee": {"name": "right_knee_link", "weight": 1.0},
    "L_Ankle": {"name": "left_ankle_link", "weight": 3.0},
    "R_Ankle": {"name": "right_ankle_link", "weight": 3.0},
    "L_Toe": {"name": "left_foot_link", "weight": 3.0},
    "R_Toe": {"name": "right_foot_link", "weight": 3.0},
    "L_Elbow": {"name": "left_elbow_link", "weight": 1.0},
    "R_Elbow": {"name": "right_elbow_link", "weight": 1.0},
    "L_Wrist": {"name": "left_arm_end_effector", "weight": 3.0},
    "R_Wrist": {"name": "right_arm_end_effector", "weight": 3.0},
    "L_Shoulder": {"name": "left_shoulder_pitch_link", "weight": 1.0},
    "R_Shoulder": {"name": "right_shoulder_pitch_link", "weight": 1.0},
}

_KEYPOINT_TO_JOINT_MAP = {
    "h1": _H1_KEYPOINT_TO_JOINT,
}

_RESCALE_FACTOR = {
    "h1": np.array([1.0, 1.0, 1.1]),
}

_OFFSET = {
    "h1": 0.0,
}

_ROOT_LINK = {
    "h1": "pelvis",
}

_H1_VELOCITY_LIMITS = {
    "left_hip_yaw_joint": 23,
    "left_hip_roll_joint": 23,
    "left_hip_pitch_joint": 23,
    "left_knee_joint": 14,
    "left_ankle_joint": 9,
    "right_hip_yaw_joint": 23,
    "right_hip_roll_joint": 23,
    "right_hip_pitch_joint": 23,
    "right_knee_joint": 14,
    "right_ankle_joint": 9,
    "torso_joint": 23,
    "left_shoulder_pitch_joint": 9,
    "left_shoulder_roll_joint": 9,
    "left_shoulder_yaw_joint": 20,
    "left_elbow_joint": 20,
    "right_shoulder_pitch_joint": 9,
    "right_shoulder_roll_joint": 9,
    "right_shoulder_yaw_joint": 20,
    "right_elbow_joint": 20,
}

_VEL_LIMITS = {
    "h1": _H1_VELOCITY_LIMITS,
}


def construct_model(robot_name: str, keypoint_names: Sequence[str]):
    root = mjcf.RootElement()

    root.visual.headlight.ambient = ".4 .4 .4"
    root.visual.headlight.diffuse = ".8 .8 .8"
    root.visual.headlight.specular = "0.1 0.1 0.1"
    root.visual.rgba.haze = "0 0 0 0"
    root.visual.quality.shadowsize = "8192"

    # 4k resolution.
    getattr(root.visual, "global").offheight = "2160"
    getattr(root.visual, "global").offwidth = "3840"

    root.asset.add(
        "texture",
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1="0 0 0",
        rgb2="0 0 0",
        width="800",
        height="800",
    )
    root.asset.add(
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1="0 0 0",
        rgb2="0 0 0",
        width="300",
        height="300",
        mark="edge",
        markrgb=".2 .3 .4",
    )
    root.asset.add(
        "material",
        name="grid",
        texture="grid",
        texrepeat="1 1",
        texuniform="true",
        reflectance=".2",
    )
    root.worldbody.add(
        "geom", name="ground", type="plane", size="0 0 .01", material="grid"
    )

    for keypoint_name in keypoint_names:
        if any(hand_name in keypoint_name for hand_name in _HAND_NAMES):
            size = 0.01
        else:
            size = 0.02
        body = root.worldbody.add(
            "body", name=f"keypoint_{keypoint_name}", mocap="true"
        )
        rgb = np.random.rand(3)
        body.add(
            "site",
            name=f"site_{keypoint_name}",
            type="sphere",
            size=f"{size}",
            rgba=f"{rgb[0]} {rgb[1]} {rgb[2]} 1",
        )
        if keypoint_name == "Pelvis":
            body.add("light", pos="0 0 2", directional="false")
            root.worldbody.add(
                "camera",
                name="tracking01",
                pos=[2.972, -0.134, 1.303],
                xyaxes="0.294 0.956 0.000 -0.201 0.062 0.978",
                mode="trackcom",
            )
            root.worldbody.add(
                "camera",
                name="tracking02",
                pos="4.137 2.642 1.553",
                xyaxes="-0.511 0.859 0.000 -0.123 -0.073 0.990",
                mode="trackcom",
            )

    if robot_name == "h1":
        humanoid_mjcf = mjcf.from_path("protomotions/data/assets/mjcf/h1.xml")
        humanoid_mjcf.worldbody.add(
            "camera",
            name="front_track",
            pos="-0.120 3.232 1.064",
            xyaxes="-1.000 -0.002 -0.000 0.000 -0.103 0.995",
            mode="trackcom",
        )
    else:
        raise ValueError(f"Unknown robot name: {robot_name}")
    root.include_copy(humanoid_mjcf)

    root_str = to_string(root, pretty=True)
    assets = get_assets(root)
    return mujoco.MjModel.from_xml_string(root_str, assets)


def to_string(
    root: mjcf.RootElement,
    precision: float = 17,
    zero_threshold: float = 0.0,
    pretty: bool = False,
) -> str:
    from lxml import etree

    xml_string = root.to_xml_string(precision=precision, zero_threshold=zero_threshold)
    root = etree.XML(xml_string, etree.XMLParser(remove_blank_text=True))

    # Remove hashes from asset filenames.
    tags = ["mesh", "texture"]
    for tag in tags:
        assets = [
            asset
            for asset in root.find("asset").iter()
            if asset.tag == tag and "file" in asset.attrib
        ]
        for asset in assets:
            name, extension = asset.get("file").split(".")
            asset.set("file", ".".join((name[:-41], extension)))

    if not pretty:
        return etree.tostring(root, pretty_print=True).decode()

    # Remove auto-generated names.
    for elem in root.iter():
        for key in elem.keys():
            if key == "name" and "unnamed" in elem.get(key):
                elem.attrib.pop(key)

    # Get string from lxml.
    xml_string = etree.tostring(root, pretty_print=True)

    # Remove redundant attributes.
    xml_string = xml_string.replace(b' gravcomp="0"', b"")

    # Insert spaces between top-level elements.
    lines = xml_string.splitlines()
    newlines = []
    for line in lines:
        newlines.append(line)
        if line.startswith(b"  <"):
            if line.startswith(b"  </") or line.endswith(b"/>"):
                newlines.append(b"")
    newlines.append(b"")
    xml_string = b"\n".join(newlines)

    return xml_string.decode()


def get_assets(root: mjcf.RootElement) -> dict[str, bytes]:
    assets = {}
    for file, payload in root.get_assets().items():
        name, extension = file.split(".")
        assets[".".join((name[:-41], extension))] = payload
    return assets


def create_h1_motion(
    poses: np.ndarray, trans: np.ndarray, orig_global_trans: np.ndarray, mocap_fr: float
) -> SkeletonMotion:
    """Create a SkeletonMotion for H1 robot from poses and translations.
    Args:
        poses: Joint angles from mujoco [N, num_dof] in proper ordering - groups of 3 hinge joints per joint
        trans: Root transform [N, 7] (pos + quat)
        orig_global_trans: Original global translations [N, num_joints, 3]
        mocap_fr: Motion capture framerate
    Returns:
        SkeletonMotion: Motion data in proper format for H1
    """
    from data.scripts.retargeting.torch_humanoid_batch import Humanoid_Batch
    from data.scripts.retargeting.config import get_config

    # Initialize H1 humanoid batch with config
    cfg = get_config("h1")
    humanoid_batch = Humanoid_Batch(cfg)

    # Convert poses to proper format
    B, seq_len = 1, poses.shape[0]

    # Convert to tensor format
    poses_tensor = torch.from_numpy(poses).float().reshape(B, seq_len, -1, 1)

    # Add root rotation from trans quaternion
    root_rot = sRot.from_quat(np.roll(trans[:, 3:7], -1)).as_rotvec()
    root_rot_tensor = torch.from_numpy(root_rot).float().reshape(B, seq_len, 1, 3)

    # Combine root rotation with joint poses
    poses_tensor = torch.cat(
        [
            root_rot_tensor,
            humanoid_batch.dof_axis * poses_tensor,
            torch.zeros((1, seq_len, len(cfg.extend_config), 3)),
        ],
        axis=2,
    )

    # Prepare root translation
    trans_tensor = torch.from_numpy(trans[:, :3]).float().reshape(B, seq_len, 3)

    # Perform forward kinematics
    motion_data = humanoid_batch.fk_batch(
        poses_tensor, trans_tensor, return_full=True, dt=1.0 / mocap_fr
    )

    # Convert back to proper kinematic structure
    fk_return_proper = humanoid_batch.convert_to_proper_kinematic(motion_data)

    # Get lowest heights for both original and retargeted motions
    orig_lowest_heights = torch.from_numpy(orig_global_trans[..., 2].min(axis=1))
    retarget_lowest_heights = (
        fk_return_proper.global_translation[..., 2].min(dim=-1).values
    )

    # Calculate height adjustment to match original motion's lowest points
    height_offset = (retarget_lowest_heights - orig_lowest_heights).unsqueeze(-1)

    # Adjust global translations to match original heights
    fk_return_proper.global_translation[..., 2] -= height_offset

    curr_motion = {
        k: v.squeeze().detach().cpu() if torch.is_tensor(v) else v
        for k, v in fk_return_proper.items()
    }
    return curr_motion


def create_skeleton_motion(
    poses: np.ndarray,
    trans: np.ndarray,
    skeleton_tree: SkeletonTree,
    orig_global_trans: np.ndarray,
    mocap_fr: float,
) -> SkeletonMotion:
    """Create a SkeletonMotion from poses and translations.
    Args:
        poses: Joint angles from mujoco [N, 153] - groups of 3 hinge joints per joint
        trans: Root transform [N, 7] (pos + quat)
        skeleton_tree: Skeleton tree for the model
        orig_global_trans: Original global translations [N, num_joints, 3]
        mocap_fr: Motion capture framerate
    """
    n_frames = poses.shape[0]
    pose_quat = np.zeros((n_frames, 51, 4))  # 51 joints, each with quaternion

    # Convert each joint's 3 hinge rotations to a single quaternion
    for i in range(51):  # 51 joints
        angles = poses[
            :, i * 3 : (i + 1) * 3
        ]  # Get angles for this joint's x,y,z hinges
        pose_quat[:, i] = sRot.from_euler("XYZ", angles).as_quat()

    # Combine root transform and joint rotations
    full_pose = np.zeros((n_frames, 52, 4))  # 52 total joints (root + 51 joints)
    full_pose[:, 0] = np.roll(trans[:, 3:7], -1)  # Root quaternion
    full_pose[:, 1:] = pose_quat  # Other joint quaternions

    # Create skeleton state
    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(full_pose),
        torch.from_numpy(trans[:, :3]),
        is_local=True,
    )

    # Get global rotations and positions
    pose_quat_global = sk_state.global_rotation.numpy()
    global_pos = sk_state.global_translation.numpy()

    # Get lowest heights for both original and retargeted motions
    orig_lowest_heights = orig_global_trans[..., 2].min(axis=1, keepdims=True)
    retarget_lowest_heights = global_pos[..., 2].min(axis=1, keepdims=True)

    # Calculate height adjustment to match original motion's lowest points
    height_offset = retarget_lowest_heights - orig_lowest_heights

    # Adjust root translation to match original heights
    adjusted_trans = trans.copy()
    adjusted_trans[:, 2] -= height_offset.squeeze()

    # Create new skeleton state with adjusted heights
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global),
        torch.from_numpy(adjusted_trans[:, :3]),
        is_local=False,
    )

    return SkeletonMotion.from_skeleton_state(new_sk_state, fps=mocap_fr)


def retarget_motion(motion: SkeletonMotion, robot_type: str, render: bool = False):
    global_translations = motion.global_translation.numpy()
    pose_quat_global = motion.global_rotation.numpy()
    pose_quat = motion.local_rotation.numpy()
    timeseries_length = global_translations.shape[0]
    fps = motion.fps

    smplx_mujoco_joint_names = SMPLH_MUJOCO_NAMES
    model = construct_model(robot_type, smplx_mujoco_joint_names)
    configuration = mink.Configuration(model)

    tasks = []

    frame_tasks = {}
    for joint_name, retarget_info in _KEYPOINT_TO_JOINT_MAP[robot_type].items():
        if robot_type == "h1":
            orientation_base_cost = 0
        else:
            orientation_base_cost = 0.0001
        task = mink.FrameTask(
            frame_name=retarget_info["name"],
            frame_type="body",
            position_cost=10.0 * retarget_info["weight"],
            orientation_cost=orientation_base_cost * retarget_info["weight"],
            lm_damping=1.0,
        )
        frame_tasks[retarget_info["name"]] = task
    tasks.extend(frame_tasks.values())

    posture_task = mink.PostureTask(model, cost=1.0)
    tasks.append(posture_task)

    # Prepare MuJoCo model and data
    model = configuration.model
    data = configuration.data

    key_callback = KeyCallback()

    # Modify the main processing loop to conditionally use the viewer
    if render:
        viewer_context = mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=key_callback,
        )
    else:
        # Use contextlib.nullcontext as a no-op context manager
        from contextlib import nullcontext

        viewer_context = nullcontext()

    retargeted_poses = []
    retargeted_trans = []

    with viewer_context as viewer:
        if render:
            # Set up camera only when rendering
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.cam("front_track").id

        # Directly set initial pose from first frame
        # Initialize qpos with zeros
        data.qpos[:] = 0

        # Set root position (first 3 values)
        data.qpos[0:3] = global_translations[0, 0]

        # Set root orientation (next 4 values)
        data.qpos[3:7] = pose_quat_global[0, 0]

        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_step(model, data)

        optimization_steps_per_frame = 2  # int(max(np.ceil(5.0 * 30 / fps), 1))
        rate = RateLimiter(frequency=fps * optimization_steps_per_frame)
        solver = "quadprog"

        t: int = int(np.ceil(-100.0 * fps / 30))
        vel = None

        # Create progress bar
        pbar = tqdm(total=timeseries_length, desc="Retargeting frames")

        while (render and viewer.is_running() or not render) and t < timeseries_length:
            if not key_callback.pause:
                # Set targets for current frame
                for i, (joint_name, retarget_info) in enumerate(
                    _KEYPOINT_TO_JOINT_MAP[robot_type].items()
                ):
                    body_idx = smplx_mujoco_joint_names.index(joint_name)
                    target_pos = global_translations[max(0, t), body_idx, :].copy()

                    if robot_type in _RESCALE_FACTOR:
                        target_pos *= _RESCALE_FACTOR[robot_type]
                    if robot_type in _OFFSET:
                        target_pos[2] += _OFFSET[robot_type]

                    target_rot = pose_quat_global[max(0, t), body_idx].copy()
                    rot_matrix = sRot.from_quat(target_rot).as_matrix()
                    rot = mink.SO3.from_matrix(rot_matrix)
                    tasks[i].set_target(
                        mink.SE3.from_rotation_and_translation(rot, target_pos)
                    )

                # Update keypoint positions.
                keypoint_pos = {}
                for keypoint_name, keypoint in zip(
                    smplx_mujoco_joint_names, global_translations[max(0, t)]
                ):
                    mid = model.body(f"keypoint_{keypoint_name}").mocapid[0]
                    data.mocap_pos[mid] = keypoint
                    keypoint_pos[keypoint_name] = keypoint

                # Perform multiple optimization steps
                for _ in range(optimization_steps_per_frame):
                    limits = [
                        mink.ConfigurationLimit(model),
                    ]
                    if robot_type in _VEL_LIMITS and t >= 0:
                        limits.append(
                            mink.VelocityLimit(model, _VEL_LIMITS[robot_type])
                        )
                    vel = mink.solve_ik(
                        configuration, tasks, rate.dt, solver, 1e-1, limits=limits
                    )
                    configuration.integrate_inplace(vel, rate.dt)
                    if render:
                        mujoco.mj_camlight(model, data)

                # Store poses and translations if we're past initialization
                if t >= 0:
                    retargeted_poses.append(data.qpos[7:].copy())
                    retargeted_trans.append(data.qpos[:7].copy())

                if render and key_callback.first_pose_only and t == 0:
                    print(
                        "First pose set. Press Enter to continue animation, Space to pause/unpause"
                    )
                    key_callback.pause = True
                    key_callback.first_pose_only = False

                t += 1
                if t >= 0:  # Only update progress bar for actual frames
                    pbar.update(1)

            if render:
                viewer.sync()
                rate.sleep()

        pbar.close()

    # Convert stored motion to numpy arrays
    retargeted_poses = np.stack(retargeted_poses)
    retargeted_trans = np.stack(retargeted_trans)

    # Create skeleton motion
    if robot_type == "h1":
        return create_h1_motion(
            retargeted_poses, retargeted_trans, global_translations, fps
        )
    else:
        skeleton_tree = SkeletonTree.from_mjcf(
            f"protomotions/data/assets/mjcf/{robot_type}.xml"
        )
        retargeted_motion = create_skeleton_motion(
            retargeted_poses, retargeted_trans, skeleton_tree, global_translations, fps
        )

    return retargeted_motion


def manually_retarget_motion(
    amass_data: str, output_path: str, robot_type: str, render: bool = False
):
    # Store retargeted motion data
    motion_data = dict(np.load(open(amass_data, "rb"), allow_pickle=True))

    mujoco_joint_names = SMPLH_MUJOCO_NAMES
    joint_names = SMPLH_BONE_ORDER_NAMES

    betas = motion_data["betas"]
    gender = motion_data["gender"]
    amass_pose = motion_data["poses"]
    amass_trans = motion_data["trans"]
    if "mocap_framerate" in motion_data:
        mocap_fr = motion_data["mocap_framerate"]
    else:
        mocap_fr = motion_data["mocap_frame_rate"]

    skip = int(mocap_fr // 30)

    pose_aa = torch.tensor(amass_pose[::skip])
    amass_trans = torch.tensor(amass_trans[::skip])
    betas = torch.from_numpy(betas)

    betas[:] = 0

    motion_data = {
        "pose_aa": pose_aa.numpy(),
        "trans": amass_trans.numpy(),
        "beta": betas.numpy(),
    }

    smpl_2_mujoco = [
        joint_names.index(q) for q in mujoco_joint_names if q in joint_names
    ]
    batch_size = motion_data["pose_aa"].shape[0]

    pose_aa = np.concatenate(
        [
            motion_data["pose_aa"][:, :66],
            motion_data["pose_aa"][:, 75:],
        ],
        axis=-1,
    )
    pose_aa_mj = pose_aa.reshape(batch_size, 52, 3)[:, smpl_2_mujoco]
    pose_quat = (
        sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, 52, 4)
    )

    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": True,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": "smplx",
        "sim": "isaacgym",
    }

    smpl_local_robot = SMPL_Robot(
        robot_cfg,
        data_dir="data/smpl",
    )

    smpl_local_robot.load_from_skeleton(betas=betas[None,], gender=[0], objs_info=None)
    TMP_SMPL_DIR = "/tmp/smpl"
    uuid_str = uuid.uuid4()
    smpl_local_robot.write_xml(f"{TMP_SMPL_DIR}/smpl_humanoid_{uuid_str}.xml")
    skeleton_tree = SkeletonTree.from_mjcf(
        f"{TMP_SMPL_DIR}/smpl_humanoid_{uuid_str}.xml"
    )

    root_trans_offset = (
        torch.from_numpy(motion_data["trans"]) + skeleton_tree.local_translation[0]
    )

    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
        torch.from_numpy(pose_quat),
        root_trans_offset,
        is_local=True,
    )

    timeseries_length = pose_aa.shape[0]
    pose_quat_global = (
        (
            sRot.from_quat(sk_state.global_rotation.reshape(-1, 4).numpy())
            * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
        )
        .as_quat()
        .reshape(timeseries_length, -1, 4)
    )

    trans = root_trans_offset.clone()
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global),
        trans,
        is_local=False,
    )
    new_sk_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30)
    sk_motion = retarget_motion(new_sk_motion, robot_type, render=render)
    if robot_type == "h1":
        torch.save(sk_motion, output_path)
    else:
        sk_motion.to_file(output_path)


if __name__ == "__main__":
    typer.run(manually_retarget_motion)
