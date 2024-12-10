"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Joint Monkey
------------
- Animates degree-of-freedom ranges for a given asset.
- Demonstrates usage of DOF properties and states.
- Demonstrates line drawing utilities to visualize DOF frames (origin and axis).
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from isaac_utils import torch_utils


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


# simple asset descriptor for selecting from a list


class AssetDesc:
    def __init__(self, file_name, asset_type, flip_visual_attachments=False):
        self.file_name = file_name
        self.asset_type = asset_type
        self.flip_visual_attachments = flip_visual_attachments


asset_descriptors = [
    AssetDesc("h1.urdf", "urdf", False),
    AssetDesc("smplx_humanoid.xml", "mjcf", False),
]


# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[
        {
            "name": "--asset_id",
            "type": int,
            "default": 0,
            "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1),
        },
        {
            "name": "--speed_scale",
            "type": float,
            "default": 1.0,
            "help": "Animation speed scale",
        },
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"},
        {
            "name": "--joint_name",
            "type": str,
            "default": None,
            "help": "Joint name to visualize",
        },
    ],
)

if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
    print(
        "*** Invalid asset_id specified.  Valid range is 0 to %d"
        % (len(asset_descriptors) - 1)
    )
    quit()

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
print(f"Using physics engine: {args.physics_engine}")
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim = gym.create_sim(
    args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
)

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
asset_root = "phys_anim/data/assets/" + asset_descriptors[args.asset_id].asset_type
asset_file = asset_descriptors[args.asset_id].file_name

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = asset_descriptors[
    args.asset_id
].flip_visual_attachments

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# get array of DOF names
dof_names = gym.get_asset_dof_names(asset)

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset)

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# get the position slice of the DOF state array
dof_positions = dof_states["pos"]

# get the limit-related slices of the DOF properties array
stiffnesses = dof_props["stiffness"]
dampings = dof_props["damping"]
armatures = dof_props["armature"]
has_limits = dof_props["hasLimits"]
lower_limits = dof_props["lower"]
upper_limits = dof_props["upper"]

# initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
defaults = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)
for i in range(num_dofs):
    if has_limits[i]:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
            upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
        # make sure our default position is in range
        if lower_limits[i] > 0.0:
            defaults[i] = lower_limits[i]
        elif upper_limits[i] < 0.0:
            defaults[i] = upper_limits[i]
    else:
        # set reasonable animation limits for unlimited joints
        if dof_types[i] == gymapi.DOF_ROTATION:
            # unlimited revolute joint
            lower_limits[i] = -math.pi
            upper_limits[i] = math.pi
        elif dof_types[i] == gymapi.DOF_TRANSLATION:
            # unlimited prismatic joint
            lower_limits[i] = -1.0
            upper_limits[i] = 1.0
    # set DOF position to default
    dof_positions[i] = defaults[i]
    # set speed depending on DOF type and range of motion
    if dof_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = args.speed_scale * clamp(
            2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi
        )
    else:
        speeds[i] = args.speed_scale * clamp(
            2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0
        )

# Print DOF properties
for i in range(num_dofs):
    print("DOF %d" % i)
    print("  Name:     '%s'" % dof_names[i])
    print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
    print("  Stiffness:  %r" % stiffnesses[i])
    print("  Damping:  %r" % dampings[i])
    print("  Armature:  %r" % armatures[i])
    print("  Limited?  %r" % has_limits[i])
    if has_limits[i]:
        print("    Lower   %f" % lower_limits[i])
        print("    Upper   %f" % upper_limits[i])

# set up the env grid
num_envs = 36
num_per_row = 6
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(17.2, 2.0, 16)
cam_target = gymapi.Vec3(5, -2.5, 13)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    gym.set_actor_dof_properties(env, actor_handle, dof_props)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

# joint animation states
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

# initialize animation state
anim_state = ANIM_FINISHED
current_dof = -1

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    speed = speeds[current_dof]

    # animate the dofs
    if anim_state == ANIM_SEEK_LOWER:
        dof_positions[current_dof] -= speed * dt
        if dof_positions[current_dof] <= lower_limits[current_dof]:
            dof_positions[current_dof] = lower_limits[current_dof]
            anim_state = ANIM_SEEK_UPPER
    elif anim_state == ANIM_SEEK_UPPER:
        dof_positions[current_dof] += speed * dt
        if dof_positions[current_dof] >= upper_limits[current_dof]:
            dof_positions[current_dof] = upper_limits[current_dof]
            anim_state = ANIM_SEEK_DEFAULT
    if anim_state == ANIM_SEEK_DEFAULT:
        dof_positions[current_dof] -= speed * dt
        if dof_positions[current_dof] <= defaults[current_dof]:
            dof_positions[current_dof] = defaults[current_dof]
            anim_state = ANIM_FINISHED
    elif anim_state == ANIM_FINISHED:
        dof_positions[current_dof] = defaults[current_dof]
        current_dof = (current_dof + 1) % num_dofs
        if args.joint_name is not None:
            while args.joint_name not in dof_names[current_dof]:
                current_dof = (current_dof + 1) % num_dofs
        anim_state = ANIM_SEEK_LOWER
        print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

    if args.show_axis:
        gym.clear_lines(viewer)

    # clone actor state in all of the environments
    for i in range(num_envs):
        gym.set_actor_dof_states(
            envs[i], actor_handles[i], dof_states, gymapi.STATE_POS
        )

        if args.show_axis:
            # get the DOF frame (origin and axis)
            dof_handle = gym.get_actor_dof_handle(
                envs[i], actor_handles[i], current_dof
            )
            frame = gym.get_dof_frame(envs[i], dof_handle)

            # draw a line from DOF origin along the DOF axis
            p1 = frame.origin
            p2 = frame.origin + frame.axis * 0.7
            color = gymapi.Vec3(1.0, 0.0, 0.0)
            gymutil.draw_line(p1, p2, color, gym, viewer, envs[i])

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
