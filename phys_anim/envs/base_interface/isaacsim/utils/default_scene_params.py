# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


default_physx_params = {
    ### Per-scene settings
    "use_gpu": False,
    "worker_thread_count": 4,
    "solver_type": 1,  # 0: PGS, 1:TGS
    "bounce_threshold_velocity": 0.2,
    "friction_offset_threshold": 0.04,  # A threshold of contact separation distance used to decide if a contact
    # point will experience friction forces.
    "friction_correlation_distance": 0.025,  # Contact points can be merged into a single friction anchor if the
    # distance between the contacts is smaller than correlation distance.
    # disabling these can be useful for debugging
    "enable_sleeping": True,
    "enable_stabilization": True,
    # GPU buffers
    "gpu_max_rigid_contact_count": 512 * 1024,
    "gpu_max_rigid_patch_count": 80 * 1024,
    "gpu_found_lost_pairs_capacity": 1024,
    "gpu_found_lost_aggregate_pairs_capacity": 1024,
    "gpu_total_aggregate_pairs_capacity": 1024,
    "gpu_max_soft_body_contacts": 1024 * 1024,
    "gpu_max_particle_contacts": 1024 * 1024,
    "gpu_heap_capacity": 64 * 1024 * 1024,
    "gpu_temp_buffer_capacity": 16 * 1024 * 1024,
    "gpu_max_num_partitions": 8,
    ### Per-actor settings ( can override in actor_options )
    "solver_position_iteration_count": 4,
    "solver_velocity_iteration_count": 1,
    "sleep_threshold": 0.0,  # Mass-normalized kinetic energy threshold below which an actor may go to sleep.
    # Allowed range [0, max_float).
    "stabilization_threshold": 0.0,  # Mass-normalized kinetic energy threshold below which an actor may
    # participate in stabilization. Allowed range [0, max_float).
    ### Per-body settings ( can override in actor_options )
    "enable_gyroscopic_forces": False,
    "density": 1000.0,  # density to be used for bodies that do not specify mass or density
    "max_depenetration_velocity": 100.0,
    ### Per-shape settings ( can override in actor_options )
    "contact_offset": 0.02,
    "rest_offset": 0.001,
}

default_physics_material = {
    "static_friction": 1.0,
    "dynamic_friction": 1.0,
    "restitution": 0.0,
}

default_sim_params = {
    "gravity": [0.0, 0.0, -9.81],
    "dt": 1.0 / 60.0,
    "substeps": 1,
    "use_gpu_pipeline": True,
    "add_ground_plane": True,
    "add_distant_light": True,
    "use_flatcache": True,
    "enable_scene_query_support": False,
    "enable_cameras": False,
    "disable_contact_processing": False,
    "default_physics_material": default_physics_material,
}

default_actor_options = {
    # -1 means use authored value from USD or default values from default_sim_params if not explicitly authored in USD.
    # If an attribute value is not explicitly authored in USD, add one with the value given here,
    # which overrides the USD default.
    "override_usd_defaults": False,
    "make_kinematic": -1,
    "enable_self_collisions": -1,
    "enable_gyroscopic_forces": -1,
    "solver_position_iteration_count": -1,
    "solver_velocity_iteration_count": -1,
    "sleep_threshold": -1,
    "stabilization_threshold": -1,
    "max_depenetration_velocity": -1,
    "density": -1,
    "mass": -1,
    "contact_offset": -1,
    "rest_offset": -1,
}
