# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The public attributes and methods of ``BaseEnv`` are exactly the documented list.

The forwarders to the state owners are properties of ``BaseEnv`` itself
(``base_env/env.py``, the *Forwarders* section, each with its in-tree
callers); a public name
that is not on the list below became public by accident. Class-level names are
checked without building anything; the instance's public attributes on the
real env over the fake sensor simulator.
"""

import pytest

from protomotions.envs.base_env.env import BaseEnv

# The forwarders to the state owners (env.py, the *Forwarders* section), each
# a property or a method with its callers named there. The tensor properties
# have no setter.
FORWARDERS = {
    "progress_buf",
    "reset_buf",
    "terminate_buf",
    "rew_buf",
    "max_episode_length",
    "reference_motion",
    "install_motion_lib",
}
TENSOR_FORWARDERS = {
    "progress_buf",
    "reset_buf",
    "terminate_buf",
    "rew_buf",
}
# The env's own public methods and properties (env.py).
ENV_OWN = {
    "step",
    "reset",
    "observe",
    "close",
    "save_state",
    "restore_state",
    "get_state_dict",
    "load_state_dict",
    "get_task_id",
    "get_obs",
    "context",
    "perturbations_enabled",
    "set_perturbations_enabled",
    "contact_body_ids",
    "non_termination_contact_body_ids",
    "is_simulation_running",
    "consume_reset_request",
    "on_epoch_end",
}
# Public instance attributes: the constructor's arguments, what is read off
# the simulator, the three objects code outside the env reaches for
# (deployment and the evaluators read ``sensors`` and ``control_manager``; the
# tasks, ADD and the mimic evaluator read the reference motion through
# ``reference``; the history ring and the two observation caches are
# private), the step's logging dict, and the storage of the two
# cached_property body-id tensors (which land in the instance dict once
# computed, at build).
INSTANCE_ATTRIBUTES = {
    "config",
    "robot_config",
    "device",
    "terrain",
    "scene_lib",
    "simulator",
    "num_envs",
    "dt",
    "sensors",
    "control_manager",
    "reference",
    "randomness",
    "extras",
    "contact_body_ids",
    "non_termination_contact_body_ids",
}


def _public(names):
    return {name for name in names if not name.startswith("_")}


def test_class_level_public_names_are_the_documented_list():
    assert _public(vars(BaseEnv)) == FORWARDERS | ENV_OWN
    # The class exposes nothing public beyond the two sets and has no base
    # but object (no inherited surprises).
    assert BaseEnv.__mro__[1:] == (object,)
    assert _public(dir(BaseEnv)) == FORWARDERS | ENV_OWN
    # The tensor forwarders are read-only names over the owners' buffers
    # (written in place, never rebound), and so is the reference-motion owner
    # (cleared and re-installed through its own methods); max_episode_length
    # keeps its setter for the curriculum.
    for name in (*TENSOR_FORWARDERS, "reference_motion"):
        assert getattr(BaseEnv, name).fset is None, name
    assert BaseEnv.max_episode_length.fset is not None


def test_instance_public_attributes_are_the_documented_list():
    pytest.importorskip("mujoco")
    from protomotions.tests.support.sensor_simulator import make_sensor_test_env

    env = make_sensor_test_env()
    try:
        env.step(env._action_stage.raw.clone())  # ``extras`` exists after a step
        assert _public(vars(env)) == INSTANCE_ATTRIBUTES
        # Everything else the env holds is private and declared in env.py's
        # list of attributes (the type annotations on the class) or set by
        # the constructor (its three flags).
        declared = set(BaseEnv.__annotations__) | {
            "_perturbations_enabled",
            "_current_context",
            "_advancing",
        }
        assert {name for name in vars(env) if name.startswith("_")} <= declared
    finally:
        env.close()
