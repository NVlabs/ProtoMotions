# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pure-tensor utility helpers in protomotions.components.pose_lib.

Avoids real MJCF parsing by constructing minimal `KinematicInfo` instances
with deterministic kinematic chains.
"""

from __future__ import annotations

import logging
import math
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import protomotions.components.pose_lib as pose_lib
from protomotions.components.pose_lib import (
    ControlInfo,
    KinematicInfo,
    build_body_ids_tensor,
    compute_angular_velocity,
    compute_body_density_weights,
    compute_cartesian_velocity,
    compute_forward_kinematics_from_transforms,
    compute_joint_loss_weights,
    compute_joint_rot_mats_from_global_mats,
    compute_kinematics_velocities,
    compute_region_uniform_weights,
    extract_control_info,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    extract_transforms_from_qpos,
    extract_transforms_from_qpos_non_root_ignore_fixed_helper,
    fk_batch_mjcf_with_velocities,
    fk_from_transforms_with_velocities,
)
from protomotions.utils.rotations import (
    exp_map_to_quat,
    quat_from_angle_axis,
    quaternion_to_matrix,
)


def _kin_info(
    parent_indices,
    local_pos,
    *,
    hinge_axes_map=None,
    local_rot_ref_mat=None,
    body_names=None,
    nq=None,
    num_bodies=None,
):
    """Build a minimal duck-typed KinematicInfo for the helpers under test."""
    if num_bodies is None:
        num_bodies = len(parent_indices)
    if hinge_axes_map is None:
        hinge_axes_map = {}
    if local_rot_ref_mat is None:
        local_rot_ref_mat = torch.eye(3).repeat(num_bodies, 1, 1)
    if body_names is None:
        body_names = [f"body_{i}" for i in range(num_bodies)]
    if nq is None:
        nq = 7 + sum(len(axes) for axes in hinge_axes_map.values())
    return SimpleNamespace(
        body_names=body_names,
        parent_indices=parent_indices,
        local_pos=torch.tensor(local_pos, dtype=torch.float32),
        local_rot_ref_mat=local_rot_ref_mat,
        hinge_axes_map=hinge_axes_map,
        nq=nq,
        num_bodies=num_bodies,
    )


# ---------- build_body_ids_tensor ----------------------------------------------


def test_build_body_ids_tensor_returns_indices_in_subset_order():
    all_names = ["root", "head", "left_hand", "right_hand"]
    subset = ["right_hand", "head"]
    ids = build_body_ids_tensor(all_names, subset, device=torch.device("cpu"))
    assert torch.equal(ids, torch.tensor([3, 1], dtype=torch.long))


def test_build_body_ids_tensor_returns_empty_tensor_when_subset_is_none():
    ids = build_body_ids_tensor(["a", "b"], None, device=torch.device("cpu"))
    assert ids.shape == (0,)
    assert ids.dtype == torch.long


def test_build_body_ids_tensor_raises_when_name_missing():
    with pytest.raises(ValueError):
        build_body_ids_tensor(["a", "b"], ["c"], device=torch.device("cpu"))


# ---------- compute_joint_loss_weights -----------------------------------------


def test_compute_joint_loss_weights_root_descendants_dominate_leaves():
    """3-body chain root→mid→leaf with both joints in hinge_axes_map.

    The root-side joint (mid) sees more descendants than the leaf joint, so
    its weight must exceed the leaf's. Weights normalize to sum == num_joints.
    """
    info = _kin_info(
        parent_indices=[-1, 0, 1],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        hinge_axes_map={1: torch.tensor([[0.0, 1.0, 0.0]]), 2: torch.tensor([[0.0, 1.0, 0.0]])},
    )

    weights = compute_joint_loss_weights(info, discount=0.9, min_weight=0.01)

    assert weights.shape == (2,)
    assert weights[0].item() > weights[1].item()
    assert math.isclose(weights.sum().item(), 2.0, rel_tol=1e-5)


def test_compute_joint_loss_weights_min_weight_floors_leaf_contribution():
    """Setting min_weight=0 collapses the leaf joint to its bone-length signal."""
    info = _kin_info(
        parent_indices=[-1, 0, 1],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        hinge_axes_map={1: torch.tensor([[1.0, 0.0, 0.0]]), 2: torch.tensor([[1.0, 0.0, 0.0]])},
    )

    weights = compute_joint_loss_weights(info, discount=0.5, min_weight=0.0)
    # Leaf body has min_weight=0 contribution; its joint weight comes solely
    # from its bone length passed up to itself in the iteration. With
    # min_weight=0 the leaf contributes 0 to its own joint weight slot.
    assert weights[1].item() == 0.0
    # Total still sums to num_joints (2.0) — i.e. normalized to that floor.
    assert math.isclose(weights.sum().item(), 2.0, rel_tol=1e-5)


# ---------- compute_body_density_weights ---------------------------------------


def test_compute_body_density_weights_equal_chain_distances_yield_symmetric_weights():
    """In a 3-body chain with equal bone lengths, the middle body has higher
    chain-density (two close neighbors) so receives lower weight than the ends.
    Endpoints are symmetric and must match exactly."""
    info = _kin_info(
        parent_indices=[-1, 0, 1],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )

    weights = compute_body_density_weights(info, discount=0.9)

    assert weights.shape == (3,)
    assert math.isclose(weights.sum().item(), 3.0, rel_tol=1e-5)
    assert math.isclose(weights[0].item(), weights[2].item(), rel_tol=1e-5)
    assert weights[1].item() < weights[0].item()


def test_compute_body_density_weights_uniform_when_only_one_body():
    info = _kin_info(parent_indices=[-1], local_pos=[[0.0, 0.0, 0.0]])
    weights = compute_body_density_weights(info, discount=0.9)
    assert weights.shape == (1,)
    assert torch.allclose(weights, torch.ones(1))


def test_compute_body_density_weights_discount_one_yields_uniform_weights():
    info = _kin_info(
        parent_indices=[-1, 0, 1, 1],
        local_pos=[
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
    )

    weights = compute_body_density_weights(info, discount=1.0)

    assert torch.allclose(weights, torch.ones(4), atol=1e-6)


# ---------- compute_region_uniform_weights -------------------------------------


def test_compute_region_uniform_weights_chain_collapses_to_single_region():
    """In a single-leaf chain, every body is on the lone leaf's path ⇒
    all assigned to that leaf's region ⇒ uniform weights."""
    info = _kin_info(
        parent_indices=[-1, 0, 1],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )

    weights = compute_region_uniform_weights(info)

    assert weights.shape == (3,)
    assert math.isclose(weights.sum().item(), 3.0, rel_tol=1e-5)
    assert torch.allclose(weights, torch.ones(3), atol=1e-5)


def test_compute_region_uniform_weights_y_shape_separates_core_from_limbs():
    """Y-shape root→spine→{a,b,c} (3 leaves). Bodies 0,1 are on all three
    leaf paths ⇒ assigned to "core". Each leaf gets its own region. So 4
    regions: core(2 bodies), and three single-body regions. Each region
    receives equal total weight, distributed across its members."""
    info = _kin_info(
        parent_indices=[-1, 0, 1, 1, 1],
        local_pos=[[0.0] * 3] * 5,
    )

    weights = compute_region_uniform_weights(info)

    assert weights.shape == (5,)
    assert math.isclose(weights.sum().item(), 5.0, rel_tol=1e-5)
    # Core bodies (0,1) share one region's worth of weight.
    assert math.isclose(weights[0].item(), weights[1].item(), rel_tol=1e-5)
    # Three leaf bodies are symmetric across single-body regions.
    assert math.isclose(weights[2].item(), weights[3].item(), rel_tol=1e-5)
    assert math.isclose(weights[3].item(), weights[4].item(), rel_tol=1e-5)
    # Single-body region weight > shared core weight (per body).
    assert weights[2].item() > weights[0].item()


# ---------- KinematicInfo ------------------------------------------------------


def test_kinematic_info_to_moves_all_tensor_fields_and_hinge_axes_dtype():
    info = KinematicInfo(
        body_names=["root", "joint"],
        dof_names=["hinge"],
        parent_indices=[-1, 0],
        local_pos=torch.zeros(2, 3, dtype=torch.float32),
        local_rot_ref_mat=torch.eye(3, dtype=torch.float32).repeat(2, 1, 1),
        hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)},
        nq=8,
        nv=7,
        num_bodies=2,
        num_dofs=1,
        dof_limits_lower=torch.tensor([-1.0], dtype=torch.float32),
        dof_limits_upper=torch.tensor([1.0], dtype=torch.float32),
    )

    returned = info.to(torch.device("cpu"), dtype=torch.float64)

    assert returned is info
    assert info.local_pos.dtype == torch.float64
    assert info.local_rot_ref_mat.dtype == torch.float64
    assert info.dof_limits_lower.dtype == torch.float64
    assert info.dof_limits_upper.dtype == torch.float64
    assert info.hinge_axes_map[1].dtype == torch.float64


# ---------- MJCF parsing -------------------------------------------------------


def _write_mjcf(tmp_path, xml: str):
    mjcf_path = tmp_path / "tiny.xml"
    mjcf_path.write_text(xml, encoding="utf-8")
    return mjcf_path


def test_extract_kinematic_info_parses_named_tree_defaults_and_degree_limits(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root" pos="0 0 0">
              <freejoint/>
              <body name="hinge_body" pos="1 2 3" quat="0.70710678 0 0 0.70710678">
                <joint name="hinge_z" type="hinge" axis="0 0 1" range="-90 45"/>
              </body>
              <body name="fixed_body" pos="0 1 0"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    info = extract_kinematic_info(str(mjcf_path))

    assert info.body_names == ["root", "hinge_body", "fixed_body"]
    assert info.dof_names == ["hinge_z"]
    assert info.parent_indices == [-1, 0, 0]
    assert info.nq == 8
    assert info.nv == 7
    assert info.num_bodies == 3
    assert info.num_dofs == 1
    assert torch.allclose(info.local_pos[1], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(info.hinge_axes_map[1], torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(
        info.dof_limits_lower,
        torch.tensor([-math.pi / 2]),
        atol=1e-6,
    )
    assert torch.allclose(info.dof_limits_upper, torch.tensor([math.pi / 4]), atol=1e-6)
    assert 2 not in info.hinge_axes_map


def test_extract_kinematic_info_honors_radian_compiler_limits(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body name="hinge_body">
                <joint name="hinge_z" type="hinge" axis="0 0 1" range="-1.5 0.25"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    info = extract_kinematic_info(str(mjcf_path))

    assert torch.allclose(info.dof_limits_lower, torch.tensor([-1.5]))
    assert torch.allclose(info.dof_limits_upper, torch.tensor([0.25]))


def test_extract_kinematic_info_inherits_default_joint_axis_and_range(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <default>
            <joint axis="0 1 0" range="-30 60"/>
          </default>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body name="defaulted_hinge">
                <joint name="hinge_y" type="hinge"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    info = extract_kinematic_info(str(mjcf_path))

    assert torch.allclose(info.hinge_axes_map[1], torch.tensor([[0.0, 1.0, 0.0]]))
    assert torch.allclose(info.dof_limits_lower, torch.tensor([-math.pi / 6]), atol=1e-6)
    assert torch.allclose(info.dof_limits_upper, torch.tensor([math.pi / 3]), atol=1e-6)


def test_extract_kinematic_info_inherits_body_childclass_axis_and_range(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <default>
            <default class="leg">
              <joint axis="1 0 0" range="-20 30"/>
            </default>
          </default>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body name="hip" childclass="leg">
                <joint name="hip_roll" type="hinge"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    info = extract_kinematic_info(str(mjcf_path))

    assert torch.allclose(info.hinge_axes_map[1], torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(info.dof_limits_lower, torch.tensor([-math.pi / 9]), atol=1e-6)
    assert torch.allclose(info.dof_limits_upper, torch.tensor([math.pi / 6]), atol=1e-6)


def test_extract_kinematic_info_joint_attributes_override_body_childclass(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <default>
            <default class="leg">
              <joint axis="1 0 0" range="-90 90"/>
            </default>
          </default>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body name="ankle" childclass="leg">
                <joint name="ankle_pitch" type="hinge" axis="0 1 0" range="-10 20"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    info = extract_kinematic_info(str(mjcf_path))

    assert torch.allclose(info.hinge_axes_map[1], torch.tensor([[0.0, 1.0, 0.0]]))
    assert torch.allclose(info.dof_limits_lower, torch.tensor([-math.pi / 18]), atol=1e-6)
    assert torch.allclose(info.dof_limits_upper, torch.tensor([math.pi / 9]), atol=1e-6)


def test_extract_kinematic_info_joint_class_overrides_body_childclass(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <default>
            <default class="leg">
              <joint axis="1 0 0" range="-90 90"/>
            </default>
            <default class="ankle">
              <joint axis="0 1 0" range="-15 25"/>
            </default>
          </default>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body name="ankle" childclass="leg">
                <joint name="ankle_pitch" type="hinge" class="ankle"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    info = extract_kinematic_info(str(mjcf_path))

    assert torch.allclose(info.hinge_axes_map[1], torch.tensor([[0.0, 1.0, 0.0]]))
    assert torch.allclose(
        info.dof_limits_lower,
        torch.tensor([-math.pi / 12]),
        atol=1e-6,
    )
    assert torch.allclose(
        info.dof_limits_upper,
        torch.tensor([5 * math.pi / 36]),
        atol=1e-6,
    )


def test_extract_kinematic_info_inherits_childclass_through_nested_bodies(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <default>
            <default class="arm">
              <joint axis="0 0 1" range="-45 45"/>
            </default>
          </default>
          <worldbody>
            <body name="root" childclass="arm">
              <freejoint/>
              <body name="shoulder">
                <body name="elbow">
                  <joint name="elbow_yaw" type="hinge"/>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    info = extract_kinematic_info(str(mjcf_path))

    assert info.body_names == ["root", "shoulder", "elbow"]
    assert torch.allclose(info.hinge_axes_map[2], torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(info.dof_limits_lower, torch.tensor([-math.pi / 4]), atol=1e-6)
    assert torch.allclose(info.dof_limits_upper, torch.tensor([math.pi / 4]), atol=1e-6)


def test_extract_kinematic_info_rejects_missing_hinge_axis_without_defaults(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body name="bad_hinge">
                <joint name="missing_axis" type="hinge"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    with pytest.raises(AssertionError, match="missing_axis.*bad_hinge.*no axis"):
        extract_kinematic_info(str(mjcf_path))


def test_extract_kinematic_info_rejects_unnamed_body(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body>
                <joint name="hinge_z" type="hinge" axis="0 0 1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    with pytest.raises(ValueError, match="All bodies need a name"):
        extract_kinematic_info(str(mjcf_path))


def test_extract_kinematic_info_accepts_joint_type_free_root(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <joint name="root_free" type="free"/>
              <body name="hinge_body">
                <joint name="hinge_z" type="hinge" axis="0 0 1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    info = extract_kinematic_info(str(mjcf_path))

    assert info.body_names == ["root", "hinge_body"]
    assert info.dof_names == ["hinge_z"]
    assert info.nq == 8


def test_extract_kinematic_info_handles_root_only_tree_with_empty_dof_limits(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root" pos="1 2 3">
              <freejoint/>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    info = extract_kinematic_info(str(mjcf_path))

    assert info.body_names == ["root"]
    assert info.dof_names == []
    assert info.parent_indices == [-1]
    assert info.hinge_axes_map == {}
    assert info.nq == 7
    assert info.nv == 6
    assert info.num_bodies == 1
    assert info.num_dofs == 0
    assert torch.equal(info.local_pos, torch.tensor([[1.0, 2.0, 3.0]]))
    assert info.dof_limits_lower.shape == (0,)
    assert info.dof_limits_upper.shape == (0,)


def test_extract_kinematic_info_parses_non_root_body_quat_reference(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body name="tilted" quat="0.70710678 0 0 0.70710678">
                <joint name="hinge_z" type="hinge" axis="0 0 1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    info = extract_kinematic_info(str(mjcf_path))

    assert torch.allclose(info.local_rot_ref_mat[0], torch.eye(3), atol=1e-6)
    assert torch.allclose(
        info.local_rot_ref_mat[1],
        _z_rotation_matrix(math.pi / 2),
        atol=1e-6,
    )


def test_extract_kinematic_info_uses_wide_limits_when_hinge_range_is_absent(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body name="unlimited">
                <joint name="hinge_z" type="hinge" axis="0 0 1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    info = extract_kinematic_info(str(mjcf_path))

    assert torch.equal(info.dof_limits_lower, torch.tensor([-1e10]))
    assert torch.equal(info.dof_limits_upper, torch.tensor([1e10]))


def test_extract_kinematic_info_parses_three_hinge_body_in_xml_order(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body name="ball">
                <joint name="ball_x" type="hinge" axis="1 0 0" range="-1.0 1.0"/>
                <joint name="ball_y" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
                <joint name="ball_z" type="hinge" axis="0 0 1" range="-3.0 3.0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    info = extract_kinematic_info(str(mjcf_path))

    assert info.body_names == ["root", "ball"]
    assert info.dof_names == ["ball_x", "ball_y", "ball_z"]
    assert torch.equal(
        info.hinge_axes_map[1],
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    assert torch.equal(info.dof_limits_lower, torch.tensor([-1.0, -2.0, -3.0]))
    assert torch.equal(info.dof_limits_upper, torch.tensor([1.0, 2.0, 3.0]))


def test_extract_kinematic_info_rejects_root_without_free_joint(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <body name="hinge_body">
                <joint name="hinge_z" type="hinge" axis="0 0 1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    with pytest.raises(AssertionError, match="Root body must have a free joint"):
        extract_kinematic_info(str(mjcf_path))


def test_extract_kinematic_info_rejects_non_identity_root_quat(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root" quat="0.70710678 0 0 0.70710678">
              <freejoint/>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    with pytest.raises(AssertionError, match="Root body must have a quat"):
        extract_kinematic_info(str(mjcf_path))


def test_extract_kinematic_info_rejects_multiple_world_roots(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root_a">
              <freejoint/>
            </body>
            <body name="root_b">
              <freejoint/>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    with pytest.raises(AssertionError, match="Multiple root bodies"):
        extract_kinematic_info(str(mjcf_path))


def test_extract_kinematic_info_rejects_two_dof_body(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body name="bad_joint_count">
                <joint name="hinge_x" type="hinge" axis="1 0 0"/>
                <joint name="hinge_y" type="hinge" axis="0 1 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    with pytest.raises(AssertionError, match="expected 1 or 3"):
        extract_kinematic_info(str(mjcf_path))


def test_extract_control_info_reads_joint_values_and_regex_overrides(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body name="arm">
                <joint
                  name="elbow"
                  type="hinge"
                  axis="0 0 1"
                  stiffness="12"
                  damping="3"
                  armature="0.4"
                  frictionloss="0.2"
                  actuatorfrcrange="-7 8"
                />
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    control_info = extract_control_info(
        str(mjcf_path),
        override_control_info={
            "elb.*": ControlInfo(
                stiffness=21.0,
                damping=9.0,
                armature=0.8,
                friction=0.6,
                effort_limit=11.0,
                velocity_limit=13.0,
            )
        },
    )

    elbow = control_info["elbow"]
    assert elbow.stiffness == pytest.approx(21.0)
    assert elbow.damping == pytest.approx(9.0)
    assert elbow.armature == pytest.approx(0.8)
    assert elbow.friction == pytest.approx(0.6)
    assert elbow.effort_limit == pytest.approx(11.0)
    assert elbow.velocity_limit == pytest.approx(13.0)


def test_extract_control_info_wraps_mjcf_parse_errors(tmp_path):
    missing_path = tmp_path / "missing.xml"

    with pytest.raises(ValueError, match="Failed to parse MJCF file"):
        extract_control_info(str(missing_path))


def test_extract_control_info_converts_string_scalars_and_sequence_efforts(monkeypatch):
    joints = [
        SimpleNamespace(type="free", name="root"),
        SimpleNamespace(
            type="hinge",
            name="string_joint",
            stiffness="1.5",
            damping="2.5",
            armature="0.125",
            frictionloss="0.75",
            actuatorfrcrange="-3 4",
        ),
        SimpleNamespace(
            type="hinge",
            name="list_effort_joint",
            actuatorfrcrange=[-5.0, 6.0],
        ),
        SimpleNamespace(
            type="hinge",
            name="array_effort_joint",
            actuatorfrcrange=np.array([-7.0, 8.0]),
        ),
        SimpleNamespace(
            type="hinge",
            name="bad_effort_joint",
            actuatorfrcrange="not-a-range",
        ),
    ]
    fake_model = SimpleNamespace(
        worldbody=SimpleNamespace(body=[SimpleNamespace(joint=joints, body=[])])
    )
    monkeypatch.setattr(pose_lib.mjcf, "from_path", lambda _: fake_model)

    control_info = extract_control_info("fake.xml")

    string_joint = control_info["string_joint"]
    assert string_joint.stiffness == pytest.approx(1.5)
    assert string_joint.damping == pytest.approx(2.5)
    assert string_joint.armature == pytest.approx(0.125)
    assert string_joint.friction == pytest.approx(0.75)
    assert string_joint.effort_limit == pytest.approx(4.0)
    assert control_info["list_effort_joint"].effort_limit == pytest.approx(6.0)
    assert control_info["array_effort_joint"].effort_limit == pytest.approx(8.0)
    assert control_info["bad_effort_joint"].effort_limit is None


def test_extract_control_info_rejects_model_with_only_free_joint(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <freejoint/>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    with pytest.raises(ValueError, match="No DOFs found"):
        extract_control_info(str(mjcf_path))


# ---------- qpos transforms and FK --------------------------------------------


def test_extract_transforms_from_qpos_normalizes_root_and_leaves_fixed_bodies_identity():
    info = _kin_info(
        parent_indices=[-1, 0, 1],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
    )
    qpos = torch.tensor([[1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0, math.pi / 2]])

    root_pos, joint_rot_mats = extract_transforms_from_qpos(info, qpos)

    assert torch.allclose(root_pos, qpos[:, :3])
    assert torch.allclose(joint_rot_mats[0, 0], torch.eye(3), atol=1e-6)
    assert torch.allclose(
        joint_rot_mats[0, 1],
        _z_rotation_matrix(math.pi / 2),
        atol=1e-6,
    )
    assert torch.allclose(joint_rot_mats[0, 2], torch.eye(3), atol=1e-6)


def test_extract_transforms_from_qpos_rejects_non_root_dof_count_mismatch():
    hinge_axes_map = {
        1: torch.tensor([[0.0, 0.0, 1.0]]),
        2: torch.tensor([[1.0, 0.0, 0.0]]),
    }

    with pytest.raises(ValueError, match="DOF count mismatch"):
        extract_transforms_from_qpos_non_root_ignore_fixed_helper(
            hinge_axes_map,
            torch.zeros(1, 1),
        )


def test_extract_transforms_from_qpos_non_root_assertion_reports_expected_dofs():
    info = _kin_info(
        parent_indices=[-1, 0],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
        nq=8,
    )

    with pytest.raises(AssertionError, match="expected 1"):
        extract_transforms_from_qpos(info, torch.zeros(1, 7))


def test_extract_transforms_from_qpos_handles_fixed_only_tree():
    info = _kin_info(
        parent_indices=[-1, 0, 1],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        hinge_axes_map={},
        nq=7,
    )
    qpos = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])

    root_pos, joint_rot_mats = extract_transforms_from_qpos(info, qpos)

    assert torch.allclose(root_pos, qpos[:, :3])
    assert torch.allclose(joint_rot_mats[0, 0], torch.eye(3), atol=1e-6)
    assert torch.allclose(joint_rot_mats[0, 1], torch.eye(3), atol=1e-6)
    assert torch.allclose(joint_rot_mats[0, 2], torch.eye(3), atol=1e-6)


def test_extract_transforms_from_qpos_assigns_non_contiguous_hinge_bodies():
    info = _kin_info(
        parent_indices=[-1, 0, 0, 2, 2],
        local_pos=[[0.0, 0.0, 0.0]] * 5,
        hinge_axes_map={
            2: torch.tensor([[0.0, 0.0, 1.0]]),
            4: torch.tensor([[1.0, 0.0, 0.0]]),
        },
    )
    qpos = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, math.pi / 2, math.pi / 3]]
    )

    _, joint_rot_mats = extract_transforms_from_qpos(info, qpos)

    assert torch.allclose(joint_rot_mats[0, 0], torch.eye(3), atol=1e-6)
    assert torch.allclose(joint_rot_mats[0, 1], torch.eye(3), atol=1e-6)
    assert torch.allclose(joint_rot_mats[0, 2], _z_rotation_matrix(math.pi / 2), atol=1e-6)
    assert torch.allclose(joint_rot_mats[0, 3], torch.eye(3), atol=1e-6)
    expected_x = quaternion_to_matrix(
        quat_from_angle_axis(
            torch.tensor([math.pi / 3]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            w_last=False,
        ),
        w_last=False,
    )[0]
    assert torch.allclose(joint_rot_mats[0, 4], expected_x, atol=1e-6)


def test_extract_qpos_from_transforms_handles_root_only_qpos():
    info = _kin_info(
        parent_indices=[-1, 0],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        hinge_axes_map={},
        nq=7,
    )
    root_pos = torch.tensor([[1.0, -2.0, 0.5]])
    root_rot = _z_rotation_matrix(math.pi / 3)
    joint_rot_mats = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
    joint_rot_mats[:, 0] = root_rot

    qpos = extract_qpos_from_transforms(info, root_pos, joint_rot_mats)
    round_trip_root_pos, round_trip_joint_rot_mats = extract_transforms_from_qpos(
        info, qpos
    )

    assert qpos.shape == (1, 7)
    assert torch.allclose(qpos[:, :3], root_pos)
    assert torch.allclose(round_trip_root_pos, root_pos)
    assert torch.allclose(round_trip_joint_rot_mats[:, 0], root_rot.unsqueeze(0))


def test_extract_qpos_from_transforms_round_trips_single_hinge_qpos():
    info = _kin_info(
        parent_indices=[-1, 0],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
    )
    qpos = torch.tensor([[0.5, -1.0, 2.0, 1.0, 0.0, 0.0, 0.0, math.pi / 4]])

    root_pos, joint_rot_mats = extract_transforms_from_qpos(info, qpos)
    reconstructed = extract_qpos_from_transforms(info, root_pos, joint_rot_mats)

    assert torch.allclose(reconstructed[:, :3], qpos[:, :3])
    assert torch.allclose(reconstructed[:, 3:7], qpos[:, 3:7], atol=1e-6)
    assert torch.allclose(reconstructed[:, 7:], qpos[:, 7:], atol=1e-5)


def test_three_dof_exp_map_qpos_round_trips_through_transforms():
    info = _kin_info(
        parent_indices=[-1, 0],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        hinge_axes_map={
            1: torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        },
        nq=10,
    )
    exp_map = torch.tensor([[0.2, -0.1, 0.3]])
    qpos = torch.cat(
        [
            torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
            exp_map,
        ],
        dim=1,
    )

    root_pos, joint_rot_mats = extract_transforms_from_qpos(
        info,
        qpos,
        qpos_is_exp_map_on_3dof_joints=True,
    )
    reconstructed = extract_qpos_from_transforms(
        info,
        root_pos,
        joint_rot_mats,
        multi_dof_decomposition_method="exp_map",
    )

    expected_joint_rot = quaternion_to_matrix(
        exp_map_to_quat(exp_map, w_last=False),
        w_last=False,
    )
    assert torch.allclose(joint_rot_mats[:, 1], expected_joint_rot, atol=1e-6)
    assert torch.allclose(reconstructed[:, 7:], exp_map, atol=1e-5)


def test_three_dof_independent_hinge_qpos_round_trips_with_euler_decomposition():
    info = _kin_info(
        parent_indices=[-1, 0],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        hinge_axes_map={
            1: torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        },
        nq=10,
    )
    joint_angles = torch.tensor([[0.2, 0.3, -0.4]])
    qpos = torch.cat(
        [
            torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
            joint_angles,
        ],
        dim=1,
    )

    root_pos, joint_rot_mats = extract_transforms_from_qpos(
        info,
        qpos,
        qpos_is_exp_map_on_3dof_joints=False,
    )
    reconstructed = extract_qpos_from_transforms(
        info,
        root_pos,
        joint_rot_mats,
        multi_dof_decomposition_method="euler_xyz",
    )

    assert torch.allclose(reconstructed[:, 7:], joint_angles, atol=1e-5)


def test_extract_qpos_from_transforms_requires_method_for_three_dof_body():
    info = _kin_info(
        parent_indices=[-1, 0],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        body_names=["root", "ball_joint"],
        hinge_axes_map={
            1: torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        },
        nq=10,
    )
    root_pos = torch.zeros(1, 3)
    joint_rot_mats = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)

    with pytest.raises(ValueError, match="ball_joint.*euler_xyz.*exp_map"):
        extract_qpos_from_transforms(info, root_pos, joint_rot_mats)


def test_extract_qpos_from_transforms_rejects_two_dof_body():
    info = _kin_info(
        parent_indices=[-1, 0],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        body_names=["root", "two_dof"],
        hinge_axes_map={
            1: torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            )
        },
        nq=9,
    )
    root_pos = torch.zeros(1, 3)
    joint_rot_mats = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)

    with pytest.raises(ValueError, match="two_dof.*has 2 hinge DOFs"):
        extract_qpos_from_transforms(info, root_pos, joint_rot_mats)


def test_fk_and_global_to_joint_round_trip_with_local_reference_rotation():
    info = _kin_info(
        parent_indices=[-1, 0],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        local_rot_ref_mat=torch.stack(
            [torch.eye(3), _z_rotation_matrix(math.pi / 2)]
        ),
        hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
    )
    root_pos = torch.tensor([[1.0, 2.0, 0.0]])
    joint_rot_mats = torch.stack(
        [torch.eye(3), _z_rotation_matrix(math.pi / 4)]
    ).unsqueeze(0)

    world_pos, world_rot_mats = compute_forward_kinematics_from_transforms(
        info,
        root_pos,
        joint_rot_mats,
    )
    recovered_joint_rot_mats = compute_joint_rot_mats_from_global_mats(
        info,
        world_rot_mats,
    )

    assert torch.allclose(world_pos[0, 0], root_pos[0])
    assert torch.allclose(world_pos[0, 1], torch.tensor([2.0, 2.0, 0.0]), atol=1e-6)
    assert torch.allclose(recovered_joint_rot_mats, joint_rot_mats, atol=1e-6)


def test_fk_from_transforms_with_velocities_returns_common_state_and_optional_velocities():
    info = _kin_info(
        parent_indices=[-1, 0],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
    )
    root_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    joint_rot_mats = torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 2, 1, 1)

    state = fk_from_transforms_with_velocities(
        info,
        root_pos,
        joint_rot_mats,
        fps=10,
        compute_velocities=True,
        velocity_max_horizon=1,
    )

    assert state.rigid_body_pos.shape == (2, 2, 3)
    assert state.rigid_body_rot.shape == (2, 2, 4)
    assert state.fps == 10
    assert torch.allclose(state.rigid_body_pos[:, 0], root_pos)
    assert torch.allclose(state.rigid_body_vel[:, :, 0], torch.full((2, 2), 10.0))
    assert torch.allclose(state.rigid_body_ang_vel, torch.zeros(2, 2, 3), atol=1e-6)


def test_fk_batch_mjcf_with_velocities_skips_velocity_fields_when_disabled():
    info = _kin_info(
        parent_indices=[-1, 0],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
    )
    qpos = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, math.pi / 2]])

    state = fk_batch_mjcf_with_velocities(info, qpos, compute_velocities=False)

    assert state.rigid_body_pos.shape == (1, 2, 3)
    assert state.rigid_body_rot.shape == (1, 2, 4)
    assert state.rigid_body_vel is None
    assert state.rigid_body_ang_vel is None


def test_fk_batch_mjcf_with_velocities_single_frame_does_not_require_fps():
    info = _kin_info(
        parent_indices=[-1, 0],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
    )
    qpos = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    state = fk_batch_mjcf_with_velocities(info, qpos, compute_velocities=True)

    assert state.rigid_body_pos.shape == (1, 2, 3)
    assert state.fps is None
    assert state.rigid_body_vel is None
    assert state.rigid_body_ang_vel is None


def test_fk_batch_velocity_matches_direct_fk_velocity_helpers():
    info = _kin_info(
        parent_indices=[-1, 0],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
    )
    qpos = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, math.pi / 8],
            [0.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, math.pi / 4],
        ]
    )

    state = fk_batch_mjcf_with_velocities(
        info,
        qpos,
        fps=20,
        compute_velocities=True,
        velocity_max_horizon=1,
    )
    root_pos, joint_rot_mats = extract_transforms_from_qpos(info, qpos)
    world_pos, world_rot_mats = compute_forward_kinematics_from_transforms(
        info,
        root_pos,
        joint_rot_mats,
    )
    expected_vel, expected_ang_vel = compute_kinematics_velocities(
        world_pos,
        world_rot_mats,
        fps=20,
        velocity_max_horizon=1,
    )

    assert torch.allclose(state.rigid_body_pos, world_pos)
    assert torch.allclose(state.rigid_body_vel, expected_vel)
    assert torch.allclose(state.rigid_body_ang_vel, expected_ang_vel, atol=1e-5)


def test_compute_kinematics_velocities_matches_independent_articulated_fk_oracle():
    info = _kin_info(
        parent_indices=[-1, 0],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
    )
    fps = 20
    root_delta = 0.10
    joint_delta = 0.05
    frame_ids = torch.arange(4, dtype=torch.float32)
    root_pos = torch.stack(
        [0.25 * frame_ids, -0.10 * frame_ids, torch.zeros_like(frame_ids)],
        dim=1,
    )
    z_axis = torch.tensor([[0.0, 0.0, 1.0]]).expand(frame_ids.shape[0], -1)
    root_quat_wxyz = quat_from_angle_axis(root_delta * frame_ids, z_axis, w_last=False)
    qpos = torch.cat(
        [root_pos, root_quat_wxyz, (joint_delta * frame_ids).unsqueeze(1)],
        dim=1,
    )
    root_pos_from_qpos, joint_rot_mats = extract_transforms_from_qpos(info, qpos)
    world_pos, world_rot_mats = compute_forward_kinematics_from_transforms(
        info,
        root_pos_from_qpos,
        joint_rot_mats,
    )

    lin_vel, ang_vel = compute_kinematics_velocities(
        world_pos,
        world_rot_mats,
        fps=fps,
        velocity_max_horizon=1,
    )

    expected_lin_vel = torch.zeros_like(world_pos)
    expected_lin_vel[:-1] = (world_pos[1:] - world_pos[:-1]) * fps
    expected_lin_vel[-1] = expected_lin_vel[-2]
    expected_ang_vel = torch.zeros_like(ang_vel)
    expected_ang_vel[:, 0, 2] = root_delta * fps
    expected_ang_vel[:, 1, 2] = (root_delta + joint_delta) * fps

    assert torch.allclose(lin_vel, expected_lin_vel, atol=1e-6)
    assert torch.allclose(ang_vel, expected_ang_vel, atol=1e-5)


def test_compute_kinematics_velocities_short_sequence_returns_zero_velocities():
    pos = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    rot_mats = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)

    lin_vel, ang_vel = compute_kinematics_velocities(
        pos,
        rot_mats,
        fps=30,
        velocity_max_horizon=3,
    )

    assert torch.equal(lin_vel, torch.zeros_like(pos))
    assert torch.equal(ang_vel, torch.zeros(1, 2, 3))


# ---------- compute_cartesian_velocity -----------------------------------------


def test_compute_cartesian_velocity_constant_motion_returns_constant_forward_diff():
    """Linear motion at 1 m/frame at fps=30 ⇒ velocity = 30 m/s, with the last
    frame mirroring the previous frame's velocity (no-lookahead carry)."""
    pos = torch.zeros(4, 1, 3)
    pos[:, 0, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])

    vel = compute_cartesian_velocity(pos, fps=30, velocity_max_horizon=1)

    assert vel.shape == (4, 1, 3)
    expected_x = torch.tensor([30.0, 30.0, 30.0, 30.0])
    assert torch.allclose(vel[:, 0, 0], expected_x, atol=1e-4)
    assert torch.allclose(vel[:, 0, 1:], torch.zeros(4, 2), atol=1e-6)


def test_compute_cartesian_velocity_short_sequence_returns_zeros():
    pos = torch.zeros(1, 2, 3)
    vel = compute_cartesian_velocity(pos, fps=30, velocity_max_horizon=1)
    assert torch.equal(vel, torch.zeros(1, 2, 3))


def test_compute_cartesian_velocity_multi_horizon_picks_minimum_magnitude():
    """A noisy single-frame jump that returns to the trend should be filtered
    by the multi-horizon minimum (over 3 frames the average velocity is small,
    over 1 frame the spike dominates)."""
    pos = torch.zeros(5, 1, 3)
    # Trend is +0.1 per frame, but frame 2→3 has a +5.0 spike that's undone
    # next frame (frame 3→4 has -4.9). Multi-horizon minimum should pick the
    # smoother long-horizon estimate at the spike frame.
    pos[:, 0, 0] = torch.tensor([0.0, 0.1, 0.2, 5.2, 0.3])

    vel_h1 = compute_cartesian_velocity(pos, fps=10, velocity_max_horizon=1)
    vel_h3 = compute_cartesian_velocity(pos, fps=10, velocity_max_horizon=3)

    # At the spike frame (index 2), 1-horizon velocity is large (~50 m/s).
    assert vel_h1[2, 0, 0].abs().item() > 40.0
    # 3-horizon picks the smaller magnitude horizon for that frame.
    assert vel_h3[2, 0, 0].abs().item() < vel_h1[2, 0, 0].abs().item()


def test_compute_cartesian_velocity_multi_horizon_longer_than_sequence_uses_simple_diff():
    pos = torch.zeros(2, 1, 3)
    pos[:, 0, 0] = torch.tensor([0.0, 1.25])

    vel = compute_cartesian_velocity(pos, fps=8, velocity_max_horizon=3)

    assert vel.shape == (2, 1, 3)
    assert torch.allclose(vel[:, 0, 0], torch.full((2,), 10.0))
    assert torch.equal(vel[:, 0, 1:], torch.zeros(2, 2))


# ---------- compute_angular_velocity -------------------------------------------


def _z_rotation_matrix(angle: float) -> torch.Tensor:
    """3×3 rotation matrix about +Z axis."""
    return quaternion_to_matrix(
        quat_from_angle_axis(
            torch.tensor([angle]),
            torch.tensor([[0.0, 0.0, 1.0]]),
            w_last=True,
        ),
        w_last=True,
    )[0]


def test_compute_angular_velocity_constant_z_rotation_returns_constant_omega_z():
    """Rotating π/3 per frame about +Z at fps=30 ⇒ ω_z = π/3 * 30 rad/s."""
    angle = math.pi / 3
    rot_mats = torch.stack(
        [
            _z_rotation_matrix(0.0),
            _z_rotation_matrix(angle),
            _z_rotation_matrix(2 * angle),
            _z_rotation_matrix(3 * angle),
        ]
    ).unsqueeze(1)  # (T, Nb=1, 3, 3)

    ang_vel = compute_angular_velocity(rot_mats, fps=30, velocity_max_horizon=1)

    assert ang_vel.shape == (4, 1, 3)
    expected_omega_z = angle * 30
    assert torch.allclose(
        ang_vel[:, 0, 2],
        torch.tensor([expected_omega_z] * 4),
        atol=1e-4,
    )
    # X and Y components should be ~zero for pure z rotation.
    assert torch.allclose(ang_vel[:, 0, :2], torch.zeros(4, 2), atol=1e-4)


def test_compute_angular_velocity_short_sequence_returns_zeros():
    rot_mats = torch.eye(3).reshape(1, 1, 3, 3)
    ang_vel = compute_angular_velocity(rot_mats, fps=30, velocity_max_horizon=1)
    assert torch.equal(ang_vel, torch.zeros(1, 1, 3))


def test_compute_angular_velocity_multi_horizon_short_sequence_uses_valid_delta():
    angle = math.pi / 6
    rot_mats = torch.stack(
        [_z_rotation_matrix(0.0), _z_rotation_matrix(angle)]
    ).unsqueeze(1)

    ang_vel = compute_angular_velocity(rot_mats, fps=12, velocity_max_horizon=3)

    expected_omega_z = angle * 12
    assert torch.allclose(
        ang_vel[:, 0, 2],
        torch.full((2,), expected_omega_z),
        atol=1e-4,
    )


def test_velocity_conventions_align_cartesian_and_angular_forward_differences():
    """Both velocity helpers should store the t->t+h delta at frame t."""
    # Translation: 1 m per frame along +X for 4 frames at fps=10.
    pos = torch.zeros(4, 1, 3)
    pos[:, 0, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    cart_vel = compute_cartesian_velocity(pos, fps=10, velocity_max_horizon=1)

    # Rotation: π/4 per frame about +Z for 4 frames.
    angle = math.pi / 4
    rot_mats = torch.stack(
        [_z_rotation_matrix(i * angle) for i in range(4)]
    ).unsqueeze(1)
    ang_vel = compute_angular_velocity(rot_mats, fps=10, velocity_max_horizon=1)

    expected_cart = 10.0  # 1 m/frame * 10 fps
    expected_ang = angle * 10  # π/4 rad/frame * 10 fps
    assert torch.allclose(cart_vel[:, 0, 0], torch.full((4,), expected_cart))
    assert torch.allclose(
        ang_vel[:, 0, 2],
        torch.full((4,), expected_ang),
        atol=1e-4,
    )


# ---------- pose_lib diagnostic helpers ----------------------------------------


def test_pose_lib_fk_diagnostic_passes_on_tiny_mujoco_model(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="root">
              <freejoint/>
              <geom type="sphere" size="0.03" mass="1"/>
              <body name="limited_hinge" pos="0.4 0 0">
                <joint name="hinge_z" type="hinge" axis="0 0 1" range="-0.5 0.5"/>
                <geom type="sphere" size="0.02" mass="1"/>
                <body name="unlimited_hinge" pos="0.2 0 0">
                  <joint name="hinge_x" type="hinge" axis="1 0 0"/>
                  <geom type="sphere" size="0.02" mass="1"/>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )
    np.random.seed(123)

    assert pose_lib.test_fk_batch(str(mjcf_path), batch_size=2, num_tests=1) is True


def test_pose_lib_fk_diagnostic_returns_false_on_fk_mismatch(tmp_path, monkeypatch):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <freejoint/>
              <geom type="sphere" size="0.03" mass="1"/>
              <body name="hinge_body" pos="0.4 0 0">
                <joint name="hinge_z" type="hinge" axis="0 0 1" range="-45 45"/>
                <geom type="sphere" size="0.02" mass="1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    def wrong_fk(kinematic_info, qpos, **_kwargs):
        batch_size = qpos.shape[0]
        num_bodies = kinematic_info.num_bodies
        quat = torch.zeros(batch_size, num_bodies, 4, device=qpos.device)
        quat[..., 3] = 1.0
        return SimpleNamespace(
            rigid_body_pos=torch.full(
                (batch_size, num_bodies, 3),
                100.0,
                device=qpos.device,
                dtype=qpos.dtype,
            ),
            rigid_body_rot=quat.to(dtype=qpos.dtype),
        )

    monkeypatch.setattr(pose_lib, "fk_batch_mjcf_with_velocities", wrong_fk)
    np.random.seed(123)

    assert pose_lib.test_fk_batch(str(mjcf_path), batch_size=2, num_tests=1) is False


def test_pose_lib_qpos_inverse_diagnostic_passes_on_single_hinge_model(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body name="hinge_body">
                <joint name="hinge_z" type="hinge" axis="0 0 1" range="-1.0 1.0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )
    np.random.seed(456)

    assert (
        pose_lib.test_qpos_transform_inverse(str(mjcf_path), batch_size=2, num_tests=2)
        is True
    )


def test_pose_lib_qpos_inverse_diagnostic_skips_multi_dof_models(tmp_path):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body name="ball">
                <joint name="ball_x" type="hinge" axis="1 0 0"/>
                <joint name="ball_y" type="hinge" axis="0 1 0"/>
                <joint name="ball_z" type="hinge" axis="0 0 1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    assert pose_lib.test_qpos_transform_inverse(str(mjcf_path), batch_size=2) is True


def test_pose_lib_qpos_inverse_diagnostic_returns_false_on_reconstruction_mismatch(
    tmp_path, monkeypatch
):
    mjcf_path = _write_mjcf(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <freejoint/>
              <body name="hinge_body">
                <joint name="hinge_z" type="hinge" axis="0 0 1" range="-45 45"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )

    def wrong_inverse(kinematic_info, root_pos, _joint_rot_mats):
        qpos = torch.zeros(
            root_pos.shape[0],
            kinematic_info.nq,
            device=root_pos.device,
            dtype=root_pos.dtype,
        )
        qpos[:, 3] = 1.0
        return qpos

    monkeypatch.setattr(pose_lib, "extract_qpos_from_transforms", wrong_inverse)
    np.random.seed(456)

    assert (
        pose_lib.test_qpos_transform_inverse(str(mjcf_path), batch_size=2, num_tests=1)
        is False
    )


def _write_default_main_mjcf(tmp_path, xml: str):
    mjcf_path = tmp_path / "protomotions/data/assets/mjcf/h1_2.xml"
    mjcf_path.parent.mkdir(parents=True)
    mjcf_path.write_text(xml, encoding="utf-8")
    return mjcf_path


def test_pose_lib_main_runs_default_mjcf_self_checks(tmp_path, monkeypatch, caplog):
    _write_default_main_mjcf(
        tmp_path,
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="root">
              <freejoint/>
              <geom type="sphere" size="0.03" mass="1"/>
              <body name="hinge_body" pos="0.4 0 0">
                <joint name="hinge_z" type="hinge" axis="0 0 1" range="-0.5 0.5"/>
                <geom type="sphere" size="0.02" mass="1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
    )
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)
    np.random.seed(789)

    runpy.run_path(pose_lib.__file__, run_name="__main__")

    assert any("All tests passed." in record.message for record in caplog.records)


def test_pose_lib_main_logs_invalid_default_mjcf(tmp_path, monkeypatch, caplog):
    _write_default_main_mjcf(tmp_path, "<mujoco>")
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.ERROR)

    runpy.run_path(pose_lib.__file__, run_name="__main__")

    assert any(
        "Configuration error in MJCF or script" in record.message
        or "An unexpected error occurred" in record.message
        for record in caplog.records
    )


def test_pose_lib_main_reports_missing_default_mjcf_when_run_outside_repo(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.ERROR)

    runpy.run_path(pose_lib.__file__, run_name="__main__")

    assert any("MJCF file not found" in record.message for record in caplog.records)
