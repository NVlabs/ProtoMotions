# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for BakedAnchorFK vs the reference pose_lib FK implementation.

Real-motion test
----------------
TestRealMotionData uses data/g1-kimodo-generated/proto/output_walk.motion to run
two end-to-end checks on real G1 captured data:

  1. pose_lib FK from (dof_pos + stored root_pos/root_rot) reproduces the stored
     world-frame body positions — validates that the MJCF FK and the motion
     capture pipeline agree.

  2. BakedAnchorFK(dof_pos) produces the same anchor-frame obs as expressing the
     stored body positions in the anchor frame directly — validates the
     zero-root-FK invariance on real data and the full BakedAnchorFK pipeline.

Run with:  pytest protomotions/tests/test_baked_fk.py -v

All synthetic fixtures are constructed without dm_control so these tests
run in CPU-only environments (e.g. CI with requirements_mujoco.txt).

Fixtures
--------
chain_ki      – 5-body linear chain, z-axis hinges, identity ref mats
branching_ki  – 5-body Y-tree, different axis hinges, identity ref mats
rotref_ki     – 5-body Y-tree with non-identity local_rot_ref_mat (90° rotations)
fixed_ki      – 5-body Y-tree where the head (body 4) is a fixed joint
"""

import io
import os
import pytest
import torch
from torch import Tensor

from protomotions.components.pose_lib import (
    KinematicInfo,
    extract_transforms_from_qpos_non_root,
    compute_forward_kinematics_from_transforms,
    matrix_to_quaternion,
)
from protomotions.utils.baked_fk import (
    BakedAnchorFK,
    BakedTargetFK,
    BakedTargetFKNoOdom,
)


# ---------------------------------------------------------------------------
# Reference implementation (formerly compute_fk_trackable_current_obs in
# target_poses.py).  Kept here as a ground-truth for BakedAnchorFK tests.
# Uses the generic pose_lib FK path — correct but without the ONNX / compile
# optimisations that BakedAnchorFK provides.
# ---------------------------------------------------------------------------


def _reference_fk_obs(
    dof_pos: Tensor,
    kinematic_info: KinematicInfo,
    anchor_idx: int,
    trackable_body_indices: Tensor,
) -> Tensor:
    """Reference FK obs via pose_lib (ground-truth for BakedAnchorFK tests)."""
    from protomotions.utils import rotations as rot_utils

    B = dof_pos.shape[0]
    n_track = len(trackable_body_indices)
    device = dof_pos.device
    dtype = dof_pos.dtype

    joint_rot_mats = extract_transforms_from_qpos_non_root(kinematic_info, dof_pos)
    root_pos_zero = torch.zeros(B, 3, device=device, dtype=dtype)
    body_pos, body_rot_mat = compute_forward_kinematics_from_transforms(
        kinematic_info, root_pos_zero, joint_rot_mats
    )

    anchor_rot_mat_inv = body_rot_mat[:, anchor_idx].transpose(-1, -2)
    sel_pos = body_pos[:, trackable_body_indices]
    sel_rot_mat = body_rot_mat[:, trackable_body_indices]

    rel_pos = sel_pos - body_pos[:, anchor_idx].unsqueeze(1)
    pos_in_anchor = torch.matmul(
        anchor_rot_mat_inv.unsqueeze(1), rel_pos.unsqueeze(-1)
    ).squeeze(-1)
    rot_in_anchor = anchor_rot_mat_inv.unsqueeze(1) @ sel_rot_mat
    rot_quat = matrix_to_quaternion(
        rot_in_anchor.reshape(-1, 3, 3), w_last=True
    ).reshape(B, n_track, 4)
    rot_6d = rot_utils.quat_to_tan_norm(rot_quat.reshape(-1, 4), w_last=True).reshape(
        B, n_track, 6
    )

    return torch.cat([pos_in_anchor, rot_6d], dim=-1).reshape(B, -1)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
ATOL = 1e-5
RTOL = 1e-4
B = 8  # default batch size for most tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ki(
    parent_indices,
    local_pos,
    local_rot_ref_mat,
    hinge_axes_map,
    body_names=None,
):
    """Construct a minimal KinematicInfo from raw data."""
    Nb = len(parent_indices)
    n_dofs = sum(len(v) for v in hinge_axes_map.values())
    dof_names = [f"j{i}" for i in range(n_dofs)]
    if body_names is None:
        body_names = [f"body{i}" for i in range(Nb)]
    return KinematicInfo(
        body_names=body_names,
        dof_names=dof_names,
        parent_indices=parent_indices,
        local_pos=torch.tensor(local_pos, dtype=torch.float32),
        local_rot_ref_mat=torch.tensor(local_rot_ref_mat, dtype=torch.float32),
        hinge_axes_map={
            k: torch.tensor(v, dtype=torch.float32) for k, v in hinge_axes_map.items()
        },
        nq=n_dofs + 7,
        nv=n_dofs + 6,
        num_bodies=Nb,
        num_dofs=n_dofs,
        dof_limits_lower=torch.full((n_dofs,), -3.14),
        dof_limits_upper=torch.full((n_dofs,), 3.14),
    )


def _ref_fk(ki, dof_pos):
    """Pose-lib FK with zero root — returns (world_pos, world_rot_mat)."""
    joint_rot_mats = extract_transforms_from_qpos_non_root(ki, dof_pos)
    root_pos = torch.zeros(dof_pos.shape[0], 3, dtype=dof_pos.dtype)
    return compute_forward_kinematics_from_transforms(ki, root_pos, joint_rot_mats)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chain_ki():
    """5-body linear chain: 0→1→2→3→4, all z-axis hinges, identity ref mats."""
    Nb = 5
    I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return _make_ki(
        parent_indices=[-1, 0, 1, 2, 3],
        local_pos=[[0.0, 0.0, 0.0]] + [[1.0, 0.0, 0.0]] * (Nb - 1),
        local_rot_ref_mat=[I3] * Nb,
        hinge_axes_map={i: [[0.0, 0.0, 1.0]] for i in range(1, Nb)},
    )


@pytest.fixture
def branching_ki():
    """5-body Y-tree: root→torso→{left, right, head}. Mixed axes, identity refs."""
    I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return _make_ki(
        parent_indices=[-1, 0, 1, 1, 1],
        local_pos=[
            [0.0, 0.0, 0.0],  # root
            [0.0, 0.0, 0.5],  # torso
            [0.0, 0.3, 0.0],  # left arm
            [0.0, -0.3, 0.0],  # right arm
            [0.3, 0.0, 0.0],  # head
        ],
        local_rot_ref_mat=[I3] * 5,
        hinge_axes_map={
            1: [[0.0, 0.0, 1.0]],  # torso: z-axis
            2: [[1.0, 0.0, 0.0]],  # left:  x-axis
            3: [[1.0, 0.0, 0.0]],  # right: x-axis
            4: [[0.0, 1.0, 0.0]],  # head:  y-axis
        },
        body_names=["root", "torso", "left", "right", "head"],
    )


@pytest.fixture
def rotref_ki():
    """Y-tree with non-identity local_rot_ref_mat (90-degree X and Y rotations).

    Exercises the lrm[i] @ joint_rot path in BakedAnchorFK with non-trivial
    reference rotations, matching what real robots (like G1) have from MJCF.
    """
    import math

    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    # Rx(90): rotate 90° around x
    Rx90 = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    # Ry(90): rotate 90° around y
    Ry90 = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return _make_ki(
        parent_indices=[-1, 0, 1, 1, 1],
        local_pos=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.3, 0.0],
            [0.0, -0.3, 0.0],
            [0.3, 0.0, 0.0],
        ],
        local_rot_ref_mat=[I3, Rx90, Rx90, Ry90, Ry90],
        hinge_axes_map={
            1: [[0.0, 0.0, 1.0]],
            2: [[1.0, 0.0, 0.0]],
            3: [[1.0, 0.0, 0.0]],
            4: [[0.0, 1.0, 0.0]],
        },
        body_names=["root", "torso", "left", "right", "head"],
    )


@pytest.fixture
def fixed_ki():
    """Y-tree where body 4 (head) has no hinge DOF — tests fixed-joint path."""
    I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return _make_ki(
        parent_indices=[-1, 0, 1, 1, 1],
        local_pos=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.3, 0.0],
            [0.0, -0.3, 0.0],
            [0.3, 0.0, 0.0],
        ],
        local_rot_ref_mat=[I3] * 5,
        hinge_axes_map={
            1: [[0.0, 0.0, 1.0]],
            2: [[1.0, 0.0, 0.0]],
            3: [[1.0, 0.0, 0.0]],
            # body 4 has no entry → fixed joint
        },
        body_names=["root", "torso", "left", "right", "head"],
    )


@pytest.fixture
def ndof_ki():
    """4-body chain where body 2 has 2 DOFs (compound hinge).

    Tests the N-DOF branch in BakedAnchorFK.
    """
    I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return _make_ki(
        parent_indices=[-1, 0, 1, 2],
        local_pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        local_rot_ref_mat=[I3] * 4,
        hinge_axes_map={
            1: [[0.0, 0.0, 1.0]],  # body 1: 1-DOF z
            2: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # body 2: 2-DOF (x then y)
            3: [[0.0, 0.0, 1.0]],  # body 3: 1-DOF z
        },
    )


# ---------------------------------------------------------------------------
# Helper: build BakedAnchorFK and reference _reference_fk_obs
# for a given fixture with anchor=torso (body 1) and trackable=[2,3,4].
# ---------------------------------------------------------------------------


def _make_baked_and_ref(ki, anchor_idx=1, trackable=None):
    if trackable is None:
        trackable = torch.tensor([2, 3, 4])
    baked = BakedAnchorFK(ki, anchor_idx=anchor_idx, trackable_body_indices=trackable)
    return baked, anchor_idx, trackable


# ===========================================================================
# BakedTargetFK: reference target poses in the current heading frame
# ===========================================================================


class TestTargetFK:
    """Validate BakedTargetFK's frame math against an independent world-frame
    ground truth, including a NON-root anchor carrying orientation (the g1
    torso_link case), plus a perfect-tracking self-consistency identity against
    BakedAnchorFK.

    Regression guard for the position-path frame fix: the reference
    body-minus-anchor displacement must be rotated into the reference-anchor
    frame (by ref_anchor_inv) before applying offset_quat, mirroring the
    rotation path.  Before the fix, non-root anchors (anchor_idx>0 with a
    twisted anchor) produced position errors of tens of centimetres.
    """

    def _ground_truth_pos(
        self, ki, anchor_idx, trackable, ref_dof, R_cur, R_ref, xy_offset
    ):
        """Brute-force target positions in the current HEADING frame.

        Places each trackable body relative to the reference anchor, expressed
        in the reference-anchor-LOCAL frame (Aref_inv @ (body - anchor)), then
        maps into the current heading frame via offset = H(R_cur)^-1 * R_ref and
        adds the odom XY shift.

        Heading, not the full anchor rotation: the odom shift below is a
        world-horizontal XY vector, and adding it to geometry built with
        quat_conjugate(R_cur) is what the mixed-frame bug did.  Using the same
        rotation for both is the property this oracle exists to pin.
        """
        from protomotions.utils import rotations as rot_utils

        Bn = ref_dof.shape[0]
        nt = trackable.numel()
        ref_pos, ref_rot_mat = _ref_fk(ki, ref_dof)
        anchor_pos = ref_pos[:, anchor_idx]
        aref_inv = ref_rot_mat[:, anchor_idx].transpose(-1, -2)
        rel = ref_pos[:, trackable] - anchor_pos.unsqueeze(1)
        local = torch.matmul(aref_inv.unsqueeze(1), rel.unsqueeze(-1)).squeeze(-1)
        offset = rot_utils.quat_mul(
            rot_utils.calc_heading_quat_inv(R_cur, w_last=True), R_ref, w_last=True
        )
        offset_exp = offset.unsqueeze(1).expand(-1, nt, -1).reshape(-1, 4)
        gt = rot_utils.quat_rotate(
            offset_exp, local.reshape(-1, 3), w_last=True
        ).reshape(Bn, nt, 3)
        xy3 = torch.cat([xy_offset, torch.zeros_like(xy_offset[:, :1])], dim=-1)
        return gt + xy3.unsqueeze(1)

    def _odom_ingredients(
        self, xy_offset, R_cur, num_bodies, anchor_idx, n_future=1, travel=None
    ):
        """Build raw odom ingredients so BakedTargetFK's internal
        ``compute_odom_offset_local`` reproduces ``xy_offset`` as the uniform XY
        shift.

        Sets the believed odometer position to the origin (zero start_xy, zero
        disp, identity start-heading) so the derived offset reduces to
        ``calc_heading_quat_inv(R_cur) @ ref_anchor_xy``.  Choosing
        ``ref_anchor_xy = calc_heading_quat(R_cur) @ xy_offset`` then makes the
        derived offset equal ``xy_offset`` exactly.

        Also returns the two required reference-travel tensors, in ``__call__``
        order.  ``travel`` is the per-step world XY displacement of the reference
        anchor [Bn, n_future, 2]; it defaults to zero so the travel term vanishes
        and the pose/odom ground truths stay valid.
        """
        from protomotions.utils import rotations as rot_utils

        Bn = xy_offset.shape[0]
        heading_fwd = rot_utils.calc_heading_quat(R_cur, w_last=True)
        xy3 = torch.cat([xy_offset, torch.zeros_like(xy_offset[:, :1])], dim=-1)
        ref_anchor_xy = rot_utils.quat_rotate(heading_fwd, xy3, w_last=True)[:, :2]
        ref_rigid_body_pos = torch.zeros(Bn, num_bodies, 3)
        ref_rigid_body_pos[:, anchor_idx, :2] = ref_anchor_xy
        odom_disp_start = torch.zeros(Bn, 2)
        odom_start_xy = torch.zeros(Bn, 2)
        odom_start_heading_inv = torch.zeros(Bn, 4)
        odom_start_heading_inv[:, 3] = 1.0

        current_ref_anchor_pos = ref_rigid_body_pos[:, anchor_idx, :]  # [Bn, 3]
        future_ref_anchor_pos = current_ref_anchor_pos.unsqueeze(1).repeat(
            1, n_future, 1
        )
        if travel is not None:
            future_ref_anchor_pos = future_ref_anchor_pos.clone()
            future_ref_anchor_pos[..., :2] += travel
        # NOTE: the full reference body array is deliberately NOT returned --
        # BakedTargetFK only needs the reference ANCHOR position, which it already
        # gets as current_ref_anchor_pos.
        return (
            odom_disp_start,
            odom_start_xy,
            odom_start_heading_inv,
            future_ref_anchor_pos,
            current_ref_anchor_pos,
        )

    @pytest.mark.parametrize("anchor_idx", [0, 1, 2])
    def test_matches_world_ground_truth(self, rotref_ki, anchor_idx):
        from protomotions.components.pose_lib import matrix_to_quaternion

        ki = rotref_ki
        trackable = torch.tensor([2, 3, 4])
        nt = trackable.numel()
        torch.manual_seed(anchor_idx + 1)
        ref_dof = torch.randn(B, ki.num_dofs)
        cur_dof = torch.randn(B, ki.num_dofs)
        _, cur_rot_mat = _ref_fk(ki, cur_dof)
        _, ref_rot_mat = _ref_fk(ki, ref_dof)
        R_cur = matrix_to_quaternion(cur_rot_mat[:, anchor_idx], w_last=True)
        R_ref = matrix_to_quaternion(ref_rot_mat[:, anchor_idx], w_last=True)
        xy_offset = torch.randn(B, 2)

        gt = self._ground_truth_pos(
            ki, anchor_idx, trackable, ref_dof, R_cur, R_ref, xy_offset
        )
        baked = BakedTargetFK(
            ki, anchor_idx=anchor_idx, trackable_body_indices=trackable
        )
        ingredients = self._odom_ingredients(
            xy_offset, R_cur, ki.num_bodies, anchor_idx
        )
        out = baked(ref_dof.unsqueeze(1), R_cur, R_ref.unsqueeze(1), *ingredients)
        baked_pos = out.reshape(B, 1, nt, 9)[:, 0, :, :3]

        torch.testing.assert_close(baked_pos, gt, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("anchor_idx", [0, 1, 2])
    def test_perfect_tracking_identity(self, rotref_ki, anchor_idx):
        """future==current, R_ref==R_cur, xy=0 => target obs == BakedAnchorFK obs.

        This is the property the shared frame buys: when the robot is exactly on
        the reference, the target obs and the current-state obs are the same
        numbers, so the policy reads tracking error as a plain difference.  Both
        sides must therefore be asked for the same frame -- BakedAnchorFK needs
        the anchor rotation to land in the heading frame the targets use.
        """
        from protomotions.components.pose_lib import matrix_to_quaternion

        ki = rotref_ki
        trackable = torch.tensor([2, 3, 4])
        torch.manual_seed(7 + anchor_idx)
        dof = torch.randn(B, ki.num_dofs)
        _, rot_mat = _ref_fk(ki, dof)
        R = matrix_to_quaternion(rot_mat[:, anchor_idx], w_last=True)

        anchor_obs = BakedAnchorFK(
            ki, anchor_idx=anchor_idx, trackable_body_indices=trackable
        )(dof, R)
        baked = BakedTargetFK(
            ki, anchor_idx=anchor_idx, trackable_body_indices=trackable
        )
        ingredients = self._odom_ingredients(
            torch.zeros(B, 2), R, ki.num_bodies, anchor_idx
        )
        tgt = baked(dof.unsqueeze(1), R, R.unsqueeze(1), *ingredients)

        torch.testing.assert_close(tgt, anchor_obs, atol=ATOL, rtol=RTOL)

    @staticmethod
    def _zyx_quat(yaw: float, pitch: float, roll: float, n: int) -> Tensor:
        """Intrinsic ZYX quaternion, so ``calc_heading`` returns exactly ``yaw``.

        Lets a test change roll and pitch while pinning the heading, which is the
        only way to isolate frame behaviour from a genuine change of facing.
        """
        from protomotions.utils import rotations as rot_utils

        def axis_q(axis: int, ang: float) -> Tensor:
            q = torch.zeros(n, 4)
            q[:, axis] = torch.sin(torch.tensor(ang / 2.0))
            q[:, 3] = torch.cos(torch.tensor(ang / 2.0))
            return q

        q = rot_utils.quat_mul(axis_q(2, yaw), axis_q(1, pitch), w_last=True)
        return rot_utils.quat_mul(q, axis_q(0, roll), w_last=True)

    @pytest.mark.parametrize("anchor_idx", [0, 1, 2])
    def test_target_obs_is_one_frame_under_tilt(self, rotref_ki, anchor_idx):
        """Every channel must move together, which here means not at all.

        The observation is three summands -- body geometry, reference travel, and
        the odometer offset -- added together, so they have to share a frame. They
        did not: geometry used the full anchor rotation while both translations
        used heading only, so once the anchor had any roll or pitch the sum was in
        no frame at all.

        With all three on the heading frame, roll and pitch of the robot cannot
        reach the output: they are not part of the heading, and the reference has
        not moved. So pin the heading, tilt the robot, and require the output to
        be bit-stable. Against the pre-fix code this fails by ~0.4.

        Both the travel term and the odometer offset are non-zero here; with them
        zeroed the mixing has nothing to act on and the test cannot see it.
        """
        from protomotions.components.pose_lib import matrix_to_quaternion

        ki = rotref_ki
        trackable = torch.tensor([2, 3, 4])
        n_future = 3
        torch.manual_seed(11 + anchor_idx)

        future_dof = torch.randn(B, n_future, ki.num_dofs)
        _, ref_rot_mat = _ref_fk(ki, future_dof[:, 0])
        R_ref = matrix_to_quaternion(ref_rot_mat[:, anchor_idx], w_last=True)
        future_rot = R_ref.unsqueeze(1).expand(-1, n_future, -1).contiguous()

        yaw = 0.9
        R_flat = self._zyx_quat(yaw, 0.0, 0.0, B)
        xy_offset = torch.tensor([[0.4, -0.7]]).repeat(B, 1)
        travel = torch.zeros(B, n_future, 2)
        travel[..., 0] = torch.tensor([0.25, 0.5, 1.0])

        baked = BakedTargetFK(
            ki, anchor_idx=anchor_idx, trackable_body_indices=trackable
        )
        # Ingredients depend on the anchor rotation only through its heading, so
        # the same ones stay valid across the whole tilt sweep.
        ingredients = self._odom_ingredients(
            xy_offset,
            R_flat,
            ki.num_bodies,
            anchor_idx,
            n_future=n_future,
            travel=travel,
        )
        flat = baked(future_dof, R_flat, future_rot, *ingredients)

        for roll, pitch in [(0.1, 0.0), (0.0, -0.2), (0.4, -0.3), (-0.35, 0.25)]:
            R_tilt = self._zyx_quat(yaw, pitch, roll, B)
            tilted = baked(future_dof, R_tilt, future_rot, *ingredients)
            torch.testing.assert_close(tilted, flat, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("anchor_idx", [0, 1, 2])
    def test_current_obs_shows_tilt_once_given_the_anchor_rotation(
        self, rotref_ki, anchor_idx
    ):
        """The current-state obs is the one that carries the robot's tilt.

        Without the anchor rotation BakedAnchorFK reports in the anchor's own
        body-fixed frame, where tilting the robot tilts the frame with it and the
        two cancel exactly. Given the rotation it reports in the heading frame,
        where roll and pitch are visible -- which is what makes it comparable to
        the targets, and what lets the policy see that it is leaning.
        """
        ki = rotref_ki
        trackable = torch.tensor([2, 3, 4])
        torch.manual_seed(23 + anchor_idx)
        dof = torch.randn(B, ki.num_dofs)
        baked = BakedAnchorFK(
            ki, anchor_idx=anchor_idx, trackable_body_indices=trackable
        )

        yaw = 0.6
        upright = self._zyx_quat(yaw, 0.0, 0.0, B)
        tilted = self._zyx_quat(yaw, -0.3, 0.25, B)

        body_fixed = baked(dof)
        # Body-fixed: no rotation input, so nothing about the robot's pose enters.
        torch.testing.assert_close(baked(dof), body_fixed, atol=ATOL, rtol=RTOL)

        # Heading frame with a level anchor is the body-fixed frame.
        torch.testing.assert_close(baked(dof, upright), body_fixed, atol=1e-5, rtol=0)

        # Tilt the anchor and the obs moves, by the tilt itself.
        heading_obs = baked(dof, tilted)
        assert (
            (heading_obs - body_fixed).abs().max() > 0.05
        ), "tilt is being cancelled out; the obs cannot be in the heading frame"

    def test_no_odom_matches_zero_odom(self, rotref_ki):
        """BakedTargetFKNoOdom == BakedTargetFK with a zero derived odom offset.

        The odom-free variant simply omits the final XY shift, so it must match
        the odom variant exactly when that shift is zero.  Uses a non-root anchor
        (anchor_idx=1) so the position-path frame rotation is exercised.
        """
        from protomotions.components.pose_lib import matrix_to_quaternion

        ki = rotref_ki
        anchor_idx = 1
        trackable = torch.tensor([2, 3, 4])
        torch.manual_seed(11)
        ref_dof = torch.randn(B, ki.num_dofs)
        cur_dof = torch.randn(B, ki.num_dofs)
        _, cur_rot_mat = _ref_fk(ki, cur_dof)
        _, ref_rot_mat = _ref_fk(ki, ref_dof)
        R_cur = matrix_to_quaternion(cur_rot_mat[:, anchor_idx], w_last=True)
        R_ref = matrix_to_quaternion(ref_rot_mat[:, anchor_idx], w_last=True)

        odom = BakedTargetFK(
            ki, anchor_idx=anchor_idx, trackable_body_indices=trackable
        )
        no_odom = BakedTargetFKNoOdom(
            ki, anchor_idx=anchor_idx, trackable_body_indices=trackable
        )

        ingredients = self._odom_ingredients(
            torch.zeros(B, 2), R_cur, ki.num_bodies, anchor_idx
        )
        out_odom = odom(ref_dof.unsqueeze(1), R_cur, R_ref.unsqueeze(1), *ingredients)
        out_no_odom = no_odom(
            ref_dof.unsqueeze(1), R_cur, R_ref.unsqueeze(1), *ingredients[3:]
        )

        torch.testing.assert_close(out_no_odom, out_odom, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("anchor_idx", [0, 1])
    def test_reference_travel_is_encoded(self, rotref_ki, anchor_idx):
        """Targets must respond to how far the reference travels.

        Regression guard for the bug this term fixes: with the root-at-identity FK,
        every future body is expressed relative to the reference anchor at that same
        step, which cancels the reference's own motion.  Without the travel term a
        reference walking forward and one standing still produce identical targets,
        leaving the policy no way to know how fast to move.
        """
        from protomotions.components.pose_lib import matrix_to_quaternion

        ki = rotref_ki
        trackable = torch.tensor([2, 3, 4])
        n_future = 3
        torch.manual_seed(23 + anchor_idx)
        dof = torch.randn(B, ki.num_dofs)
        _, rot_mat = _ref_fk(ki, dof)
        R = matrix_to_quaternion(rot_mat[:, anchor_idx], w_last=True)

        baked = BakedTargetFK(
            ki, anchor_idx=anchor_idx, trackable_body_indices=trackable
        )
        future_dof = dof.unsqueeze(1).repeat(1, n_future, 1)
        future_rot = R.unsqueeze(1).repeat(1, n_future, 1)
        xy_offset = torch.zeros(B, 2)

        # Same pose and orientation at every step; only the reference's global
        # translation differs between the two calls.
        still = baked(
            future_dof,
            R,
            future_rot,
            *self._odom_ingredients(
                xy_offset, R, ki.num_bodies, anchor_idx, n_future=n_future
            ),
        )
        walk_travel = torch.zeros(B, n_future, 2)
        walk_travel[..., 0] = torch.tensor([0.25, 0.50, 1.00])
        walking = baked(
            future_dof,
            R,
            future_rot,
            *self._odom_ingredients(
                xy_offset,
                R,
                ki.num_bodies,
                anchor_idx,
                n_future=n_future,
                travel=walk_travel,
            ),
        )

        assert (walking - still).abs().max() > 0.1, (
            "BakedTargetFK is invariant to the reference's travel — the "
            "reference-travel term is not being applied"
        )

        # Zero travel must leave the output untouched.
        same = baked(
            future_dof,
            R,
            future_rot,
            *self._odom_ingredients(
                xy_offset,
                R,
                ki.num_bodies,
                anchor_idx,
                n_future=n_future,
                travel=torch.zeros(B, n_future, 2),
            ),
        )
        torch.testing.assert_close(same, still, atol=ATOL, rtol=RTOL)

    def test_future_steps_selects_from_the_published_horizon(self, rotref_ki):
        """``future_steps`` must carve the requested steps out of what the
        control publishes, the same way every other target-pose factory does.

        It used to be dropped on the floor, so a config whose MimicControl
        published 25 steps while its baked-FK observation asked for 4 silently
        emitted all 25.
        """
        ki = rotref_ki
        anchor_idx = 1
        trackable = torch.tensor([2, 3, 4])
        nt = trackable.numel()
        published, requested = 4, [1, 3]
        torch.manual_seed(7)

        future_dof = torch.randn(B, published, ki.num_dofs)
        future_rot = torch.randn(B, published, 4)
        future_rot = future_rot / future_rot.norm(dim=-1, keepdim=True)
        R = torch.zeros(B, 4)
        R[:, 3] = 1.0
        xy_offset = torch.zeros(B, 2)
        travel = torch.zeros(B, published, 2)
        travel[..., 0] = torch.tensor([0.1, 0.2, 0.4, 0.8])

        ingredients = self._odom_ingredients(
            xy_offset,
            R,
            ki.num_bodies,
            anchor_idx,
            n_future=published,
            travel=travel,
        )
        selecting = BakedTargetFK(
            ki,
            anchor_idx=anchor_idx,
            trackable_body_indices=trackable,
            future_steps=requested,
        )
        out = selecting(future_dof, R, future_rot, *ingredients)

        # Only the requested steps come out.
        assert out.shape == (B, len(requested) * nt * 9)

        # And they are the right ones: hand-select the inputs, feed a component
        # that does no selection, and the two must agree exactly. 1-indexed.
        positions = [s - 1 for s in requested]
        odom_disp, odom_xy, odom_head, future_anchor_pos, cur_anchor_pos = ingredients
        passthrough = BakedTargetFK(
            ki, anchor_idx=anchor_idx, trackable_body_indices=trackable
        )
        expected = passthrough(
            future_dof[:, positions],
            R,
            future_rot[:, positions],
            odom_disp,
            odom_xy,
            odom_head,
            future_anchor_pos[:, positions],
            cur_anchor_pos,
        )
        torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)

    def test_requesting_a_step_the_control_never_publishes_raises(self, rotref_ki):
        """Silently emitting the wrong horizon is what shipped; fail instead."""
        ki = rotref_ki
        anchor_idx = 1
        published = 4
        baked = BakedTargetFK(
            ki,
            anchor_idx=anchor_idx,
            trackable_body_indices=torch.tensor([2, 3, 4]),
            future_steps=[1, 2, 4, 8],  # step 8 is past the published horizon
        )
        future_rot = torch.zeros(B, published, 4)
        future_rot[..., 3] = 1.0
        R = torch.zeros(B, 4)
        R[:, 3] = 1.0

        with pytest.raises(ValueError, match="publishes only 4 step"):
            baked(
                torch.zeros(B, published, ki.num_dofs),
                R,
                future_rot,
                *self._odom_ingredients(
                    torch.zeros(B, 2),
                    R,
                    ki.num_bodies,
                    anchor_idx,
                    n_future=published,
                ),
            )


# ===========================================================================
# 1. World-position sanity: BakedAnchorFK internal FK matches pose_lib
# ===========================================================================


class TestWorldPositions:
    """Verify BakedAnchorFK's FK loop produces the same world positions as pose_lib."""

    def _check_world_pos(self, ki):
        """For anchor=root (idx=0): anchor-frame == world frame (zero root)."""
        trackable = torch.arange(ki.num_bodies)
        baked = BakedAnchorFK(ki, anchor_idx=0, trackable_body_indices=trackable)

        torch.manual_seed(42)
        dof_pos = torch.randn(B, ki.num_dofs)

        ref_pos, _ = _ref_fk(ki, dof_pos)
        baked_obs = baked(dof_pos).reshape(B, ki.num_bodies, 9)
        # Positions are the first 3 elements per body
        pos_from_baked = baked_obs[:, :, :3]

        torch.testing.assert_close(pos_from_baked, ref_pos, atol=ATOL, rtol=RTOL)

    def test_chain(self, chain_ki):
        self._check_world_pos(chain_ki)

    def test_branching(self, branching_ki):
        self._check_world_pos(branching_ki)

    def test_rotref(self, rotref_ki):
        self._check_world_pos(rotref_ki)

    def test_fixed_joint(self, fixed_ki):
        """Fixed joint (body 4): world position still correct."""
        trackable = torch.arange(fixed_ki.num_bodies)
        baked = BakedAnchorFK(fixed_ki, anchor_idx=0, trackable_body_indices=trackable)
        dof_pos = torch.randn(B, fixed_ki.num_dofs)
        ref_pos, _ = _ref_fk(fixed_ki, dof_pos)
        pos_from_baked = baked(dof_pos).reshape(B, fixed_ki.num_bodies, 9)[:, :, :3]
        torch.testing.assert_close(pos_from_baked, ref_pos, atol=ATOL, rtol=RTOL)


# ===========================================================================
# 2. Full obs match: BakedAnchorFK vs _reference_fk_obs
# ===========================================================================


class TestMatchesReference:
    """BakedAnchorFK output must be numerically identical to the reference."""

    def _check(self, ki, anchor_idx, trackable, dof_pos):
        baked = BakedAnchorFK(
            ki, anchor_idx=anchor_idx, trackable_body_indices=trackable
        )
        ref = _reference_fk_obs(dof_pos, ki, anchor_idx, trackable)
        out = baked(dof_pos)
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_chain_anchor_mid(self, chain_ki):
        """Chain with anchor at middle body (body 2)."""
        trackable = torch.tensor([0, 1, 3, 4])
        dof_pos = torch.randn(B, chain_ki.num_dofs)
        self._check(chain_ki, anchor_idx=2, trackable=trackable, dof_pos=dof_pos)

    def test_branching_anchor_torso(self, branching_ki):
        """Y-tree with torso anchor, arms + head trackable."""
        trackable = torch.tensor([2, 3, 4])
        dof_pos = torch.randn(B, branching_ki.num_dofs)
        self._check(branching_ki, anchor_idx=1, trackable=trackable, dof_pos=dof_pos)

    def test_rotref_anchor_torso(self, rotref_ki):
        """Non-identity ref mats — exercises lrm @ joint_rot."""
        trackable = torch.tensor([2, 3, 4])
        dof_pos = torch.randn(B, rotref_ki.num_dofs)
        self._check(rotref_ki, anchor_idx=1, trackable=trackable, dof_pos=dof_pos)

    def test_fixed_joint(self, fixed_ki):
        """Fixed head joint — exercises the d_count==0 branch."""
        trackable = torch.tensor([2, 3, 4])
        dof_pos = torch.randn(B, fixed_ki.num_dofs)
        self._check(fixed_ki, anchor_idx=1, trackable=trackable, dof_pos=dof_pos)

    def test_ndof_compound(self, ndof_ki):
        """2-DOF body — exercises the N-DOF compound Rodrigues branch."""
        trackable = torch.tensor([1, 2, 3])
        dof_pos = torch.randn(B, ndof_ki.num_dofs)
        self._check(ndof_ki, anchor_idx=0, trackable=trackable, dof_pos=dof_pos)

    def test_zero_angles(self, branching_ki):
        """All-zero dof_pos: all bodies at their rest poses."""
        trackable = torch.tensor([2, 3, 4])
        dof_pos = torch.zeros(B, branching_ki.num_dofs)
        self._check(branching_ki, anchor_idx=1, trackable=trackable, dof_pos=dof_pos)

    @pytest.mark.parametrize("angle", [0.0, 0.785, 1.571, 3.14, -1.571])
    def test_uniform_angles(self, branching_ki, angle):
        """All DOFs at the same angle — covers sin/cos edge cases."""
        trackable = torch.tensor([2, 3, 4])
        dof_pos = torch.full((1, branching_ki.num_dofs), angle)
        self._check(branching_ki, anchor_idx=1, trackable=trackable, dof_pos=dof_pos)

    @pytest.mark.parametrize("batch_size", [1, 4, 32])
    def test_batch_sizes(self, branching_ki, batch_size):
        trackable = torch.tensor([2, 3, 4])
        dof_pos = torch.randn(batch_size, branching_ki.num_dofs)
        self._check(branching_ki, anchor_idx=1, trackable=trackable, dof_pos=dof_pos)


# ===========================================================================
# 3. Output shape
# ===========================================================================


class TestOutputShape:
    def test_shape(self, branching_ki):
        trackable = torch.tensor([2, 3, 4])
        baked = BakedAnchorFK(
            branching_ki, anchor_idx=1, trackable_body_indices=trackable
        )
        out = baked(torch.randn(5, branching_ki.num_dofs))
        assert out.shape == (5, 3 * 9), f"expected (5, 27), got {out.shape}"

    def test_all_bodies_trackable(self, branching_ki):
        trackable = torch.arange(branching_ki.num_bodies)
        baked = BakedAnchorFK(
            branching_ki, anchor_idx=1, trackable_body_indices=trackable
        )
        out = baked(torch.randn(3, branching_ki.num_dofs))
        assert out.shape == (3, branching_ki.num_bodies * 9)


# ===========================================================================
# 4. Anchor body at origin in its own frame
# ===========================================================================


class TestAnchorBodyZero:
    """The anchor body itself always has (0,0,0) position in the anchor frame."""

    def test_anchor_is_zero(self, branching_ki):
        anchor_idx = 1
        trackable = torch.tensor([0, 1, 2, 3, 4])  # include anchor (body 1)
        baked = BakedAnchorFK(
            branching_ki, anchor_idx=anchor_idx, trackable_body_indices=trackable
        )
        dof_pos = torch.randn(B, branching_ki.num_dofs)
        obs = baked(dof_pos).reshape(B, 5, 9)
        # The anchor body (body 1) is at index 1 in the trackable list
        anchor_pos_in_frame = obs[:, 1, :3]  # should be ~0
        torch.testing.assert_close(
            anchor_pos_in_frame,
            torch.zeros_like(anchor_pos_in_frame),
            atol=1e-5,
            rtol=0.0,
        )


# ===========================================================================
# 5. ONNX trace test
# ===========================================================================


class _BakedFKModule(torch.nn.Module):
    """Thin nn.Module wrapper so torch.onnx.export can call model.modules().

    PyTorch 2.7+ requires the exported object to be an nn.Module or
    ScriptFunction.  BakedAnchorFK is a plain callable; wrapping it here
    keeps the class itself lightweight while satisfying the exporter.
    Note: in production the exporter receives UnifiedPipelineModule (already
    an nn.Module) which calls BakedAnchorFK internally — no wrapper needed.
    """

    def __init__(self, baked: BakedAnchorFK) -> None:
        super().__init__()
        self._baked = baked

    def forward(self, dof_pos: Tensor) -> Tensor:
        return self._baked(dof_pos)


def _export_baked(baked: BakedAnchorFK, sample_input: Tensor) -> bytes:
    """Helper: wrap in nn.Module and export to ONNX bytes."""
    module = _BakedFKModule(baked)
    buf = io.BytesIO()
    torch.onnx.export(
        module,
        (sample_input,),
        buf,
        input_names=["dof_pos"],
        output_names=["fk_obs"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"dof_pos": {0: "batch"}, "fk_obs": {0: "batch"}},
        dynamo=False,
    )
    return buf.getvalue()


class TestONNXTrace:
    """Verify torch.onnx.export succeeds and produces correct results."""

    def test_onnx_export_and_run(self, branching_ki):
        """Export BakedAnchorFK to ONNX and verify output matches PyTorch."""
        onnxruntime = pytest.importorskip("onnxruntime")
        import numpy as np

        trackable = torch.tensor([2, 3, 4])
        baked = BakedAnchorFK(
            branching_ki, anchor_idx=1, trackable_body_indices=trackable
        )

        sample_input = torch.randn(1, branching_ki.num_dofs)
        pt_out = baked(sample_input).detach().numpy()

        onnx_bytes = _export_baked(baked, sample_input)

        sess = onnxruntime.InferenceSession(
            onnx_bytes, providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(["fk_obs"], {"dof_pos": sample_input.numpy()})[0]

        np.testing.assert_allclose(ort_out, pt_out, atol=1e-4, rtol=1e-4)

    def test_onnx_batch_inference(self, branching_ki):
        """ONNX model with dynamic batch axis produces correct outputs for B>1."""
        onnxruntime = pytest.importorskip("onnxruntime")
        import numpy as np

        trackable = torch.tensor([2, 3, 4])
        baked = BakedAnchorFK(
            branching_ki, anchor_idx=1, trackable_body_indices=trackable
        )

        sample_input = torch.randn(1, branching_ki.num_dofs)
        onnx_bytes = _export_baked(baked, sample_input)

        sess = onnxruntime.InferenceSession(
            onnx_bytes, providers=["CPUExecutionProvider"]
        )

        for B_test in [1, 4, 8]:
            batch_input = torch.randn(B_test, branching_ki.num_dofs)
            pt_out = baked(batch_input).detach().numpy()
            ort_out = sess.run(["fk_obs"], {"dof_pos": batch_input.numpy()})[0]
            np.testing.assert_allclose(ort_out, pt_out, atol=1e-4, rtol=1e-4)


class _BakedTargetFKModule(torch.nn.Module):
    """nn.Module wrapper for BakedTargetFK so torch.onnx.export can trace it."""

    def __init__(self, baked) -> None:
        super().__init__()
        self._baked = baked

    def forward(self, *args: Tensor) -> Tensor:
        return self._baked(*args)


class TestONNXTraceTargetFK:
    """BakedTargetFK (and the no-odom variant) must stay ONNX-exportable.

    The target path carries the odometer offset and the reference-travel term on
    top of the FK, both built from quaternion ops.  This is the deploy path, so
    export has to keep working.
    """

    @pytest.mark.parametrize("no_odom", [False, True])
    def test_target_fk_onnx_export_and_run(self, branching_ki, no_odom):
        onnxruntime = pytest.importorskip("onnxruntime")
        import numpy as np

        ki = branching_ki
        anchor_idx = 1
        trackable = torch.tensor([2, 3, 4])
        n_future = 2
        cls = BakedTargetFKNoOdom if no_odom else BakedTargetFK
        baked = cls(ki, anchor_idx=anchor_idx, trackable_body_indices=trackable)

        dof = torch.randn(1, ki.num_dofs)
        future_dof = dof.unsqueeze(1).repeat(1, n_future, 1)
        R = torch.zeros(1, 4)
        R[:, 3] = 1.0
        future_rot = R.unsqueeze(1).repeat(1, n_future, 1)
        current_ref_anchor_pos = torch.randn(1, 3)
        future_ref_anchor_pos = torch.randn(1, n_future, 3)

        if no_odom:
            args = (
                future_dof,
                R,
                future_rot,
                future_ref_anchor_pos,
                current_ref_anchor_pos,
            )
            names = [
                "future_dof_pos",
                "current_anchor_rot",
                "future_anchor_rot",
                "future_ref_anchor_pos",
                "current_ref_anchor_pos",
            ]
        else:
            args = (
                future_dof,
                R,
                future_rot,
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                R.clone(),
                future_ref_anchor_pos,
                current_ref_anchor_pos,
            )
            names = [
                "future_dof_pos",
                "current_anchor_rot",
                "future_anchor_rot",
                "odom_disp_start",
                "odom_start_xy",
                "odom_start_heading_inv",
                "future_ref_anchor_pos",
                "current_ref_anchor_pos",
            ]

        pt_out = baked(*args).detach().numpy()

        buf = io.BytesIO()
        torch.onnx.export(
            _BakedTargetFKModule(baked),
            args,
            buf,
            input_names=names,
            output_names=["target_obs"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        sess = onnxruntime.InferenceSession(
            buf.getvalue(), providers=["CPUExecutionProvider"]
        )
        feed = {n: a.numpy() for n, a in zip(names, args)}
        ort_out = sess.run(["target_obs"], feed)[0]
        np.testing.assert_allclose(ort_out, pt_out, atol=1e-4, rtol=1e-4)


# ===========================================================================
# 6. GPU test
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPU:
    def test_gpu_matches_cpu(self, branching_ki):
        trackable = torch.tensor([2, 3, 4])
        baked = BakedAnchorFK(
            branching_ki, anchor_idx=1, trackable_body_indices=trackable
        )
        dof_pos = torch.randn(8, branching_ki.num_dofs)
        cpu_out = baked(dof_pos)
        cuda_out = baked(dof_pos.cuda())
        torch.testing.assert_close(cpu_out, cuda_out.cpu(), atol=1e-4, rtol=1e-4)


# ===========================================================================
# 7. Picklability (needed for resolved_configs.pt)
# ===========================================================================


class TestPicklability:
    def test_pickle_round_trip(self, branching_ki):
        """BakedAnchorFK survives torch.save / torch.load round-trip."""
        import tempfile
        import os

        trackable = torch.tensor([2, 3, 4])
        baked = BakedAnchorFK(
            branching_ki, anchor_idx=1, trackable_body_indices=trackable
        )
        dof_pos = torch.randn(4, branching_ki.num_dofs)
        out_before = baked(dof_pos)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(baked, path)
            loaded = torch.load(path, weights_only=False)
            out_after = loaded(dof_pos)
        finally:
            os.unlink(path)

        torch.testing.assert_close(out_before, out_after, atol=ATOL, rtol=RTOL)


# ===========================================================================
# 8. G1 integration test (requires dm_control)
# ===========================================================================


class TestG1Integration:
    def test_g1_matches_reference(self):
        pytest.importorskip("dm_control")
        from protomotions.robot_configs.factory import robot_config  # noqa: F401

        cfg = robot_config("g1")
        ki = cfg.kinematic_info
        anchor_idx = cfg.anchor_body_index

        trackable_names = [n for n in cfg.trackable_bodies_subset if n != "head"]
        trackable = torch.tensor([ki.body_names.index(n) for n in trackable_names])

        baked = BakedAnchorFK(
            ki, anchor_idx=anchor_idx, trackable_body_indices=trackable
        )

        torch.manual_seed(123)
        dof_pos = torch.randn(16, ki.num_dofs) * 0.3  # small angles

        ref = _reference_fk_obs(dof_pos, ki, anchor_idx, trackable)
        out = baked(dof_pos)
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


# ===========================================================================
# 9. Real G1 motion data integration test
# ===========================================================================

_WALK_MOTION = "data/g1-kimodo-generated/proto/output_walk.motion"
# FK pos/rot error tolerance between pose_lib and the stored motion data.
# Sub-millimetre (1 mm) is expected if the motion was generated with the same
# MJCF-based FK pipeline.  A larger value would indicate a different pipeline
# was used for capture.
_FK_POS_TOL = 5e-3  # metres
_FK_ROT_TOL = 5e-3  # rotation matrix element error


class TestRealMotionData:
    """End-to-end validation against stored G1 kimodo motion capture data."""

    @pytest.fixture(autouse=True)
    def _require_deps(self):
        """Skip if dm_control or motion file are missing."""
        pytest.importorskip("dm_control")
        if not os.path.exists(_WALK_MOTION):
            pytest.skip(f"Motion file not found: {_WALK_MOTION}")

    @pytest.fixture
    def g1_data(self):
        """Load G1 config and walk motion, return everything needed by the tests."""
        from protomotions.robot_configs.factory import robot_config
        from protomotions.simulator.base_simulator.simulator_state import (
            RobotState,
            StateConversion,
        )

        cfg = robot_config("g1")
        ki = cfg.kinematic_info
        anchor_idx = cfg.anchor_body_index

        raw = torch.load(_WALK_MOTION, weights_only=False)
        rs = RobotState.from_dict(raw, state_conversion=StateConversion.COMMON)

        trackable_names = [n for n in cfg.trackable_bodies_subset if n != "head"]
        trackable = torch.tensor([ki.body_names.index(n) for n in trackable_names])

        return dict(
            ki=ki,
            anchor_idx=anchor_idx,
            trackable=trackable,
            dof_pos=rs.dof_pos,  # [F, 29]
            body_pos=rs.rigid_body_pos,  # [F, 33, 3]  world-frame, COMMON
            body_rot=rs.rigid_body_rot,  # [F, 33, 4]  xyzw, COMMON
        )

    # ------------------------------------------------------------------

    def test_poselib_fk_matches_stored_positions(self, g1_data):
        """pose_lib FK from (dof_pos + stored root) reproduces stored body positions.

        This is a sanity check on the motion file: if the kimodo pipeline used
        the same MJCF-based FK, world-body positions should agree to < 1 mm.
        The test prints the actual error so you can see how well they agree.
        """
        from protomotions.components.pose_lib import (
            extract_transforms_from_qpos_non_root,
            compute_forward_kinematics_from_transforms,
            quaternion_to_matrix,
        )

        ki = g1_data["ki"]
        dof_pos = g1_data["dof_pos"]  # [F, 29]
        body_pos = g1_data["body_pos"]  # [F, 33, 3]
        body_rot = g1_data["body_rot"]  # [F, 33, 4]

        # Build joint_rot_mats using stored joint angles
        joint_rot_mats = extract_transforms_from_qpos_non_root(ki, dof_pos)
        # Plug in the stored root rotation (index 0) so FK is in world frame
        root_rot_mat = quaternion_to_matrix(body_rot[:, 0, :], w_last=True)
        joint_rot_mats[:, 0] = root_rot_mat
        root_pos = body_pos[:, 0, :]  # pelvis world position

        fk_pos, fk_rot_mat = compute_forward_kinematics_from_transforms(
            ki, root_pos, joint_rot_mats
        )  # [F, 33, 3], [F, 33, 3, 3]

        pos_err = (fk_pos - body_pos).norm(dim=-1)  # [F, 33]
        print(
            f"\n  FK pos error (m):  mean={pos_err.mean():.5f}  "
            f"max={pos_err.max():.5f}  "
            f"(tolerance {_FK_POS_TOL:.3f})"
        )

        # Also check rotation matrices (compare as matrices to avoid quat sign)
        stored_rot_mat = quaternion_to_matrix(
            body_rot.reshape(-1, 4), w_last=True
        ).reshape(*body_rot.shape[:2], 3, 3)
        rot_err = (fk_rot_mat - stored_rot_mat).abs()
        print(
            f"  FK rot error (mat): mean={rot_err.mean():.5f}  "
            f"max={rot_err.max():.5f}  "
            f"(tolerance {_FK_ROT_TOL:.3f})"
        )

        assert pos_err.max() < _FK_POS_TOL, (
            f"pose_lib FK position error {pos_err.max():.5f} m exceeds "
            f"{_FK_POS_TOL} m — check that the motion file was generated "
            f"with the same MJCF."
        )
        assert rot_err.max() < _FK_ROT_TOL, (
            f"pose_lib FK rotation error {rot_err.max():.5f} exceeds {_FK_ROT_TOL}"
        )

    def test_baked_fk_obs_matches_stored_anchor_frame(self, g1_data):
        """BakedAnchorFK(dof_pos) matches stored body poses expressed in anchor frame.

        This is the key end-to-end check:
          - LHS: BakedAnchorFK runs FK with zero root and expresses trackable
                 bodies in the anchor frame.
          - RHS: Take stored world-frame body positions/rotations, compute the
                 anchor body's pose from stored data, express trackable bodies
                 relative to that anchor.
        Both should give identical anchor-relative obs because root pose cancels.
        The tolerance here matches the FK accuracy seen in the previous test.
        """
        from protomotions.components.pose_lib import (
            matrix_to_quaternion,
            quaternion_to_matrix,
        )
        from protomotions.utils import rotations as rot_utils

        ki = g1_data["ki"]
        anchor_idx = g1_data["anchor_idx"]
        trackable = g1_data["trackable"]
        dof_pos = g1_data["dof_pos"]
        body_pos = g1_data["body_pos"]
        body_rot = g1_data["body_rot"]

        F = dof_pos.shape[0]
        n_track = len(trackable)

        # --- LHS: BakedAnchorFK from dof_pos only ---
        baked = BakedAnchorFK(
            ki, anchor_idx=anchor_idx, trackable_body_indices=trackable
        )
        baked_obs = baked(dof_pos)  # [F, n_track * 9]

        # --- RHS: express stored world-frame poses in anchor frame ---
        anchor_pos = body_pos[:, anchor_idx, :]  # [F, 3]
        anchor_rot = body_rot[:, anchor_idx, :]  # [F, 4] xyzw
        R_a = quaternion_to_matrix(anchor_rot, w_last=True)  # [F, 3, 3]
        R_a_inv = R_a.transpose(-1, -2)  # [F, 3, 3]

        sel_pos = body_pos[:, trackable, :]  # [F, n_track, 3]
        sel_rot = body_rot[:, trackable, :]  # [F, n_track, 4]
        sel_rot_mat = quaternion_to_matrix(sel_rot.reshape(-1, 4), w_last=True).reshape(
            F, n_track, 3, 3
        )

        # R_a^T @ (p_i - p_a)  →  [F, n_track, 3]
        rel_pos = sel_pos - anchor_pos.unsqueeze(1)
        pos_in_anchor = torch.matmul(
            R_a_inv.unsqueeze(1), rel_pos.unsqueeze(-1)
        ).squeeze(-1)

        # R_a^T @ R_i  →  [F, n_track, 3, 3]
        rot_in_anchor = R_a_inv.unsqueeze(1) @ sel_rot_mat

        rot_quat = matrix_to_quaternion(
            rot_in_anchor.reshape(-1, 3, 3), w_last=True
        ).reshape(F, n_track, 4)
        rot_6d = rot_utils.quat_to_tan_norm(
            rot_quat.reshape(-1, 4), w_last=True
        ).reshape(F, n_track, 6)

        ref_obs = torch.cat([pos_in_anchor, rot_6d], dim=-1).reshape(F, -1)

        diff = (baked_obs - ref_obs).abs()
        print(
            f"\n  BakedAnchorFK vs stored anchor-frame obs:  "
            f"mean={diff.mean():.5f}  max={diff.max():.5f}"
        )

        # The tolerance matches what we allow for FK pos/rot accuracy.
        # If the motion was generated with same-MJCF FK these should be ~1e-5.
        assert diff.max() < _FK_POS_TOL, (
            f"BakedAnchorFK obs differs from stored anchor-frame obs by "
            f"{diff.max():.5f} — exceeds tolerance {_FK_POS_TOL}"
        )
