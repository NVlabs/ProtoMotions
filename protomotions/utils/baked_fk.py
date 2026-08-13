# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-compiled FK → observation callables for current state and reference targets.

Captures all kinematic constants from a KinematicInfo at construction time, so at
inference time only joint angles (plus the IMU anchor rotation) are required.

What is in here
---------------
- ``BakedAnchorFK`` — the robot's own body, from joint angles.
- ``BakedTargetFKNoOdom`` — reference targets in the current heading frame, using
  only the reference motion and the robot's heading.  No odometer.
- ``BakedTargetFK`` — the above plus the odometer offset, so the per-body targets
  also encode how far the robot has drifted from the reference.

The odometer is a layer, not a fixed ingredient.  ``BakedTargetFK`` computes the
odom-free targets first (``targets_in_current_frame``) and then adds the offset,
so anyone who wants reference targets without subtracting a believed position can
use ``BakedTargetFKNoOdom``, or call ``targets_in_current_frame`` directly, and
get the same numbers minus that one term.

Frames
------
All three emit their observations in the **current heading frame**: yaw of the
anchor removed, roll and pitch left in.  One shared frame is what makes the
current-state obs and the target obs directly comparable — under perfect tracking
they are equal, so their difference is the tracking error.  It also matches the
convention in ``protomotions/envs/obs/target_poses.py`` and ``humanoid.py``.

``BakedAnchorFK`` still supports its original body-fixed anchor frame; see the
note on ``current_anchor_rot`` below.

Design goals
------------
- Auto-generates for any robot from KinematicInfo (handles 1-DOF, N-DOF, and
  fixed joints without separate code paths at construction time).
- The resulting __call__ is flat: Python-level conditionals on constant ints are
  fully resolved at ONNX trace time, leaving no conditional ops in the ONNX graph.
- ONNX-safe: uses Python list accumulation for FK (no index_put_ / ScatterElements),
  ending with a single torch.stack (ONNX Concat).
- torch.compile-friendly (dynamic=False): the Python for-loop over range(Nb) with
  all-constant Python-int indices unrolls at compile time.

Why FK can run with a zero root
-------------------------------
Anchor-relative body positions are invariant to root pose:

    anchor_rot_inv @ (body_pos - anchor_pos)
        = local_chain_rot_anchor^{-1} @ local_chain_pos_delta(dof_pos)

Root position and root rotation cancel exactly, so FK runs with root_pos=0 and
root_rot=I and still gives the right anchor-relative answer.  No root position
sensor is ever needed — which is why none of these callables take one.

Expressing the result in the heading frame needs the anchor's world *rotation*
(an IMU quantity), but never its position.  That keeps the whole stack
ONNX-exportable and deployable from IMU plus joint encoders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from torch import Tensor

if TYPE_CHECKING:
    from protomotions.components.pose_lib import KinematicInfo


def _build_skew_sym(axes: Tensor) -> Tensor:
    """Build skew-symmetric matrices for Rodrigues formula.

    Args:
        axes: [N, 3] unit vectors.

    Returns:
        [N, 3, 3] skew-symmetric matrices K where K @ v = axes × v.
    """
    N = axes.shape[0]
    K = torch.zeros(N, 3, 3, dtype=axes.dtype, device=axes.device)
    K[:, 0, 1] = -axes[:, 2]
    K[:, 0, 2] = axes[:, 1]
    K[:, 1, 0] = axes[:, 2]
    K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]
    K[:, 2, 1] = axes[:, 0]
    return K


class BakedAnchorFK:
    """Pre-compiled FK → anchor-frame obs for robots with hinge joints.

    Construction inspects the kinematic tree once and pre-compiles all
    constants.  __call__ receives only dof_pos and runs a flat sequence
    of matrix operations with no runtime type dispatch.

    Supports any robot topology parseable by pose_lib:
    - Fixed joints (body not in hinge_axes_map) → identity rotation
    - 1-DOF hinge  (len(axes) == 1)             → single Rodrigues matrix
    - N-DOF hinge  (len(axes) == N)             → compound of N Rodrigues matrices

    The exp-map 3-DOF format used by MotionLib is NOT needed here because
    the current-state dof_pos from the simulator is always individual hinge
    angles regardless of robot type.

    Picklability: stores plain tensors as instance attributes, compatible with
    torch.save / torch.load(weights_only=False) for resolved_configs.pt.
    """

    def __init__(
        self,
        kinematic_info: "KinematicInfo",
        anchor_idx: int,
        trackable_body_indices: Tensor,
    ) -> None:
        Nb = kinematic_info.num_bodies
        self._Nb: int = Nb
        self._anchor_idx: int = anchor_idx

        # ── Kinematic tree constants ──────────────────────────────────────────
        # Stored as Python lists of ints so loop access is a Python-level
        # operation (not a tensor gather) during ONNX tracing.
        self._parent_indices: List[int] = list(kinematic_info.parent_indices)

        # Per-body local offset and reference rotation (float32, CPU; moved to
        # device on first __call__).
        self._local_pos = kinematic_info.local_pos.float().clone().detach()  # [Nb, 3]
        self._local_rot_ref = (
            kinematic_info.local_rot_ref_mat.float().clone().detach()
        )  # [Nb, 3, 3]
        self._trackable = trackable_body_indices.long().clone()

        # ── Per-body DOF mapping ──────────────────────────────────────────────
        # Two parallel Python lists (indexed by body_idx) so the loop in
        # __call__ uses only Python-int indexing, never tensor indexing.
        body_dof_start: List[int] = [-1] * Nb  # DOF offset in dof_pos (-1 = fixed)
        body_dof_count: List[int] = [0] * Nb  # number of DOFs for this body

        all_axes: List[Tensor] = []
        dof_counter = 0
        for body_idx, axes in kinematic_info.hinge_axes_map.items():
            n = len(axes)
            body_dof_start[body_idx] = dof_counter
            body_dof_count[body_idx] = n
            for k in range(n):
                all_axes.append(axes[k].float().clone().detach())
            dof_counter += n

        self._body_dof_start: List[int] = body_dof_start
        self._body_dof_count: List[int] = body_dof_count
        self._num_dofs: int = dof_counter

        # ── Rodrigues constants: K and K² for every DOF ───────────────────────
        if all_axes:
            axes_t = torch.stack(all_axes)  # [num_dofs, 3]
            K = _build_skew_sym(axes_t)  # [num_dofs, 3, 3]
            self._K = K
            self._K_sq = K @ K  # [num_dofs, 3, 3]
        else:
            self._K = torch.zeros(0, 3, 3)
            self._K_sq = torch.zeros(0, 3, 3)

    # ─────────────────────────────────────────────────────────────────────────

    def __call__(self, dof_pos: Tensor, current_anchor_rot: Tensor = None) -> Tensor:
        """Compute FK obs for the robot's own body from joint angles.

        Args:
            dof_pos: [B, num_dofs] joint angles in pose_lib order.
            current_anchor_rot: [B, 4] world anchor quaternion (w-last), optional.

                Omit it and the obs comes back in the anchor's own body-fixed
                frame: computable from joint encoders alone, and identical no
                matter how the robot is oriented (see the module docstring).

                Pass it and the obs is expressed in the current *heading* frame
                instead -- yaw only, so roll and pitch of the anchor show up in
                the observation rather than being cancelled out.  This is the
                frame ``BakedTargetFK`` puts its targets in, and matching them is
                the point: under perfect tracking the two are then equal, so the
                policy can read tracking error as a difference.  The anchor
                rotation is already an IMU quantity every deployment has.

        Returns:
            [B, n_trackable * 9] obs (pos 3 + 6D rot 6 per body).
        """
        from protomotions.components.pose_lib import matrix_to_quaternion
        from protomotions.utils import rotations as rot_utils

        B = dof_pos.shape[0]
        device = dof_pos.device
        trackable = self._trackable.to(device=device)

        world_pos_t, world_rot_t = self._run_fk_raw(dof_pos)

        # Anchor-relative transform
        anchor_rot_mat = world_rot_t[:, self._anchor_idx]
        anchor_pos = world_pos_t[:, self._anchor_idx]
        R_a_inv = anchor_rot_mat.transpose(-1, -2)

        n_track = len(trackable)
        sel_pos = world_pos_t[:, trackable]
        sel_rot = world_rot_t[:, trackable]

        rel_pos = sel_pos - anchor_pos.unsqueeze(1)
        pos_in_anchor = torch.matmul(
            R_a_inv.unsqueeze(1), rel_pos.unsqueeze(-1)
        ).squeeze(-1)
        rot_in_anchor = R_a_inv.unsqueeze(1) @ sel_rot

        rot_quat = matrix_to_quaternion(
            rot_in_anchor.reshape(-1, 3, 3), w_last=True
        ).reshape(B, n_track, 4)

        if current_anchor_rot is not None:
            # Anchor-local -> heading frame.  A vector in the anchor's own frame
            # reaches the world through R_anchor and the heading frame through
            # H^-1, so one rotation by (H^-1 * R_anchor) does both.  That product
            # is exactly the anchor's roll and pitch, which is what stops being
            # cancelled.  Applied to positions and rotations alike so all nine
            # channels stay in one frame.
            tilt = rot_utils.quat_mul(
                rot_utils.calc_heading_quat_inv(current_anchor_rot, w_last=True),
                current_anchor_rot,
                w_last=True,
            )  # [B, 4]
            tilt_exp = tilt.unsqueeze(1).expand(-1, n_track, -1).reshape(-1, 4)
            pos_in_anchor = rot_utils.quat_rotate(
                tilt_exp, pos_in_anchor.reshape(-1, 3), w_last=True
            ).reshape(B, n_track, 3)
            rot_quat = rot_utils.quat_mul(
                tilt_exp, rot_quat.reshape(-1, 4), w_last=True
            ).reshape(B, n_track, 4)

        rot_6d = rot_utils.quat_to_tan_norm(
            rot_quat.reshape(-1, 4), w_last=True
        ).reshape(B, n_track, 6)

        obs = torch.cat([pos_in_anchor, rot_6d], dim=-1)
        return obs.reshape(B, -1)

    def __repr__(self) -> str:
        return (
            f"BakedAnchorFK(Nb={self._Nb}, num_dofs={self._num_dofs}, "
            f"anchor={self._anchor_idx}, n_trackable={len(self._trackable)})"
        )

    # Allow compute_func.__name__ lookup from MdpComponent.to_dict()
    __name__ = "BakedAnchorFK"

    def _run_fk_raw(self, dof_pos: Tensor):
        """Run FK and return raw world-frame positions and rotation matrices.

        Internal helper for BakedTargetFK.  Does not apply anchor-relative
        transform — returns positions and rotations in the FK root frame
        (root at origin, root rotation = I).

        Args:
            dof_pos: [B, num_dofs] joint angles.

        Returns:
            Tuple of (world_pos [B, Nb, 3], world_rot [B, Nb, 3, 3]).
        """
        B = dof_pos.shape[0]
        device = dof_pos.device
        dtype = dof_pos.dtype

        lp = self._local_pos.to(device=device, dtype=dtype)
        lrm = self._local_rot_ref.to(device=device, dtype=dtype)
        K = self._K.to(device=device, dtype=dtype)
        K_sq = self._K_sq.to(device=device, dtype=dtype)

        I3 = torch.eye(3, device=device, dtype=dtype)

        if K.shape[0] > 0:
            sin_q = torch.sin(dof_pos)
            cos_q = torch.cos(dof_pos)
            hinge_rots = (
                I3[None, None]
                + sin_q[:, :, None, None] * K[None]
                + (1.0 - cos_q[:, :, None, None]) * K_sq[None]
            )
        else:
            hinge_rots = I3.unsqueeze(0).unsqueeze(0).expand(B, 0, 3, 3)

        world_rot: List = [None] * self._Nb
        world_pos: List = [None] * self._Nb
        world_rot[0] = I3.unsqueeze(0).expand(B, -1, -1)
        world_pos[0] = torch.zeros(B, 3, device=device, dtype=dtype)

        for i in range(1, self._Nb):
            p: int = self._parent_indices[i]
            p_rot: Tensor = world_rot[p]
            p_pos: Tensor = world_pos[p]
            d_start: int = self._body_dof_start[i]
            d_count: int = self._body_dof_count[i]

            if d_count == 0:
                eff_local = lrm[i]
            elif d_count == 1:
                eff_local = lrm[i] @ hinge_rots[:, d_start]
            else:
                joint_rot = hinge_rots[:, d_start]
                for k in range(1, d_count):
                    joint_rot = joint_rot @ hinge_rots[:, d_start + k]
                eff_local = lrm[i] @ joint_rot

            world_rot[i] = p_rot @ eff_local
            world_pos[i] = p_pos + (p_rot @ lp[i].unsqueeze(-1)).squeeze(-1)

        return (
            torch.stack(world_pos, dim=1),
            torch.stack(world_rot, dim=1),
        )


class BakedTargetFK:
    """FK on reference dof_pos, expressed in the current robot's heading frame.

    Produces per-body (pos3 + rot6D) target obs directly comparable to
    ``BakedAnchorFK``'s current-state obs, provided that one is given
    ``current_anchor_rot`` so both land in the heading frame.

    This is ``BakedTargetFKNoOdom`` plus the odometer offset.  The odom-free part
    is ``targets_in_current_frame``; everything below about FK, frames and
    reference travel applies to both.

    The anchor-to-anchor alignment uses:
    - **Rotation**: ``current_anchor_rot`` and ``future_anchor_rot`` (from IMU
      and motion file) to rotate reference body poses into the current frame.
    - **Reference travel**: ``future_ref_anchor_pos`` minus
      ``current_ref_anchor_pos`` -- how far the reference itself moves over the
      horizon.  Anchor-relative FK cancels this out, so without it a reference
      walking forward and one standing still look identical to the policy.
    - **Tracking offset**: a heading-local odometer offset derived *internally*
      from ``current_ref_anchor_pos``, ``current_anchor_rot``,
      ``odom_disp_start``, ``odom_start_xy`` and ``odom_start_heading_inv``, via
      ``protomotions.envs.obs.target_poses.compute_odom_offset_local``.  The
      resulting [B, 2] offset shifts target positions so the per-body tracking
      error in the output reflects world-frame error.  Without it, both anchors
      sit at the origin and the policy cannot see global displacement.

    This callable is ONNX-safe.  Every input is a raw sensor or motion quantity;
    the odometer offset is reconstructed inside the graph (plain quaternion math),
    so a deployment supplies the ingredients and never a pre-derived offset.

    Note which reference data is *not* required: the full reference body array.
    Only the reference ANCHOR (position now, position at each future step, and
    orientation) plus the reference joint angles are needed, so a deployment does
    not have to ship every reference body pose.

    Construction
    ------------
    Receives the same ``KinematicInfo`` as ``BakedAnchorFK`` and reuses
    its internal ``_run_fk_raw`` method.  ``future_steps`` selects 1-indexed steps
    out of the horizon ``MimicControl`` publishes; leave it ``None`` to consume
    the published horizon whole.

    Call signature
    --------------
    ``(future_dof_pos, current_anchor_rot, future_anchor_rot, odom_disp_start,
    odom_start_xy, odom_start_heading_inv, future_ref_anchor_pos,
    current_ref_anchor_pos) -> [B, n_selected * n_track * 9]``
    """

    def __init__(
        self,
        kinematic_info: "KinematicInfo",
        anchor_idx: int,
        trackable_body_indices: Tensor,
        future_steps: List[int] = None,
    ) -> None:
        self._inner = BakedAnchorFK(kinematic_info, anchor_idx, trackable_body_indices)
        self._anchor_idx: int = anchor_idx
        self._trackable = trackable_body_indices.long().clone()
        self._future_steps = future_steps

    def targets_in_current_frame(
        self,
        future_dof_pos: Tensor,
        current_anchor_rot: Tensor,
        future_anchor_rot: Tensor,
        future_ref_anchor_pos: Tensor,
        current_ref_anchor_pos: Tensor,
    ):
        """Reference targets in the current heading frame, WITHOUT the odometer.

        This is the primitive the rest of the class is built on, and it is public
        because it is useful on its own.  It answers "what should my body look
        like, and how far is the reference travelling" using nothing but the
        reference motion and the robot's own IMU heading.  It never subtracts the
        robot's believed position, so it needs no odometer and cannot inherit an
        odometer's drift.

        ``BakedTargetFKNoOdom`` is exactly this, packaged as a callable
        observation.  ``BakedTargetFK`` calls this and then adds the odometer
        offset on top, so the odometer is a separable layer rather than something
        baked through the FK math.

        Runs FK on the reference DOF positions and expresses each trackable body
        as a (pos3, rot6D) target in the current heading frame.

        How far the reference moves
        ---------------------------
        FK is run with the root at identity, and each future body is expressed
        relative to the reference anchor at that same future step.  That cancels
        the reference's own motion: a reference walking forward and one standing
        still produce identical targets, so the policy cannot tell how fast it is
        supposed to move.  We add the missing translation back by taking how far
        the reference anchor moves between now and each future step,
        ``(a_ref[s] - a_ref[now]).xy``, and rotating it into the robot's current
        heading frame.

        This comes from the reference motion alone — never from an estimate of
        where the robot is — so it can be computed on the real robot and is safe
        to deploy.  It is also plain quaternion math, so ONNX export and
        ``torch.compile`` keep working.

        Both inputs are required.  They were briefly optional while we A/B-tested
        the term, but a missing binding silently fell back to the broken
        translation-free behaviour, so a mis-wired component now fails loudly.

        One frame throughout
        --------------------
        Body geometry, the travel term, and the odometer shift that
        ``BakedTargetFK`` adds on top are all expressed with the same
        ``calc_heading_quat_inv(current_anchor_rot)``.  They are summed, so they
        have to be: geometry used to use the full anchor rotation while the two
        translations used heading only, which meant the sum was not in any frame
        at all once the anchor had roll or pitch.  Anything added to
        ``pos_in_current`` must use this same rotation.

        Args:
            future_dof_pos: Reference DOF positions [B, n_future, num_dofs].
            current_anchor_rot: Current anchor quaternion [B, 4] (w-last).
            future_anchor_rot: Reference anchor quaternions [B, n_future, 4] (w-last).
            future_ref_anchor_pos: Where the reference anchor will be at each future
                step [B, n_future, 3] (``ctx.mimic.future_anchor_pos``).
            current_ref_anchor_pos: Where the reference anchor is now [B, 3]
                (``ctx.mimic.ref_anchor_pos``).

        Returns:
            Tuple ``(pos_in_current, rot_6d, B, n_future, n_track)`` where
            ``pos_in_current`` is [B*n_future, n_track, 3] (no XY shift applied),
            ``rot_6d`` is [B*n_future, n_track, 6], and ``B``/``n_future``/``n_track``
            are the resolved shape ints (``n_future`` is post ``future_steps``
            selection).
        """
        from protomotions.components.pose_lib import matrix_to_quaternion
        from protomotions.utils import rotations as rot_utils

        B, n_future, D = future_dof_pos.shape
        device = future_dof_pos.device

        if self._future_steps is not None:
            from protomotions.envs.obs.utils import select_step_indices

            # select_step_indices takes 1-indexed step numbers into the horizon
            # MimicControl publishes. Asking for step 8 when the control was
            # configured to publish only 4 steps is a config mismatch, not
            # something to silently clamp or let fail as an index error.
            requested = (
                self._future_steps
                if isinstance(self._future_steps, int)
                else max(self._future_steps)
            )
            if requested > n_future:
                raise ValueError(
                    f"{type(self).__name__} was asked for future step {requested}, "
                    f"but MimicControl publishes only {n_future} step(s). Either "
                    f"widen MimicControlConfig.future_steps, or drop the "
                    f"future_steps argument on this observation if the control "
                    f"already publishes exactly the steps you want."
                )

            future_dof_pos = select_step_indices(future_dof_pos, self._future_steps)
            future_anchor_rot = select_step_indices(
                future_anchor_rot, self._future_steps
            )
            future_ref_anchor_pos = select_step_indices(
                future_ref_anchor_pos, self._future_steps
            )
            n_future = future_dof_pos.shape[1]

        trackable = self._trackable.to(device=device)
        n_track = len(trackable)

        # Run FK on all future steps at once: reshape to [B*n_future, D]
        flat_dof = future_dof_pos.reshape(B * n_future, D)
        flat_pos, flat_rot_mat = self._inner._run_fk_raw(flat_dof)
        # flat_pos: [B*n_future, Nb, 3], flat_rot_mat: [B*n_future, Nb, 3, 3]

        # Extract the reference anchor frame for each future step
        ref_anchor_pos = flat_pos[:, self._anchor_idx, :]  # [B*nf, 3]
        ref_anchor_rot_mat = flat_rot_mat[:, self._anchor_idx, :]  # [B*nf, 3, 3]
        ref_anchor_inv = ref_anchor_rot_mat.transpose(-1, -2)  # [B*nf, 3, 3]

        # Select trackable bodies
        sel_pos = flat_pos[:, trackable, :]  # [B*nf, n_track, 3]
        sel_rot = flat_rot_mat[:, trackable, :]  # [B*nf, n_track, 3, 3]

        # Compute positions relative to reference anchor, expressed in the
        # reference-anchor-LOCAL frame.  The FK runs with root=I, so
        # (sel_pos - ref_anchor_pos) is a displacement in the FK-root frame, not
        # the anchor frame.  We must rotate it by ref_anchor_inv (= A_ref^-1) so
        # that the subsequent offset_quat (= H(R_cur_anchor)^-1 * R_ref_anchor,
        # built from WORLD anchor rotations) maps it correctly into the current
        # heading frame.  This mirrors the rotation path below; omitting it is
        # only correct when the anchor is the kinematic root (A_ref = I), which
        # is NOT the case for robots whose anchor is a non-root body (e.g. g1's
        # torso_link above the waist joints).
        ref_rel_pos = sel_pos - ref_anchor_pos.unsqueeze(1)  # [B*nf, n_track, 3]
        ref_rel_pos = torch.matmul(
            ref_anchor_inv.unsqueeze(1), ref_rel_pos.unsqueeze(-1)
        ).squeeze(-1)  # [B*nf, n_track, 3] in reference-anchor-local frame

        # Compute rotations relative to reference anchor
        ref_rel_rot = ref_anchor_inv.unsqueeze(1) @ sel_rot  # [B*nf, n_track, 3, 3]

        # Transform from the reference-anchor frame into the CURRENT HEADING frame
        # using quaternion operations (compile-safe — no torch.zeros + in-place
        # assignment).  offset_quat = H(R_current_anchor)^-1 * R_ref_anchor.
        #
        # Heading-only, not the full anchor rotation: the two translation terms
        # below (reference travel and the odometer shift) are world-horizontal
        # vectors that only make sense yaw-rotated, and all three summands have to
        # live in one frame to be added.  It also matches the convention in
        # protomotions/envs/obs/target_poses.py and humanoid.py, where current and
        # target observations share a heading frame so their difference is the
        # tracking error.  Pass ``current_anchor_rot`` to ``BakedAnchorFK`` to put
        # the current-state obs in this same frame.
        cur_rot_inv = rot_utils.calc_heading_quat_inv(
            current_anchor_rot, w_last=True
        )  # [B, 4]
        cur_rot_inv_exp = (
            cur_rot_inv.unsqueeze(1).expand(-1, n_future, -1).reshape(B * n_future, 4)
        )
        ref_rot_flat = future_anchor_rot.reshape(B * n_future, 4)
        offset_quat = rot_utils.quat_mul(
            cur_rot_inv_exp, ref_rot_flat, w_last=True
        )  # [B*nf, 4]

        # Rotate positions: quat_rotate(offset_quat, ref_rel_pos) (no xy shift here)
        offset_quat_exp = offset_quat.unsqueeze(1).expand(
            -1, n_track, -1
        )  # [B*nf, n_track, 4]
        pos_in_current = rot_utils.quat_rotate(
            offset_quat_exp.reshape(-1, 4), ref_rel_pos.reshape(-1, 3), w_last=True
        ).reshape(B * n_future, n_track, 3)

        # Rotate rotations: convert ref_rel_rot matrices to quats, apply offset, then to 6D.
        # ref_rel_rot is [B*nf, n_track, 3, 3] — convert to quaternion.
        ref_rel_rot_quat = matrix_to_quaternion(
            ref_rel_rot.reshape(-1, 3, 3), w_last=True
        ).reshape(B * n_future, n_track, 4)
        rot_in_current_quat = rot_utils.quat_mul(
            offset_quat_exp.reshape(-1, 4), ref_rel_rot_quat.reshape(-1, 4), w_last=True
        ).reshape(B * n_future, n_track, 4)
        rot_6d = rot_utils.quat_to_tan_norm(
            rot_in_current_quat.reshape(-1, 4), w_last=True
        ).reshape(B * n_future, n_track, 6)

        # Reference-travel feedforward: add back (a_ref[s] - a_ref[current]).xy,
        # which the anchor-relative FK above cancels out.  Clip-side data only.
        # NOTE: these params are deliberately NOT named ``ref_anchor_pos`` — that
        # name is already a local above (the FK-frame anchor, [B*nf, 3]).
        travel_xy = (
            future_ref_anchor_pos[..., :2] - current_ref_anchor_pos[:, None, :2]
        )  # [B, nf, 2] world XY
        travel_3d = torch.cat(
            [travel_xy, torch.zeros_like(travel_xy[..., :1])], dim=-1
        )  # [B, nf, 3]
        # Same rotation the geometry above used -- reusing it is the point, since
        # a separate one is how the two ended up in different frames.
        travel_local = rot_utils.quat_rotate(
            cur_rot_inv_exp, travel_3d.reshape(B * n_future, 3), w_last=True
        )  # [B*nf, 3]
        pos_in_current = pos_in_current + travel_local.reshape(B * n_future, 1, 3)

        return pos_in_current, rot_6d, B, n_future, n_track

    def __call__(
        self,
        future_dof_pos: Tensor,
        current_anchor_rot: Tensor,
        future_anchor_rot: Tensor,
        odom_disp_start: Tensor,
        odom_start_xy: Tensor,
        odom_start_heading_inv: Tensor,
        future_ref_anchor_pos: Tensor,
        current_ref_anchor_pos: Tensor,
    ) -> Tensor:
        """Compute reference target poses in the current robot's heading frame.

        The only reference data this needs is the anchor pose (position now and at
        each future step, plus orientation) and the reference joint angles — never
        the full reference body array.  Deployment therefore only has to ship the
        reference anchor and joint angles.

        Args:
            future_dof_pos: Reference DOF positions [B, n_future, num_dofs].
            current_anchor_rot: Current anchor quaternion [B, 4] (w-last).
            future_anchor_rot: Reference anchor quaternions [B, n_future, 4] (w-last).
            odom_disp_start: Believed odometer displacement from start [B, 2], in
                the start-heading frame (the stochastic corruption boundary).
            odom_start_xy: Anchor XY at episode start [B, 2].
            odom_start_heading_inv: Inverse start-heading quaternion [B, 4].
            future_ref_anchor_pos: Where the reference anchor will be at each future
                step [B, n_future, 3] (``ctx.mimic.future_anchor_pos``).  Required —
                tells the policy how far the reference moves (see
                ``targets_in_current_frame``).
            current_ref_anchor_pos: Where the reference anchor is now [B, 3]
                (``ctx.mimic.ref_anchor_pos``).  Required — also the reference point
                for the odometer offset below.

        Returns:
            [B, n_selected * n_trackable * 9] target obs in the current heading frame.
        """
        # Lazy import to avoid an import cycle at module load, matching the lazy
        # rotations imports used elsewhere in this module.
        from protomotions.envs.obs.target_poses import compute_odom_offset_local

        pos_in_current, rot_6d, B, n_future, n_track = self.targets_in_current_frame(
            future_dof_pos,
            current_anchor_rot,
            future_anchor_rot,
            future_ref_anchor_pos=future_ref_anchor_pos,
            current_ref_anchor_pos=current_ref_anchor_pos,
        )

        # Derive the heading-local odometer XY offset. The reference anchor position
        # is already an input, so no other reference body pose is needed.
        odom_offset_local = compute_odom_offset_local(
            current_ref_anchor_pos,
            current_anchor_rot,
            odom_disp_start,
            odom_start_xy,
            odom_start_heading_inv,
        )

        # Shift by odometer XY displacement (broadcast across future steps and bodies)
        xy_3d = torch.cat(
            [odom_offset_local, torch.zeros_like(odom_offset_local[:, :1])], dim=-1
        )  # [B, 3]
        xy_shift = (
            xy_3d.unsqueeze(1).expand(-1, n_future, -1).reshape(B * n_future, 1, 3)
        )
        pos_in_current = pos_in_current + xy_shift

        obs = torch.cat([pos_in_current, rot_6d], dim=-1)  # [B*nf, n_track, 9]

        return obs.reshape(B, -1)

    def __repr__(self) -> str:
        inner = self._inner
        return (
            f"BakedTargetFK(Nb={inner._Nb}, num_dofs={inner._num_dofs}, "
            f"anchor={self._anchor_idx}, n_trackable={len(self._trackable)}, "
            f"future_steps={self._future_steps})"
        )

    __name__ = "BakedTargetFK"


class BakedTargetFKNoOdom(BakedTargetFK):
    """Reference targets in the current heading frame, with no odometer.

    This is the odometer-free observation in its own right, not a stripped-down
    special case: it is ``targets_in_current_frame`` packaged as a callable, and
    :class:`BakedTargetFK` is this plus one additive offset.  Reach for it if you
    want the reference targets without ever subtracting a believed position --
    then nothing in the observation can inherit an odometer's drift.

    Produces the same per-body ``(pos3, rot6D)`` reference targets, but **without**
    applying any global XY translation from an odometer.  The output therefore
    carries orientation and joint-relative body positions only — both anchors sit
    at the origin, so the per-body targets do NOT encode the global displacement
    between robot and reference.

    Use this when training a policy that has no odometer observation and must
    instead dead-reckon its global position from local motion + reward shaping
    (e.g. a strong ``anchor_xy`` reward).  Construction and the shared FK math are
    inherited from ``BakedTargetFK`` (same ``KinematicInfo``, ``anchor_idx``,
    ``trackable_body_indices``, ``future_steps``), so it is equally ONNX-safe and
    pickle-safe.

    "No odometer" means the policy is not told where it is relative to the
    reference.  It is still told how far the reference itself moves: the
    ``future_ref_anchor_pos`` / ``current_ref_anchor_pos`` term is read from the
    reference motion alone, so it needs no odometer and no estimate of the robot's
    own position.  It stays required here, and stays safe to deploy.

    Call signature
    --------------
    ``(future_dof_pos, current_anchor_rot, future_anchor_rot, future_ref_anchor_pos,
    current_ref_anchor_pos) -> [B, n_future * n_track * 9]``
    """

    def __call__(
        self,
        future_dof_pos: Tensor,
        current_anchor_rot: Tensor,
        future_anchor_rot: Tensor,
        future_ref_anchor_pos: Tensor,
        current_ref_anchor_pos: Tensor,
    ) -> Tensor:
        """Compute odom-free reference target poses in the current heading frame.

        Args:
            future_dof_pos: Reference DOF positions [B, n_future, num_dofs].
            current_anchor_rot: Current anchor quaternion [B, 4] (w-last).
            future_anchor_rot: Reference anchor quaternions [B, n_future, 4] (w-last).
            future_ref_anchor_pos: Where the reference anchor will be at each future
                step [B, n_future, 3] (``ctx.mimic.future_anchor_pos``).  Required —
                read from the reference motion, no odometer involved.
            current_ref_anchor_pos: Where the reference anchor is now [B, 3]
                (``ctx.mimic.ref_anchor_pos``).  Required.

        Returns:
            [B, n_selected * n_trackable * 9] target obs in the current heading frame,
            with no odometer XY shift applied.
        """
        pos_in_current, rot_6d, B, n_future, n_track = self.targets_in_current_frame(
            future_dof_pos,
            current_anchor_rot,
            future_anchor_rot,
            future_ref_anchor_pos=future_ref_anchor_pos,
            current_ref_anchor_pos=current_ref_anchor_pos,
        )

        obs = torch.cat([pos_in_current, rot_6d], dim=-1)  # [B*nf, n_track, 9]

        return obs.reshape(B, -1)

    def __repr__(self) -> str:
        inner = self._inner
        return (
            f"BakedTargetFKNoOdom(Nb={inner._Nb}, num_dofs={inner._num_dofs}, "
            f"anchor={self._anchor_idx}, n_trackable={len(self._trackable)}, "
            f"future_steps={self._future_steps})"
        )

    __name__ = "BakedTargetFKNoOdom"
