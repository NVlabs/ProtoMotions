// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "protomotions/deployment/input.hpp"
#include "protomotions/deployment/metatron_v1_arm_ik.hpp"

namespace protomotions::deployment::metatron_v1 {

// The frame hand goals are expressed in while the robot stands: the midpoint
// of the two planted feet on the floor, facing the heading the feet were
// planted at. Operator hand offsets are mapped into this frame rather than the
// torso's, so the goal does not drift when the torso leans toward it.
struct SupportFrame {
  std::array<float, 3> center{};  // midpoint of the planted feet, z = floor under them
  float yaw = 0.0F;
  bool valid = false;
};

// Operator-to-robot mapping. Horizontal offsets are read in the operator's
// headset-yaw frame relative to their root and scaled for the robot's reach;
// height comes from the headset tracking's floor frame. The workspace box is a
// safety bound on where a goal may be placed, not a reachability model.
struct TeleopHandMapping {
  float horizontal_scale = 1.0F;
  float height_scale = 1.0F;
  float height_offset = 0.0F;
  // Relative to the feet midpoint. The old 0.65 m limit let a forward reach
  // pull the torso far over the toes on hardware; retain floor/overhead reach
  // while capping only forward extension.
  std::array<float, 2> forward_range{-0.15F, 0.40F};
  std::array<float, 2> lateral_range{-0.65F, 0.65F};
  // Low enough to pick from the floor (a G1 rubber hand is about 4 cm thick);
  // the lower bound exists to keep goals above the floor, not to forbid reaching it.
  std::array<float, 2> height_range{0.05F, 1.55F};
};

inline std::array<float, 3> rotate_yaw_xy(float yaw, const std::array<float, 3>& v) {
  const float c = std::cos(yaw), s = std::sin(yaw);
  return {c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]};
}

// World-frame goal for one hand from the operator's local hand offset
// (x forward, y left, relative to their root, headset-yaw frame) and the hand's
// height above the operator's floor.
// The operator's hand scaled for the robot, as a support-frame offset (forward,
// lateral, height above the floor) before the workspace clamp.
inline std::array<float, 3> hand_offset_from_operator(const std::array<float, 3>& local_offset,
                                                      float height_above_floor,
                                                      const TeleopHandMapping& mapping) {
  return {local_offset[0] * mapping.horizontal_scale, local_offset[1] * mapping.horizontal_scale,
          height_above_floor * mapping.height_scale + mapping.height_offset};
}

inline std::array<float, 3> hand_goal_from_offset(const SupportFrame& support,
                                                  const std::array<float, 3>& offset,
                                                  const TeleopHandMapping& mapping) {
  const float forward = std::clamp(offset[0], mapping.forward_range[0], mapping.forward_range[1]);
  const float lateral = std::clamp(offset[1], mapping.lateral_range[0], mapping.lateral_range[1]);
  const float height = std::clamp(offset[2], mapping.height_range[0], mapping.height_range[1]);
  const auto rotated = rotate_yaw_xy(support.yaw, {forward, lateral, 0.0F});
  return {support.center[0] + rotated[0], support.center[1] + rotated[1], support.center[2] + height};
}

inline std::array<float, 3> hand_goal_from_operator(const SupportFrame& support,
                                                    const std::array<float, 3>& local_offset,
                                                    float height_above_floor,
                                                    const TeleopHandMapping& mapping) {
  return hand_goal_from_offset(support, hand_offset_from_operator(local_offset, height_above_floor, mapping),
                               mapping);
}

// A world position as a support-frame offset: the inverse of hand_goal_from_offset
// without the clamp.
inline std::array<float, 3> support_offset(const SupportFrame& support, const std::array<float, 3>& world) {
  const auto local = rotate_yaw_xy(-support.yaw, {world[0] - support.center[0], world[1] - support.center[1], 0.0F});
  return {local[0], local[1], world[2] - support.center[2]};
}

// Apply the same workspace bound to the final policy prompt, after feedback
// bias or any other goal shaping has been added.
inline std::array<float, 3> clamp_hand_prompt(const SupportFrame& support,
                                              const std::array<float, 3>& world,
                                              const TeleopHandMapping& mapping) {
  return hand_goal_from_offset(support, support_offset(support, world), mapping);
}

inline std::array<float, 2> slew_velocity_xy(const std::array<float, 2>& previous,
                                              const std::array<float, 2>& requested,
                                              float max_delta) {
  const float dx = requested[0] - previous[0], dy = requested[1] - previous[1];
  const float length = std::hypot(dx, dy);
  const float scale = length > max_delta ? max_delta / length : 1.0F;
  return {previous[0] + scale * dx, previous[1] + scale * dy};
}

inline bool stationary_with_hysteresis(bool was_stationary, float speed) {
  return was_stationary ? speed < 0.07F : speed <= 0.03F;
}

// Quaternions as xyzw, the convention of the policy's goal tensors.
inline std::array<float, 4> quat_multiply(const std::array<float, 4>& a, const std::array<float, 4>& b) {
  return {a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
          a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
          a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
          a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]};
}
inline std::array<float, 4> quat_conjugate(const std::array<float, 4>& q) { return {-q[0], -q[1], -q[2], q[3]}; }
inline std::array<float, 4> quat_normalized(std::array<float, 4> q) {
  const float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (!(n > 1.0e-8F)) return {0.0F, 0.0F, 0.0F, 1.0F};
  for (float& v : q) v /= n;
  return q;
}
inline std::array<float, 4> yaw_quat_xyzw(float yaw) { return {0.0F, 0.0F, std::sin(yaw * 0.5F), std::cos(yaw * 0.5F)}; }

// Hand anchoring. The operator and the robot never start in the same pose, so
// an absolute mapping yanks the arms the instant teleop turns on. Anchoring
// instead makes the goal coincide with where each hand already is at that
// moment and lets the operator's motion from then on move it, so the hand-over
// is seamless: nothing moves until the operator does.
//
// Rotation is anchored the same way. A controller and a robot hand have
// unrelated body frames, so the anchor stores, per hand, the rotation that
// takes the controller's world orientation to the hand's measured one. It is
// applied on the right, so a controller turn expressed in the world (turning
// the wrist about the vertical, tilting it forward) turns the hand goal by the
// same world rotation, whatever either body's frame convention is.
struct HandAnchor {
  std::array<std::array<float, 3>, 2> offset{};  // added to the operator's support-frame offset, per hand
  std::array<std::array<float, 4>, 2> rotation_offset{{{0.0F, 0.0F, 0.0F, 1.0F}, {0.0F, 0.0F, 0.0F, 1.0F}}};
  bool set = false;
  bool rotation_set = false;
};

// A controller's orientation in the world: its headset-yaw-frame orientation
// turned to the support frame's heading, the same frame the hand positions use.
inline std::array<float, 4> controller_rotation_world(const SupportFrame& support,
                                                      const std::array<float, 4>& local_xyzw) {
  return quat_normalized(quat_multiply(yaw_quat_xyzw(support.yaw), local_xyzw));
}

inline HandAnchor anchor_hands(const SupportFrame& support, const PicoInputState& pico,
                               const std::array<std::array<float, 3>, 2>& measured_hands_world,
                               const std::array<std::array<float, 4>, 2>& measured_hand_rotations_xyzw,
                               const TeleopHandMapping& mapping) {
  HandAnchor anchor;
  if (!support.valid || !pico.pose_valid) return anchor;
  const std::array<std::array<float, 3>, 2> operator_offsets = {
      hand_offset_from_operator(pico.left_hand_position_local, pico.left_hand_position_world[2], mapping),
      hand_offset_from_operator(pico.right_hand_position_local, pico.right_hand_position_world[2], mapping)};
  const std::array<const std::array<float, 4>*, 2> local_rotations = {&pico.left_hand_rotation_local,
                                                                      &pico.right_hand_rotation_local};
  for (int side = 0; side < 2; ++side) {
    const auto robot = support_offset(support, measured_hands_world[side]);
    for (int i = 0; i < 3; ++i) anchor.offset[side][i] = robot[i] - operator_offsets[side][i];
    if (pico.hand_rotations_valid) {
      const auto controller = controller_rotation_world(support, *local_rotations[side]);
      anchor.rotation_offset[side] = quat_normalized(
          quat_multiply(quat_conjugate(controller), quat_normalized(measured_hand_rotations_xyzw[side])));
    }
  }
  anchor.set = true;
  anchor.rotation_set = pico.hand_rotations_valid;
  return anchor;
}

// Operator height to robot torso height. The operator's crouch is read from
// the headset: the torso target scales the robot's upright height by the
// headset's fraction of the operator's standing height, so a half squat on
// the operator is a half squat on the robot regardless of their sizes.
// Standing height is learned as the running maximum of the headset height
// (people are tallest standing up), or supplied when known.
struct OperatorHeight {
  float standing = 0.0F;  // metres above the operator's floor; 0 = not yet seen
  void observe(float headset_height) {
    if (std::isfinite(headset_height) && headset_height > standing) standing = headset_height;
  }
  bool valid() const { return standing > 0.5F; }
};

inline float torso_height_from_operator(float headset_height, const OperatorHeight& operator_height,
                                        float robot_upright_height) {
  if (!operator_height.valid() || !std::isfinite(headset_height)) return robot_upright_height;
  const float fraction = std::clamp(headset_height / operator_height.standing, 0.0F, 1.0F);
  return robot_upright_height * fraction;
}

// The headset reports heights from its own tracking origin, which is only the
// floor when the headset was set up that way; an eye-level origin puts shoulder
// height near zero, so every hand goal clamps to the floor and the torso follows
// them down. With the operator's standing headset height known, one reading taken
// while they stand upright gives the floor's offset in headset coordinates, and
// this shifts every height so the mapping sees heights above the floor.
inline PicoInputState level_to_floor(PicoInputState pico, float floor_offset) {
  pico.headset_height -= floor_offset;
  pico.left_hand_position_world[2] -= floor_offset;
  pico.right_hand_position_world[2] -= floor_offset;
  return pico;
}

// Prompt pre-emphasis, the web demo's optional "prompt feedback" (Codex's V3
// in the harness study): the policy is prompted with the hand goal pushed a
// little further out in the direction the hand is still short, so the residual
// error is driven down without touching the joints. Proportional on position
// with velocity damping, capped and slewed; no integral, so it cannot wind up.
// Hand error is always measured against the real goal, never the prompt.
struct PromptEmphasis {
  float gain = 1.0F;              // metres of bias per metre of error
  float damping_seconds = 0.15F;  // velocity damping, metres per (m/s)
  float max_bias = 0.12F;         // cap on the bias length
  float slew_mps = 0.1F;          // how fast the bias may move
  float smoothing_seconds = 0.15F;  // first-order lag toward the desired bias
};

inline std::array<float, 3> clamp_length(std::array<float, 3> v, float limit) {
  const float length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (length > limit && length > 0.0F)
    for (float& x : v) x *= limit / length;
  return v;
}

// One update of the bias for one hand. ``velocity`` is the measured hand
// velocity; the caller supplies zero on the first observation (no bias then).
inline std::array<float, 3> prompt_emphasis_step(const std::array<float, 3>& bias,
                                                 const std::array<float, 3>& goal,
                                                 const std::array<float, 3>& measured,
                                                 const std::array<float, 3>& velocity,
                                                 const PromptEmphasis& p, float dt) {
  std::array<float, 3> desired{};
  for (int i = 0; i < 3; ++i)
    desired[i] = p.gain * (goal[i] - measured[i]) - p.damping_seconds * velocity[i];
  desired = clamp_length(desired, p.max_bias);
  const float blend = 1.0F - std::exp(-dt / p.smoothing_seconds);
  std::array<float, 3> change{};
  for (int i = 0; i < 3; ++i) change[i] = (desired[i] - bias[i]) * blend;
  change = clamp_length(change, p.slew_mps * dt);
  return {bias[0] + change[0], bias[1] + change[1], bias[2] + change[2]};
}

// Both hands from a Pico snapshot. A hand without a valid pose is disabled.
// Rotation goals need the anchor: without the per-hand frame offset it holds,
// a controller orientation says nothing about the hand's, so an unanchored
// mapping stays position-only and the policy fills in a natural wrist.
inline std::array<HandGoal, 2> hand_goals_from_pico(const SupportFrame& support,
                                                    const PicoInputState& pico,
                                                    const TeleopHandMapping& mapping,
                                                    const HandAnchor& anchor = {}) {
  std::array<HandGoal, 2> goals{};
  if (!support.valid || !pico.pose_valid) return goals;
  const std::array<const std::array<float, 3>*, 2> local = {&pico.left_hand_position_local,
                                                            &pico.right_hand_position_local};
  const std::array<const std::array<float, 3>*, 2> world = {&pico.left_hand_position_world,
                                                            &pico.right_hand_position_world};
  const std::array<const std::array<float, 4>*, 2> local_rotations = {&pico.left_hand_rotation_local,
                                                                      &pico.right_hand_rotation_local};
  for (int side = 0; side < 2; ++side) {
    HandGoal& goal = goals[side];
    goal.enabled = true;
    goal.translation = true;
    auto offset = hand_offset_from_operator(*local[side], (*world[side])[2], mapping);
    if (anchor.set)
      for (int i = 0; i < 3; ++i) offset[i] += anchor.offset[side][i];
    goal.position = hand_goal_from_offset(support, offset, mapping);
    if (anchor.rotation_set && pico.hand_rotations_valid) {
      goal.rotation = true;
      goal.rotation_xyzw = quat_normalized(quat_multiply(
          controller_rotation_world(support, *local_rotations[side]), anchor.rotation_offset[side]));
    }
  }
  return goals;
}

// Grip is a clutch, not a goal-enable switch: releasing it holds each hand's
// last goal in the feet frame. Re-gripping re-anchors the controller at that
// held goal, so moving the controller while released cannot cause a jump.
class PicoHandClutch final {
 public:
  void reset() noexcept { sides_ = {}; }

  std::array<HandGoal, 2> update(const SupportFrame& support, const PicoInputState& pico,
                                 const TeleopHandMapping& mapping, const HandAnchor& initial_anchor) {
    std::array<HandGoal, 2> goals{};
    if (!support.valid || !pico.pose_valid || !initial_anchor.set) return goals;
    const std::array<const std::array<float, 3>*, 2> local = {&pico.left_hand_position_local,
                                                              &pico.right_hand_position_local};
    const std::array<const std::array<float, 3>*, 2> world = {&pico.left_hand_position_world,
                                                              &pico.right_hand_position_world};
    const std::array<const std::array<float, 4>*, 2> rotations = {&pico.left_hand_rotation_local,
                                                                   &pico.right_hand_rotation_local};
    const std::array<float, 2> grips = {pico.left_grip, pico.right_grip};
    for (int side = 0; side < 2; ++side) {
      Side& state = sides_[side];
      if (!state.ready) {
        const auto neutral = hand_goals_from_pico(support, pico, mapping, initial_anchor)[side];
        state.held_offset = support_offset(support, neutral.position);
        state.rotation = neutral.rotation;
        if (state.rotation)
          state.held_rotation_local = quat_normalized(quat_multiply(
              yaw_quat_xyzw(-support.yaw), neutral.rotation_xyzw));
        state.ready = true;
      }
      const bool engaged = state.engaged ? grips[side] > 0.45F : grips[side] >= 0.55F;
      const auto operator_offset = hand_offset_from_operator(*local[side], (*world[side])[2], mapping);
      if (engaged && !state.engaged) {
        for (int axis = 0; axis < 3; ++axis)
          state.motion_offset[axis] = state.held_offset[axis] - operator_offset[axis];
      }
      if (engaged && state.rotation && pico.hand_rotations_valid && !state.rotation_clutched) {
        const auto held_world = quat_multiply(yaw_quat_xyzw(support.yaw), state.held_rotation_local);
        state.motion_rotation = quat_normalized(quat_multiply(
            quat_conjugate(controller_rotation_world(support, *rotations[side])), held_world));
      }
      HandGoal& goal = goals[side];
      goal.enabled = goal.translation = true;
      if (engaged) {
        auto offset = operator_offset;
        for (int axis = 0; axis < 3; ++axis) offset[axis] += state.motion_offset[axis];
        goal.position = hand_goal_from_offset(support, offset, mapping);
        state.held_offset = support_offset(support, goal.position);
        if (state.rotation && pico.hand_rotations_valid) {
          const auto world_rotation = quat_normalized(quat_multiply(
              controller_rotation_world(support, *rotations[side]), state.motion_rotation));
          state.held_rotation_local = quat_normalized(quat_multiply(
              yaw_quat_xyzw(-support.yaw), world_rotation));
        }
      } else {
        goal.position = hand_goal_from_offset(support, state.held_offset, mapping);
      }
      goal.rotation = state.rotation;
      if (goal.rotation)
        goal.rotation_xyzw = quat_normalized(quat_multiply(yaw_quat_xyzw(support.yaw),
                                                          state.held_rotation_local));
      state.engaged = engaged;
      state.rotation_clutched = engaged && pico.hand_rotations_valid;
    }
    return goals;
  }

 private:
  struct Side {
    bool ready = false;
    bool engaged = false;
    bool rotation = false;
    bool rotation_clutched = false;
    std::array<float, 3> held_offset{};
    std::array<float, 3> motion_offset{};
    std::array<float, 4> held_rotation_local{0.0F, 0.0F, 0.0F, 1.0F};
    std::array<float, 4> motion_rotation{0.0F, 0.0F, 0.0F, 1.0F};
  };
  std::array<Side, 2> sides_{};
};

}  // namespace protomotions::deployment::metatron_v1
