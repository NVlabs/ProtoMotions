// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/metatron_v1_teleop.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace protomotions::deployment;
using namespace protomotions::deployment::metatron_v1;

int main() {
  // Support frame at (2, 1) on a 5 cm floor step, facing +y.
  SupportFrame support;
  support.center = {2.0F, 1.0F, 0.05F};
  support.yaw = 1.5707963F;
  support.valid = true;
  TeleopHandMapping mapping;

  // 1. A hand 0.3 m forward and 0.2 m left of the operator at 1.1 m height
  //    lands forward along +y, left along -x, 1.1 m above the support floor.
  const auto goal = hand_goal_from_operator(support, {0.3F, 0.2F, 0.0F}, 1.1F, mapping);
  assert(std::abs(goal[0] - (2.0F - 0.2F)) < 1e-5F);
  assert(std::abs(goal[1] - (1.0F + 0.3F)) < 1e-5F);
  assert(std::abs(goal[2] - 1.15F) < 1e-5F);

  // 2. The workspace box clamps a goal the operator cannot expect the robot to reach.
  const auto far = hand_goal_from_operator(support, {2.0F, -3.0F, 0.0F}, 2.5F, mapping);
  assert(std::abs(far[1] - (1.0F + mapping.forward_range[1])) < 1e-5F);
  assert(std::abs(far[1] - 1.40F) < 1e-5F);  // 40 cm ahead of the planted feet, not 65 cm.
  assert(std::abs(far[0] - (2.0F - mapping.lateral_range[0])) < 1e-5F);
  assert(std::abs(far[2] - (0.05F + mapping.height_range[1])) < 1e-5F);
  const auto biased = clamp_hand_prompt(support, {2.0F, 1.52F, 0.8F}, mapping);
  assert(std::abs(support_offset(support, biased)[0] - 0.40F) < 1e-5F);

  // 3. From a Pico snapshot: both hands enabled, invalid pose disables both,
  //    and an invalid support frame disables both.
  PicoInputState pico;
  pico.pose_valid = true;
  pico.left_hand_position_local = {0.3F, 0.2F, 0.0F};
  pico.left_hand_position_world = {0.0F, 0.0F, 1.1F};
  pico.right_hand_position_local = {0.3F, -0.2F, 0.0F};
  pico.right_hand_position_world = {0.0F, 0.0F, 0.9F};
  auto goals = hand_goals_from_pico(support, pico, mapping);
  assert(goals[0].enabled && goals[1].enabled && !goals[0].rotation);  // position goals only
  assert(std::abs(goals[0].position[2] - 1.15F) < 1e-5F && std::abs(goals[1].position[2] - 0.95F) < 1e-5F);
  assert(std::abs(goals[1].position[0] - 2.2F) < 1e-5F);  // right hand: lateral -0.2 -> +x when facing +y
  pico.pose_valid = false;
  goals = hand_goals_from_pico(support, pico, mapping);
  assert(!goals[0].enabled && !goals[1].enabled);
  pico.pose_valid = true;
  support.valid = false;
  goals = hand_goals_from_pico(support, pico, mapping);
  assert(!goals[0].enabled && !goals[1].enabled);
  support.valid = true;
  // Anchoring: at the hand-over the goals equal the robot's measured hands, wherever the
  // operator holds theirs; afterwards the operator's motion moves the goal by the same amount.
  const std::array<std::array<float, 3>, 2> robot_hands = {{{1.8F, 1.1F, 0.8F}, {2.2F, 1.1F, 0.8F}}};
  // Robot hands measured 90 degrees about z; controllers report an unrelated frame (identity).
  constexpr float kPi = 3.14159265358979F;
  const std::array<float, 4> z90 = yaw_quat_xyzw(kPi / 2);
  const std::array<std::array<float, 4>, 2> robot_rotations = {z90, z90};
  const HandAnchor unrotated = anchor_hands(support, pico, robot_hands, robot_rotations, mapping);
  assert(unrotated.set && !unrotated.rotation_set);  // no valid controller rotations: positions only
  goals = hand_goals_from_pico(support, pico, mapping, unrotated);
  assert(!goals[0].rotation);
  for (int side = 0; side < 2; ++side)
    for (int i = 0; i < 3; ++i) assert(std::abs(goals[side].position[i] - robot_hands[side][i]) < 1e-5F);
  pico.hand_rotations_valid = true;
  const HandAnchor anchor = anchor_hands(support, pico, robot_hands, robot_rotations, mapping);
  assert(anchor.set && anchor.rotation_set);
  goals = hand_goals_from_pico(support, pico, mapping, anchor);
  assert(goals[0].rotation);
  for (int i = 0; i < 4; ++i) assert(std::abs(goals[0].rotation_xyzw[i] - z90[i]) < 1e-5F);  // holds the hand's frame
  pico.left_hand_position_local[0] += 0.1F;  // operator reaches 10 cm forward (+y in this support frame)
  pico.left_hand_position_world[2] += 0.2F;  // and 20 cm up
  pico.left_hand_rotation_local = yaw_quat_xyzw(kPi / 4);  // and turns the wrist 45 degrees about the vertical
  goals = hand_goals_from_pico(support, pico, mapping, anchor);
  assert(std::abs(goals[0].position[1] - 1.2F) < 1e-5F && std::abs(goals[0].position[2] - 1.0F) < 1e-5F);
  assert(std::abs(goals[0].position[0] - 1.8F) < 1e-5F);
  const auto expected = yaw_quat_xyzw(kPi / 2 + kPi / 4);  // the hand goal turns by the same world rotation
  for (int i = 0; i < 4; ++i) assert(std::abs(goals[0].rotation_xyzw[i] - expected[i]) < 1e-5F);
  pico.left_hand_position_local[0] += 1.0F;
  goals = hand_goals_from_pico(support, pico, mapping, anchor);
  assert(std::abs(support_offset(support, goals[0].position)[0] - 0.40F) < 1e-5F);
  assert(!anchor_hands(support, PicoInputState{}, robot_hands, robot_rotations, mapping).set);  // no pose, no anchor

  // Grip clutch: the hand holds relative to the feet while released, then
  // re-grips at that held pose without snapping to the moved controller.
  PicoInputState clutch_pico{};
  clutch_pico.pose_valid = clutch_pico.hand_rotations_valid = true;
  clutch_pico.left_hand_position_local = {0.3F, 0.2F, 0.0F};
  clutch_pico.right_hand_position_local = {0.3F, -0.2F, 0.0F};
  clutch_pico.left_hand_position_world[2] = 1.1F;
  clutch_pico.right_hand_position_world[2] = 0.9F;
  const auto clutch_anchor = anchor_hands(support, clutch_pico, robot_hands, robot_rotations, mapping);
  PicoHandClutch clutch;
  auto held = clutch.update(support, clutch_pico, mapping, clutch_anchor);
  assert(std::abs(held[0].position[1] - robot_hands[0][1]) < 1e-5F);
  clutch_pico.left_hand_position_local[0] += 0.2F;
  clutch_pico.right_hand_position_local[0] += 0.2F;
  held = clutch.update(support, clutch_pico, mapping, clutch_anchor);
  assert(std::abs(held[0].position[1] - robot_hands[0][1]) < 1e-5F);
  assert(std::abs(held[1].position[1] - robot_hands[1][1]) < 1e-5F);
  clutch_pico.left_grip = 1.0F;
  held = clutch.update(support, clutch_pico, mapping, clutch_anchor);
  assert(std::abs(held[0].position[1] - robot_hands[0][1]) < 1e-5F);  // no re-grip jump
  clutch_pico.left_hand_position_local[0] += 0.1F;
  clutch_pico.left_hand_rotation_local = yaw_quat_xyzw(kPi / 4);
  held = clutch.update(support, clutch_pico, mapping, clutch_anchor);
  assert(std::abs(held[0].position[1] - (robot_hands[0][1] + 0.1F)) < 1e-5F);
  assert(std::abs(held[1].position[1] - robot_hands[1][1]) < 1e-5F);  // independent sides
  const auto moving_rotation = held[0].rotation_xyzw;
  clutch_pico.left_grip = 0.0F;
  clutch_pico.left_hand_position_local[0] += 0.3F;
  clutch_pico.left_hand_rotation_local = yaw_quat_xyzw(kPi / 2);
  held = clutch.update(support, clutch_pico, mapping, clutch_anchor);
  assert(std::abs(held[0].position[1] - (robot_hands[0][1] + 0.1F)) < 1e-5F);
  for (int i = 0; i < 4; ++i)
    assert(std::abs(held[0].rotation_xyzw[i] - moving_rotation[i]) < 1e-5F);
  clutch_pico.left_grip = 1.0F;
  held = clutch.update(support, clutch_pico, mapping, clutch_anchor);
  assert(std::abs(held[0].position[1] - (robot_hands[0][1] + 0.1F)) < 1e-5F);
  support.center[1] += 0.1F;
  held = clutch.update(support, clutch_pico, mapping, clutch_anchor);
  assert(std::abs(held[0].position[1] - (robot_hands[0][1] + 0.2F)) < 1e-5F);

  const auto stepped = slew_velocity_xy({0.0F, 0.0F}, {1.0F, 0.0F}, 0.04F);
  assert(std::abs(stepped[0] - 0.04F) < 1e-6F && stepped[1] == 0.0F);
  const auto diagonal = slew_velocity_xy({}, {1.0F, 1.0F}, 0.04F);
  assert(std::abs(std::hypot(diagonal[0], diagonal[1]) - 0.04F) < 1e-6F);
  assert(stationary_with_hysteresis(true, 0.05F));
  assert(!stationary_with_hysteresis(true, 0.07F));
  assert(!stationary_with_hysteresis(false, 0.05F));
  assert(stationary_with_hysteresis(false, 0.03F));

  // 5. Operator torso height: headset fraction of standing height scales the robot's upright height.
  OperatorHeight op;
  assert(!op.valid());
  assert(torso_height_from_operator(1.0F, op, 0.8F) == 0.8F);  // unknown operator: upright
  op.observe(1.7F);
  op.observe(1.2F);  // crouching never lowers the learned standing height
  assert(op.valid() && op.standing == 1.7F);
  assert(std::abs(torso_height_from_operator(0.85F, op, 0.8F) - 0.4F) < 1e-5F);
  assert(torso_height_from_operator(1.9F, op, 0.8F) == 0.8F);  // tiptoe clamps to upright
  assert(torso_height_from_operator(std::nanf(""), op, 0.8F) == 0.8F);
  // An eye-level headset origin reports shoulder-height hands near zero; levelling by
  // the floor offset (headset reading minus the known standing height) restores heights.
  PicoInputState raw{};
  raw.headset_height = 0.1F;  // standing, origin 1.6 m up
  raw.left_hand_position_world = {0.3F, 0.2F, -0.2F};
  raw.right_hand_position_world = {0.3F, -0.2F, -0.3F};
  const PicoInputState levelled = level_to_floor(raw, raw.headset_height - 1.7F);
  assert(std::abs(levelled.headset_height - 1.7F) < 1e-5F);
  assert(std::abs(levelled.left_hand_position_world[2] - 1.4F) < 1e-5F);
  assert(std::abs(levelled.right_hand_position_world[2] - 1.3F) < 1e-5F);
  assert(levelled.left_hand_position_world[0] == raw.left_hand_position_world[0]);  // only heights move
  // 6. Prompt pre-emphasis: bias grows toward the error, slew-limited, capped, damped by velocity.
  PromptEmphasis pe;
  std::array<float, 3> bias{};
  const std::array<float, 3> pe_goal{1.0F, 0.0F, 1.0F}, hand{0.9F, 0.0F, 1.0F}, still{};
  bias = prompt_emphasis_step(bias, pe_goal, hand, still, pe, 0.02F);
  assert(bias[0] > 0.0F && bias[1] == 0.0F && bias[2] == 0.0F);
  assert(bias[0] <= pe.slew_mps * 0.02F + 1e-6F);  // one cycle cannot move the bias more than the slew allows
  for (int i = 0; i < 500; ++i) bias = prompt_emphasis_step(bias, pe_goal, hand, still, pe, 0.02F);
  assert(std::abs(bias[0] - 0.1F) < 1e-3F);  // settles at gain * error
  std::array<float, 3> capped{};
  for (int i = 0; i < 500; ++i) capped = prompt_emphasis_step(capped, {2.0F, 0.0F, 1.0F}, hand, still, pe, 0.02F);
  assert(std::abs(std::hypot(capped[0], capped[1], capped[2]) - pe.max_bias) < 1e-3F);  // capped
  std::array<float, 3> damped{};
  const std::array<float, 3> closing{0.5F, 0.0F, 0.0F};  // hand already moving toward the goal
  for (int i = 0; i < 500; ++i) damped = prompt_emphasis_step(damped, pe_goal, hand, closing, pe, 0.02F);
  assert(damped[0] < bias[0]);
  std::printf("teleop mapping: support-frame goals, workspace clamp, Pico validity, operator-height and prompt-emphasis checks pass\n");
  return 0;
}
