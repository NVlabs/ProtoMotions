// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/metatron_v1_arm_ik.hpp"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <stdexcept>

namespace protomotions::deployment::metatron_v1 {
namespace {

constexpr int kArmJoints = 7;
constexpr int kRows = 6;

struct ArmChain {
  int body_id = -1;
  std::array<int, kArmJoints> policy_index{};
  std::array<int, kArmJoints> dof_adr{};
  std::array<double, kArmJoints> lower{};
  std::array<double, kArmJoints> upper{};
  std::array<double, kArmJoints> offset{};
};

std::array<double, 3> rotation_error(const std::array<float, 4>& desired_xyzw,
                                     const double* measured_wxyz) {
  // q = desired * conj(measured), shortest arc, as a rotation vector.
  const double dw = desired_xyzw[3], dx = desired_xyzw[0], dy = desired_xyzw[1],
               dz = desired_xyzw[2];
  const double mw = measured_wxyz[0], mx = -measured_wxyz[1], my = -measured_wxyz[2],
               mz = -measured_wxyz[3];
  double w = dw * mw - dx * mx - dy * my - dz * mz;
  double x = dw * mx + dx * mw + dy * mz - dz * my;
  double y = dw * my - dx * mz + dy * mw + dz * mx;
  double z = dw * mz + dx * my - dy * mx + dz * mw;
  const double n = std::sqrt(w * w + x * x + y * y + z * z);
  if (n > 0.0) { w /= n; x /= n; y /= n; z /= n; }
  if (w < 0.0) { w = -w; x = -x; y = -y; z = -z; }
  const double v = std::sqrt(x * x + y * y + z * z);
  if (v < 1e-10) return {2.0 * x, 2.0 * y, 2.0 * z};
  const double s = 2.0 * std::atan2(v, w) / v;
  return {s * x, s * y, s * z};
}

void clamp_length(std::array<double, 3>& v, double limit) {
  const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (n > limit && n > 0.0) {
    const double s = limit / n;
    v[0] *= s; v[1] *= s; v[2] *= s;
  }
}

}  // namespace

std::array<float, 2> correction_bounds(float target, float offset, float lower,
                                       float upper, float dt, float budget,
                                       float max_rate) {
  const float rate = max_rate * dt;
  const float allowed_low = std::clamp(lower - target, -budget, budget);
  const float allowed_high = std::clamp(upper - target, -budget, budget);
  return {std::clamp(allowed_low - offset, -rate, rate),
          std::clamp(allowed_high - offset, -rate, rate)};
}

class ArmIkCorrection::Impl {
 public:
  Impl(const std::string& xml_path, const std::vector<std::string>& joint_names,
       const std::array<float, kDofCount>& scale,
       const std::array<float, kDofCount>& offset, ArmIkSettings settings)
      : scale_(scale), offset_(offset), settings_(settings) {
    if (joint_names.size() != kDofCount)
      throw std::invalid_argument("arm IK needs the policy's 29 joint names");
    char error[1024]{};
    model_ = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
    if (!model_) throw std::runtime_error(error[0] ? error : "arm IK: MuJoCo XML load failed");
    data_ = mj_makeData(model_);
    if (!data_) {
      mj_deleteModel(model_);
      throw std::runtime_error("arm IK: mj_makeData failed");
    }
    for (int joint = 0; joint < model_->njnt; ++joint) {
      if (model_->jnt_type[joint] == mjJNT_FREE) {
        root_qpos_adr_ = model_->jnt_qposadr[joint];
        break;
      }
    }
    if (root_qpos_adr_ < 0) throw std::invalid_argument("arm IK needs a free-root robot model");
    for (std::size_t index = 0; index < kDofCount; ++index) {
      const int joint = mj_name2id(model_, mjOBJ_JOINT, joint_names[index].c_str());
      if (joint < 0 || model_->jnt_type[joint] != mjJNT_HINGE)
        throw std::invalid_argument("arm IK: policy joint missing or not a hinge: " + joint_names[index]);
      joint_ids_[index] = joint;
      qpos_adr_[index] = model_->jnt_qposadr[joint];
    }
    const char* sides[2] = {"left", "right"};
    for (int side = 0; side < 2; ++side) {
      ArmChain& arm = arms_[side];
      const std::string hand = std::string(sides[side]) + "_rubber_hand";
      const std::string shoulder = std::string(sides[side]) + "_shoulder_pitch_joint";
      arm.body_id = mj_name2id(model_, mjOBJ_BODY, hand.c_str());
      const int shoulder_joint = mj_name2id(model_, mjOBJ_JOINT, shoulder.c_str());
      if (arm.body_id < 0 || shoulder_joint < 0)
        throw std::invalid_argument("arm IK: missing " + hand + " or " + shoulder);
      // Walk the hand's ancestry up to the shoulder's parent, never the waist.
      std::set<int> chain;
      int body = arm.body_id;
      const int stop = model_->body_parentid[model_->jnt_bodyid[shoulder_joint]];
      while (body != stop && body > 0) {
        chain.insert(body);
        body = model_->body_parentid[body];
      }
      if (body != stop) throw std::invalid_argument("arm IK: hand is not below " + shoulder);
      int count = 0;
      for (std::size_t index = 0; index < kDofCount; ++index) {
        const int joint = joint_ids_[index];
        if (!chain.count(model_->jnt_bodyid[joint])) continue;
        if (count == kArmJoints) throw std::invalid_argument("arm IK: more than seven joints in the arm chain");
        const double lo = model_->jnt_range[2 * joint], hi = model_->jnt_range[2 * joint + 1];
        if (!(lo < hi)) throw std::invalid_argument("arm IK: arm joint without a range: " + joint_names[index]);
        const double mid = 0.5 * (lo + hi), half = 0.5 * (hi - lo) * settings_.soft_limit_factor;
        arm.policy_index[count] = static_cast<int>(index);
        arm.dof_adr[count] = model_->jnt_dofadr[joint];
        arm.lower[count] = mid - half;
        arm.upper[count] = mid + half;
        ++count;
      }
      if (count != kArmJoints) throw std::invalid_argument("arm IK: expected seven joints from shoulder to hand");
    }
    jac_pos_.assign(3 * static_cast<std::size_t>(model_->nv), 0.0);
    jac_rot_.assign(3 * static_cast<std::size_t>(model_->nv), 0.0);
  }

  ~Impl() {
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
  }

  void reset() noexcept {
    for (auto& arm : arms_) arm.offset.fill(0.0);
    delta_.fill(0.0F);
    diagnostics_ = {};
  }

  void apply(Outputs& outputs, const RobotState& state, float dt) {
    const auto start = std::chrono::steady_clock::now();
    delta_.fill(0.0F);
    if (state.dof_position.size() != kDofCount) throw std::invalid_argument("arm IK: state has wrong joint count");
    // Pose the kinematic model from the measurement; the physics is never stepped.
    data_->qpos[root_qpos_adr_ + 0] = state.root_position[0];
    data_->qpos[root_qpos_adr_ + 1] = state.root_position[1];
    data_->qpos[root_qpos_adr_ + 2] = state.root_position[2];
    data_->qpos[root_qpos_adr_ + 3] = state.root_orientation_xyzw[3];
    data_->qpos[root_qpos_adr_ + 4] = state.root_orientation_xyzw[0];
    data_->qpos[root_qpos_adr_ + 5] = state.root_orientation_xyzw[1];
    data_->qpos[root_qpos_adr_ + 6] = state.root_orientation_xyzw[2];
    for (std::size_t index = 0; index < kDofCount; ++index)
      data_->qpos[qpos_adr_[index]] = state.dof_position[index];
    mj_kinematics(model_, data_);
    mj_comPos(model_, data_);
    for (int side = 0; side < 2; ++side) {
      const HandGoal& goal = goals_[side];
      ArmChain& arm = arms_[side];
      if (!goal.enabled || !(goal.translation || goal.rotation)) {
        arm.offset.fill(0.0);
        diagnostics_.offset_norm_rad[side] = 0.0F;
        continue;
      }
      correct_arm(arm, goal, outputs, dt, side);
    }
    diagnostics_.solve_milliseconds = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }

  void correct_arm(ArmChain& arm, const HandGoal& goal, Outputs& outputs, float dt, int side) {
    const double* xpos = data_->xpos + 3 * arm.body_id;
    const double* xquat = data_->xquat + 4 * arm.body_id;
    std::array<double, 3> translation = {goal.position[0] - xpos[0], goal.position[1] - xpos[1],
                                         goal.position[2] - xpos[2]};
    diagnostics_.hand_error_m[side] = static_cast<float>(std::sqrt(
        translation[0] * translation[0] + translation[1] * translation[1] + translation[2] * translation[2]));
    clamp_length(translation, settings_.translation_clamp);
    std::array<double, 3> orientation = rotation_error(goal.rotation_xyzw, xquat);
    diagnostics_.hand_rotation_rad[side] = static_cast<float>(std::sqrt(
        orientation[0] * orientation[0] + orientation[1] * orientation[1] + orientation[2] * orientation[2]));
    clamp_length(orientation, settings_.rotation_clamp);
    for (auto& v : orientation) v *= settings_.orientation_length;
    const std::array<double, kRows> error = {translation[0], translation[1], translation[2],
                                             orientation[0], orientation[1], orientation[2]};
    mj_jacBody(model_, data_, jac_pos_.data(), jac_rot_.data(), arm.body_id);
    const int nv = model_->nv;
    std::array<std::array<double, kArmJoints>, kRows> jac{};
    for (int row = 0; row < kRows; ++row) {
      const bool enabled = row < 3 ? goal.translation : goal.rotation;
      for (int i = 0; i < kArmJoints; ++i) {
        if (!enabled) { jac[row][i] = 0.0; continue; }
        jac[row][i] = row < 3 ? jac_pos_[row * nv + arm.dof_adr[i]]
                              : settings_.orientation_length * jac_rot_[(row - 3) * nv + arm.dof_adr[i]];
      }
    }
    // Damped least squares as a box-constrained QP in the offset increment:
    // min 1/2 x'Hx + g'x with H = J'J + damping^2 I and
    // g = -fraction J'e + H old, so old + x re-solves the offset from scratch.
    std::array<double, kArmJoints * kArmJoints> normal{};
    std::array<double, kArmJoints> rhs{};
    const double damping2 = static_cast<double>(settings_.damping) * settings_.damping;
    for (int i = 0; i < kArmJoints; ++i) {
      double projected = 0.0;
      for (int row = 0; row < kRows; ++row) projected += jac[row][i] * error[row];
      rhs[i] = -static_cast<double>(settings_.fraction) * projected;
      for (int j = 0; j < kArmJoints; ++j) {
        double sum = 0.0;
        for (int row = 0; row < kRows; ++row) sum += jac[row][i] * jac[row][j];
        normal[i * kArmJoints + j] = sum + (i == j ? damping2 : 0.0);
      }
    }
    for (int i = 0; i < kArmJoints; ++i)
      for (int j = 0; j < kArmJoints; ++j) rhs[i] += normal[i * kArmJoints + j] * arm.offset[j];
    std::array<double, kArmJoints> lower{}, upper{};
    std::array<bool, kArmJoints> fixed{};
    for (int i = 0; i < kArmJoints; ++i) {
      const int policy = arm.policy_index[i];
      const float budget = std::min(settings_.max_offset, settings_.max_raw_offset * scale_[policy]);
      const auto bounds = correction_bounds(outputs.action[policy], static_cast<float>(arm.offset[i]),
                                            static_cast<float>(arm.lower[i]), static_cast<float>(arm.upper[i]),
                                            dt, budget, settings_.max_rate);
      lower[i] = bounds[0];
      upper[i] = bounds[1];
      fixed[i] = bounds[0] == bounds[1];
    }
    // MuJoCo needs strictly separated bounds: substitute fixed increments into
    // the objective exactly and leave an independent dummy variable in the slot.
    for (int i = 0; i < kArmJoints; ++i)
      for (int j = 0; j < kArmJoints; ++j)
        if (fixed[j]) rhs[i] += normal[i * kArmJoints + j] * lower[j];
    for (int i = 0; i < kArmJoints; ++i) {
      if (!fixed[i]) continue;
      for (int j = 0; j < kArmJoints; ++j) normal[i * kArmJoints + j] = normal[j * kArmJoints + i] = 0.0;
      normal[i * kArmJoints + i] = 1.0;
      rhs[i] = 0.0;
    }
    std::array<double, kArmJoints> lower_qp = lower, upper_qp = upper;
    for (int i = 0; i < kArmJoints; ++i)
      if (fixed[i]) { lower_qp[i] = -1.0; upper_qp[i] = 1.0; }
    std::array<double, kArmJoints> solution{};
    std::array<double, kArmJoints * (kArmJoints + 7)> factor{};
    std::array<int, kArmJoints> index{};
    const int rank = mju_boxQP(solution.data(), factor.data(), index.data(), normal.data(), rhs.data(),
                               kArmJoints, lower_qp.data(), upper_qp.data());
    if (rank < 0) throw std::runtime_error("arm IK constrained solve failed");
    double norm = 0.0;
    for (int i = 0; i < kArmJoints; ++i) {
      const double increment = fixed[i] ? lower[i] : solution[i];
      const int policy = arm.policy_index[i];
      const double delta = arm.offset[i] + increment;
      arm.offset[i] = delta;
      delta_[policy] = static_cast<float>(delta);
      norm += delta * delta;
      if (delta != 0.0) {
        // Invert only edited joints, then apply the same affine map as ONNX so
        // the committed history matches what the motors receive.
        const float raw = (outputs.action[policy] + static_cast<float>(delta) - offset_[policy]) / scale_[policy];
        outputs.raw_action[policy] = raw;
        outputs.action[policy] = offset_[policy] + scale_[policy] * raw;
      }
    }
    diagnostics_.offset_norm_rad[side] = static_cast<float>(std::sqrt(norm));
  }

  std::array<float, kDofCount> scale_;
  std::array<float, kDofCount> offset_;
  ArmIkSettings settings_;
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  int root_qpos_adr_ = -1;
  std::array<int, kDofCount> joint_ids_{};
  std::array<int, kDofCount> qpos_adr_{};
  std::array<ArmChain, 2> arms_{};
  std::array<HandGoal, 2> goals_{};
  std::vector<double> jac_pos_;
  std::vector<double> jac_rot_;
  std::array<float, kDofCount> delta_{};
  ArmIkDiagnostics diagnostics_{};
};

ArmIkCorrection::ArmIkCorrection(const std::string& xml_path,
                                 const std::vector<std::string>& joint_names,
                                 const std::array<float, kDofCount>& scale,
                                 const std::array<float, kDofCount>& offset,
                                 ArmIkSettings settings)
    : impl_(std::make_unique<Impl>(xml_path, joint_names, scale, offset, settings)) {}
ArmIkCorrection::~ArmIkCorrection() = default;
void ArmIkCorrection::reset() noexcept { impl_->reset(); }
void ArmIkCorrection::set_goal(int side, const HandGoal& goal) {
  if (side < 0 || side > 1) throw std::invalid_argument("arm IK side must be 0 (left) or 1 (right)");
  impl_->goals_[side] = goal;
}
const HandGoal& ArmIkCorrection::goal(int side) const { return impl_->goals_.at(static_cast<std::size_t>(side)); }
ArmIkSettings& ArmIkCorrection::settings() noexcept { return impl_->settings_; }
void ArmIkCorrection::apply(Outputs& outputs, const RobotState& state, float dt) { impl_->apply(outputs, state, dt); }
const std::array<float, kDofCount>& ArmIkCorrection::delta() const noexcept { return impl_->delta_; }
const ArmIkDiagnostics& ArmIkCorrection::diagnostics() const noexcept { return impl_->diagnostics_; }

}  // namespace protomotions::deployment::metatron_v1
