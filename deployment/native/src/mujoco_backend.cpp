// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/mujoco_backend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>

namespace protomotions::deployment {
namespace {

}  // namespace

class MuJoCoBackend::Impl {
 public:
  explicit Impl(MuJoCoBackendConfig config) : config_(std::move(config)) {
    if (config_.xml_path.empty()) {
      throw std::invalid_argument("MuJoCo xml_path is required");
    }
    if (config_.substeps_per_command < 1)
      throw std::invalid_argument("MuJoCo substeps_per_command must be positive");
    if (!std::isfinite(config_.emergency_damping) ||
        config_.emergency_damping < 0.0) {
      throw std::invalid_argument(
          "MuJoCo emergency_damping must be finite and non-negative");
    }
    char error[1024]{};
    model_ = mj_loadXML(config_.xml_path.c_str(), nullptr, error,
                        static_cast<int>(sizeof(error)));
    if (!model_) throw std::runtime_error(error[0] ? error : "mj_loadXML failed");
    data_ = mj_makeData(model_);
    if (!data_) {
      mj_deleteModel(model_);
      model_ = nullptr;
      throw std::runtime_error("mj_makeData failed");
    }
    mj_forward(model_, data_);
    if (model_->nu <= 0) {
      mj_deleteData(data_);
      mj_deleteModel(model_);
      data_ = nullptr;
      model_ = nullptr;
      throw std::invalid_argument("MuJoCo model must have at least one actuator");
    }
    for (int actuator = 0; actuator < model_->nu; ++actuator) {
      const int joint = model_->actuator_trnid[2 * actuator];
      if (joint < 0 ||
          (model_->jnt_type[joint] != mjJNT_HINGE &&
           model_->jnt_type[joint] != mjJNT_SLIDE)) {
        mj_deleteData(data_);
        mj_deleteModel(model_);
        data_ = nullptr;
        model_ = nullptr;
        throw std::invalid_argument(
            "MuJoCo backend supports only scalar hinge/slide actuators");
      }
      joint_qpos_adr_.push_back(model_->jnt_qposadr[joint]);
      joint_qvel_adr_.push_back(model_->jnt_dofadr[joint]);
      base_gainprm_.push_back(model_->actuator_gainprm[10 * actuator]);
      base_biasprm_1_.push_back(model_->actuator_biasprm[10 * actuator + 1]);
      base_biasprm_2_.push_back(model_->actuator_biasprm[10 * actuator + 2]);
    }
    if (!config_.initial_dof_position.empty()) {
      if (config_.initial_dof_position.size() != joint_qpos_adr_.size() ||
          !std::all_of(config_.initial_dof_position.begin(),
                       config_.initial_dof_position.end(),
                       [](double value) { return std::isfinite(value); }))
        throw std::invalid_argument("invalid MuJoCo startup joint pose");
      for (std::size_t joint = 0; joint < joint_qpos_adr_.size(); ++joint)
        data_->qpos[joint_qpos_adr_[joint]] = config_.initial_dof_position[joint];
      mj_forward(model_, data_);
    }
    // MuJoCo permits fixed-base and multi-root models. Only expose a base
    // pose when an actual free joint is present; never infer it from nq/nv.
    for (int joint = 0; joint < model_->njnt; ++joint) {
      if (model_->jnt_type[joint] == mjJNT_FREE) {
        free_joint_qpos_adr_ = model_->jnt_qposadr[joint];
        free_joint_qvel_adr_ = model_->jnt_dofadr[joint];
        break;
      }
    }
    if (!config_.policy_anchor_body_name.empty()) {
      if (free_joint_qpos_adr_ < 0 || free_joint_qvel_adr_ < 0)
        throw std::invalid_argument("scene-policy MuJoCo model needs a free root");
      anchor_body_id_ = mj_name2id(model_, mjOBJ_BODY,
                                   config_.policy_anchor_body_name.c_str());
      if (anchor_body_id_ < 0)
        throw std::invalid_argument("scene-policy MuJoCo anchor body not found");
    }
    capabilities_.dof_count = static_cast<std::size_t>(model_->nu);
    capabilities_.has_odometry = anchor_body_id_ >= 0;
    capabilities_.has_projected_gravity = anchor_body_id_ >= 0;
    capabilities_.supports_position =
        config_.command_mode == ControlMode::kPosition;
    capabilities_.supports_torque = config_.command_mode == ControlMode::kTorque;
    capabilities_.supports_velocity =
        config_.command_mode == ControlMode::kVelocity;
    capabilities_.supports_anchor_reset = true;
    capabilities_.resets_state_sequence = true;
  }

  ~Impl() {
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
  }

  const RobotCapabilities& capabilities() const noexcept { return capabilities_; }

  bool read_state(RobotState& state) {
    state.timestamp = std::chrono::steady_clock::now();
    state.sequence = ++sequence_;
    state.has_anchor_position = false;
    state.has_root_kinematics = false;
    state.has_projected_gravity = false;
    state.dof_position.resize(joint_qpos_adr_.size());
    state.dof_velocity.resize(joint_qvel_adr_.size());
    state.dof_torque.resize(joint_qvel_adr_.size());
    for (std::size_t i = 0; i < joint_qpos_adr_.size(); ++i) {
      state.dof_position[i] = data_->qpos[joint_qpos_adr_[i]];
      state.dof_velocity[i] = data_->qvel[joint_qvel_adr_[i]];
      state.dof_torque[i] = data_->qfrc_actuator[joint_qvel_adr_[i]];
    }
    if (free_joint_qpos_adr_ >= 0 && free_joint_qvel_adr_ >= 0) {
      const auto qpos = free_joint_qpos_adr_;
      const auto qvel = free_joint_qvel_adr_;
      state.base_position = {data_->qpos[qpos + 0], data_->qpos[qpos + 1],
                             data_->qpos[qpos + 2]};
      // MuJoCo stores free-joint quaternions as wxyz; deployment is xyzw.
      state.base_orientation_xyzw = {data_->qpos[qpos + 4], data_->qpos[qpos + 5],
                                     data_->qpos[qpos + 6], data_->qpos[qpos + 3]};
      state.base_angular_velocity = {data_->qvel[qvel + 3], data_->qvel[qvel + 4],
                                     data_->qvel[qvel + 5]};
      if (anchor_body_id_ >= 0) {
        state.root_position = state.base_position;
        state.root_orientation_xyzw = state.base_orientation_xyzw;
        state.root_angular_velocity = state.base_angular_velocity;
        state.has_root_kinematics = true;
        const mjtNum* anchor_pos = data_->xpos + 3 * anchor_body_id_;
        const mjtNum* anchor_wxyz = data_->xquat + 4 * anchor_body_id_;
        state.anchor_position = {anchor_pos[0], anchor_pos[1], anchor_pos[2]};
        state.base_orientation_xyzw = {anchor_wxyz[1], anchor_wxyz[2],
                                       anchor_wxyz[3], anchor_wxyz[0]};
        mjtNum local_velocity[6]{};
        mj_objectVelocity(model_, data_, mjOBJ_BODY, anchor_body_id_,
                          local_velocity, 1);
        state.base_angular_velocity = {local_velocity[0], local_velocity[1],
                                       local_velocity[2]};
        mjtNum inverse[4] = {anchor_wxyz[0], -anchor_wxyz[1],
                             -anchor_wxyz[2], -anchor_wxyz[3]};
        const mjtNum gravity_world[3] = {0, 0, -1};
        mjtNum gravity_local[3]{};
        mju_rotVecQuat(gravity_local, gravity_world, inverse);
        state.projected_gravity = {gravity_local[0], gravity_local[1],
                                   gravity_local[2]};
        state.has_anchor_position = true;
        state.has_projected_gravity = true;
        state.has_odometry = true;
        state.odometry_velocity = {data_->qvel[qvel], data_->qvel[qvel + 1],
                                   data_->qvel[qvel + 2]};
      }
    }
    return state.finite();
  }

  bool send_command(const ActuatorCommand& command) {
    if (command.mode != config_.command_mode ||
        command.values.size() != capabilities_.dof_count || !command.finite()) {
      return false;
    }
    if ((!command.stiffness.empty() &&
         command.stiffness.size() != capabilities_.dof_count) ||
        (!command.damping.empty() &&
         command.damping.size() != capabilities_.dof_count)) {
      return false;
    }
    if (command.mode == ControlMode::kPosition) {
      // Position actuators use gainprm[0] for kp and biasprm[1:2] for the
      // corresponding -kp/-kd terms. An explicit vector overrides the XML
      // value; an omitted vector restores the XML value. This matters after a
      // fault: emergency_stop() deliberately installs damping gains, but a
      // later gain-omitting policy command must not inherit them.
      const bool stiffness_changed =
          !command.stiffness.empty() && stiffness_cache_ != command.stiffness;
      const bool damping_changed =
          !command.damping.empty() && damping_cache_ != command.damping;
      const bool restore_stiffness =
          command.stiffness.empty() && !stiffness_cache_.empty();
      const bool restore_damping = command.damping.empty() && !damping_cache_.empty();
      if (stiffness_changed || damping_changed || restore_stiffness ||
          restore_damping) {
        for (std::size_t i = 0; i < capabilities_.dof_count; ++i) {
          if (stiffness_changed) {
            const double value = command.stiffness[i];
            model_->actuator_gainprm[10 * i] = value;
            model_->actuator_biasprm[10 * i + 1] = -value;
          } else if (restore_stiffness) {
            model_->actuator_gainprm[10 * i] = base_gainprm_[i];
            model_->actuator_biasprm[10 * i + 1] = base_biasprm_1_[i];
          }
          if (damping_changed) {
            model_->actuator_biasprm[10 * i + 2] = -command.damping[i];
          } else if (restore_damping) {
            model_->actuator_biasprm[10 * i + 2] = base_biasprm_2_[i];
          }
        }
        if (stiffness_changed) {
          stiffness_cache_ = command.stiffness;
        } else if (restore_stiffness) {
          stiffness_cache_.clear();
        }
        if (damping_changed) {
          damping_cache_ = command.damping;
        } else if (restore_damping) {
          damping_cache_.clear();
        }
      }
    }
    std::copy(command.values.begin(), command.values.end(), data_->ctrl);
    if (config_.step_simulation_on_send)
      for (int substep = 0; substep < config_.substeps_per_command; ++substep)
        mj_step(model_, data_);
    return true;
  }

  bool reset_anchor_frame() {
    mj_resetData(model_, data_);
    restore_xml_gains();
    mj_forward(model_, data_);
    sequence_ = 0;
    return true;
  }

  bool emergency_stop() noexcept {
    if (!data_ || !model_) return false;
    try {
      ActuatorCommand stop;
      stop.mode = config_.command_mode;
      stop.values.resize(capabilities_.dof_count);
      if (config_.command_mode == ControlMode::kPosition) {
        stop.stiffness.assign(capabilities_.dof_count, 0.0);
        stop.damping.assign(capabilities_.dof_count,
                            config_.emergency_damping);
        for (std::size_t i = 0; i < joint_qpos_adr_.size(); ++i) {
          const double position = data_->qpos[joint_qpos_adr_[i]];
          stop.values[i] = std::isfinite(position) ? position : 0.0;
        }
      } else {
        std::fill(stop.values.begin(), stop.values.end(), 0.0);
      }
      return send_command(stop);
    } catch (...) {
      return false;
    }
  }

 private:
  void restore_xml_gains() noexcept {
    for (std::size_t i = 0; i < capabilities_.dof_count; ++i) {
      model_->actuator_gainprm[10 * i] = base_gainprm_[i];
      model_->actuator_biasprm[10 * i + 1] = base_biasprm_1_[i];
      model_->actuator_biasprm[10 * i + 2] = base_biasprm_2_[i];
    }
    stiffness_cache_.clear();
    damping_cache_.clear();
  }

  MuJoCoBackendConfig config_;
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  RobotCapabilities capabilities_{};
  std::vector<int> joint_qpos_adr_;
  std::vector<int> joint_qvel_adr_;
  std::vector<double> base_gainprm_;
  std::vector<double> base_biasprm_1_;
  std::vector<double> base_biasprm_2_;
  std::vector<double> stiffness_cache_;
  std::vector<double> damping_cache_;
  int free_joint_qpos_adr_ = -1;
  int free_joint_qvel_adr_ = -1;
  int anchor_body_id_ = -1;
  std::uint64_t sequence_ = 0;
};

MuJoCoBackend::MuJoCoBackend(MuJoCoBackendConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

MuJoCoBackend::~MuJoCoBackend() = default;

const RobotCapabilities& MuJoCoBackend::capabilities() const noexcept {
  return impl_->capabilities();
}
bool MuJoCoBackend::read_state(RobotState& state) { return impl_->read_state(state); }
bool MuJoCoBackend::send_command(const ActuatorCommand& command) {
  return impl_->send_command(command);
}
bool MuJoCoBackend::reset_anchor_frame() { return impl_->reset_anchor_frame(); }
bool MuJoCoBackend::emergency_stop() noexcept { return impl_->emergency_stop(); }

}  // namespace protomotions::deployment
