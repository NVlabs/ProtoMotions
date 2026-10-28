// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/controller_manager.hpp"

namespace protomotions::deployment {

bool ControllerManager::add(ControllerPtr controller) {
  if (!controller) return false;
  const std::string key(controller->name());
  if (key.empty() || controllers_.contains(key)) return false;
  controllers_.emplace(key, std::move(controller));
  return true;
}

bool ControllerManager::request_switch(const std::string& name) {
  if (!controllers_.contains(name)) return false;
  requested_ = name;
  return true;
}

bool ControllerManager::activate_requested(const RobotState& state) {
  if (requested_.empty()) return false;
  auto it = controllers_.find(requested_);
  if (it == controllers_.end()) return false;
  if (active_ == it->second.get()) return true;
  if (active_) active_->stop();
  if (!it->second->start(state)) {
    active_ = nullptr;
    return false;
  }
  active_ = it->second.get();
  return true;
}

std::optional<ActuatorCommand> ControllerManager::step(const RobotState& state) {
  if (!activate_requested(state)) return std::nullopt;
  return active_->step(state);
}

void ControllerManager::on_command_accepted(const ActuatorCommand& command) {
  if (active_) active_->on_command_accepted(command);
}

void ControllerManager::stop() noexcept {
  if (active_) active_->stop();
  active_ = nullptr;
  requested_.clear();
}

std::string ControllerManager::active_name() const {
  return active_ ? std::string(active_->name()) : std::string();
}

std::string ControllerManager::requested_name() const { return requested_; }

}  // namespace protomotions::deployment
