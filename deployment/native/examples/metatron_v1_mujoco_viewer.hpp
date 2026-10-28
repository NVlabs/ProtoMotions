// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Optional visualizer for the native MuJoCo control loop. It mirrors the
// backend's measured state and never provides policy observations or actions.
#include <dlfcn.h>
#include <mujoco/mujoco.h>

#include <stdexcept>
#include <string>

#include "protomotions/deployment/types.hpp"

struct GLFWwindow;
struct GLFWmonitor;

namespace protomotions::deployment::metatron_v1 {

class MujocoViewer final {
 public:
  MujocoViewer(const std::string& xml, const std::string& glfw_library) {
    library_ = dlopen(glfw_library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error("cannot load GLFW viewer library: " + glfw_library);
    try {
      init_ = load<int (*)()>("glfwInit");
      terminate_ = load<void (*)()>("glfwTerminate");
      create_ = load<GLFWwindow* (*)(int, int, const char*, GLFWmonitor*, GLFWwindow*)>("glfwCreateWindow");
      destroy_ = load<void (*)(GLFWwindow*)>("glfwDestroyWindow");
      current_ = load<void (*)(GLFWwindow*)>("glfwMakeContextCurrent");
      swap_ = load<void (*)(GLFWwindow*)>("glfwSwapBuffers");
      poll_ = load<void (*)()>("glfwPollEvents");
      should_close_ = load<int (*)(GLFWwindow*)>("glfwWindowShouldClose");
      size_ = load<void (*)(GLFWwindow*, int*, int*)>("glfwGetFramebufferSize");
      if (!init_()) throw std::runtime_error("GLFW initialization failed; check DISPLAY");
      initialized_ = true;
      window_ = create_(960, 720, "Metatron native MuJoCo", nullptr, nullptr);
      if (!window_) throw std::runtime_error("GLFW window creation failed");
      current_(window_);
      char error[1024]{};
      model_ = mj_loadXML(xml.c_str(), nullptr, error, sizeof(error));
      if (!model_) throw std::runtime_error(error[0] ? error : "MuJoCo viewer XML failed");
      data_ = mj_makeData(model_);
      if (!data_) throw std::runtime_error("MuJoCo viewer data creation failed");
      mjv_defaultCamera(&camera_);
      mjv_defaultOption(&option_);
      mjv_defaultScene(&scene_);
      mjr_defaultContext(&context_);
      mjv_makeScene(model_, &scene_, 3000);
      mjr_makeContext(model_, &context_, mjFONTSCALE_150);
      camera_.azimuth = 135;
      camera_.elevation = -18;
      camera_.distance = 2.7;
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~MujocoViewer() { cleanup(); }
  MujocoViewer(const MujocoViewer&) = delete;
  MujocoViewer& operator=(const MujocoViewer&) = delete;

  bool render(const RobotState& state) {
    if (should_close_(window_)) return false;
    if (state.dof_position.size() != static_cast<std::size_t>(model_->nu))
      throw std::runtime_error("viewer state DOF dimension mismatch");
    for (int axis = 0; axis < 3; ++axis)
      data_->qpos[axis] = state.root_position[axis];
    data_->qpos[3] = state.root_orientation_xyzw[3];
    for (int axis = 0; axis < 3; ++axis)
      data_->qpos[4 + axis] = state.root_orientation_xyzw[axis];
    for (int joint = 0; joint < model_->nu; ++joint) {
      const int id = model_->actuator_trnid[2 * joint];
      data_->qpos[model_->jnt_qposadr[id]] = state.dof_position[joint];
    }
    mj_forward(model_, data_);
    current_(window_);
    int width = 0, height = 0;
    size_(window_, &width, &height);
    camera_.lookat[0] = state.root_position[0];
    camera_.lookat[1] = state.root_position[1];
    camera_.lookat[2] = state.root_position[2];
    mjv_updateScene(model_, data_, &option_, nullptr, &camera_, mjCAT_ALL, &scene_);
    mjrRect viewport{0, 0, width, height};
    mjr_render(viewport, &scene_, &context_);
    swap_(window_);
    poll_();
    return true;
  }

 private:
  template <typename Function>
  Function load(const char* name) {
    void* symbol = dlsym(library_, name);
    if (!symbol) throw std::runtime_error(std::string("GLFW symbol missing: ") + name);
    return reinterpret_cast<Function>(symbol);
  }

  void cleanup() noexcept {
    if (window_ && current_) current_(window_);
    if (model_) {
      mjr_freeContext(&context_);
      mjv_freeScene(&scene_);
    }
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
    if (window_ && destroy_) destroy_(window_);
    if (initialized_ && terminate_) terminate_();
    if (library_) dlclose(library_);
    model_ = nullptr;
    data_ = nullptr;
    window_ = nullptr;
    library_ = nullptr;
  }

  void* library_ = nullptr;
  bool initialized_ = false;
  GLFWwindow* window_ = nullptr;
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  mjvCamera camera_{};
  mjvOption option_{};
  mjvScene scene_{};
  mjrContext context_{};
  int (*init_)() = nullptr;
  void (*terminate_)() = nullptr;
  GLFWwindow* (*create_)(int, int, const char*, GLFWmonitor*, GLFWwindow*) = nullptr;
  void (*destroy_)(GLFWwindow*) = nullptr;
  void (*current_)(GLFWwindow*) = nullptr;
  void (*swap_)(GLFWwindow*) = nullptr;
  void (*poll_)() = nullptr;
  int (*should_close_)(GLFWwindow*) = nullptr;
  void (*size_)(GLFWwindow*, int*, int*) = nullptr;
};

}  // namespace protomotions::deployment::metatron_v1
