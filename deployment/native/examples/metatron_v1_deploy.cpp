// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// One native 39-input launcher. Only RobotBackend differs across simulation
// and hardware; task goals, FK, history, TensorRT and safety are shared.
#include <mujoco/mujoco.h>
#include <nlohmann/json.hpp>
#include <net/if.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "protomotions/deployment/controller_manager.hpp"
#include "protomotions/deployment/evdev_xbox_input.hpp"
#include "protomotions/deployment/metatron_v1_controller.hpp"
#include "protomotions/deployment/metatron_v1_engine.hpp"
#include "protomotions/deployment/metatron_v1_fk_engine.hpp"
#include "protomotions/deployment/metatron_v1_manifest.hpp"
#include "protomotions/deployment/metatron_v1_task.hpp"
#include "metatron_v1_async_viewer.hpp"
#include "protomotions/deployment/mujoco_backend.hpp"
#include "protomotions/deployment/runtime.hpp"
#include "protomotions/deployment/unitree_dds_config.hpp"
#include "protomotions/deployment/unitree_sdk2_backend.hpp"

namespace {
using namespace protomotions::deployment;
using namespace protomotions::deployment::metatron_v1;
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t g_stop = 0;
void signal_stop(int) { g_stop = 1; }

struct Options {
  std::string backend = "mujoco";
  std::string xml, engine, fk_engine, manifest, recovery_plan, xbox_device, net_if;
  std::string glfw_library;
  bool arm = false;
  bool cuda_graph = false;
  bool viewer = false;
  bool scripted_sim = false;
  int steps = 0;
  int dds_participant_index = 0;
};

Options parse(int argc, char** argv) {
  Options args;
  for (int index = 1; index < argc; ++index) {
    const std::string flag(argv[index]);
    if (flag == "--arm") args.arm = true;
    else if (flag == "--cuda-graphs") args.cuda_graph = true;
    else if (flag == "--viewer") args.viewer = true;
    else if (flag == "--scripted-sim") args.scripted_sim = true;
    else {
      if (index + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
      const std::string value(argv[++index]);
      if (flag == "--backend") args.backend = value;
      else if (flag == "--xml") args.xml = value;
      else if (flag == "--engine") args.engine = value;
      else if (flag == "--fk-engine") args.fk_engine = value;
      else if (flag == "--manifest") args.manifest = value;
      else if (flag == "--recovery-plan") args.recovery_plan = value;
      else if (flag == "--xbox-device") args.xbox_device = value;
      else if (flag == "--net-if") args.net_if = value;
      else if (flag == "--glfw-lib") args.glfw_library = value;
      else if (flag == "--steps") args.steps = std::stoi(value);
      else if (flag == "--dds-participant-index")
        args.dds_participant_index = std::stoi(value);
      else throw std::invalid_argument("unknown option " + flag);
    }
  }
  if ((args.backend != "mujoco" && args.backend != "unitree") ||
      (args.backend == "mujoco" && args.xml.empty()) ||
      args.engine.empty() || args.fk_engine.empty() ||
      args.manifest.empty() || args.recovery_plan.empty() ||
      args.xbox_device.empty() || args.steps < 0)
    throw std::invalid_argument("required: --backend mujoco|unitree --engine --fk-engine --manifest --recovery-plan --xbox-device; MuJoCo also requires --xml");
  if (args.backend == "unitree" &&
      (!args.arm || args.net_if.empty() || if_nametoindex(args.net_if.c_str()) == 0))
    throw std::invalid_argument("hardware requires --arm and an existing --net-if");
  if (args.scripted_sim && args.backend != "mujoco")
    throw std::invalid_argument("scripted input is simulation-only");
  if (args.viewer && (args.backend != "mujoco" || args.glfw_library.empty()))
    throw std::invalid_argument("--viewer needs --backend mujoco and --glfw-lib");
  return args;
}

float yaw(const std::array<double, 4>& q) {
  return std::atan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                    1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
}
float wrap(float value) { return std::remainder(value, 6.283185307179586F); }

std::vector<double> standing_pose() {
  std::vector<double> q(29, 0.0);
  q[0] = q[6] = -0.312;
  q[3] = q[9] = 0.669;
  q[4] = q[10] = -0.363;
  q[15] = q[22] = 0.2;
  q[16] = 0.2;
  q[23] = -0.2;
  q[18] = q[25] = 0.6;
  return q;
}

int sim_substeps_and_validate(const std::string& xml,
                              const nlohmann::json& manifest) {
  char error[1024]{};
  mjModel* model = mj_loadXML(xml.c_str(), nullptr, error, sizeof(error));
  if (!model) throw std::runtime_error(error[0] ? error : "MuJoCo XML load failed");
  try {
    const auto names = manifest.at("robot_contract").at("joint_names");
    if (model->nu != 29 || model->nq != 36)
      throw std::runtime_error("sim model does not have the scene G1's 29 actuators and free root");
    for (int actuator = 0; actuator < 29; ++actuator) {
      const int joint = model->actuator_trnid[2 * actuator];
      const char* name = mj_id2name(model, mjOBJ_JOINT, joint);
      const double kp = model->actuator_gainprm[10 * actuator];
      if (!name || names[actuator] != name || kp <= 0.0 ||
          std::abs(model->actuator_biasprm[10 * actuator + 1] + kp) > 1e-4 ||
          model->actuator_biasprm[10 * actuator + 2] >= 0.0)
        throw std::runtime_error("sim XML needs matching position actuators; run prepare_metatron_mujoco_xml.py");
    }
    if (mj_name2id(model, mjOBJ_GEOM, "floor") < 0)
      throw std::runtime_error("sim XML needs a floor geom");
    const int steps = static_cast<int>(std::round(0.02 / model->opt.timestep));
    if (steps < 1 || std::abs(steps * model->opt.timestep - 0.02) > 1e-5)
      throw std::runtime_error("sim timestep must divide 20 ms control period");
    mj_deleteModel(model);
    return steps;
  } catch (...) {
    mj_deleteModel(model);
    throw;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options args = parse(argc, argv);
    const auto json = nlohmann::json::parse(std::ifstream(args.manifest));
    const auto manifest = Manifest::load(args.manifest);
    const auto body_names = json.at("robot_contract").at("body_names").get<std::vector<std::string>>();
    FkEngine fk(args.fk_engine, args.cuda_graph);
    Engine engine(args.engine, args.cuda_graph);
    if (args.cuda_graph && (!fk.cuda_graph_enabled() || !engine.cuda_graph_enabled()))
      throw std::runtime_error("requested CUDA graphs did not capture for both engines");
    RestPose rest = make_rest_pose(fk);
    MetatronTask task(fk, rest, args.recovery_plan, body_names);
    engine.warmup(10);
    EvdevXboxInputSource xbox(args.xbox_device);
    std::unique_ptr<AsyncMujocoViewer> viewer;
    if (args.viewer)
      viewer = std::make_unique<AsyncMujocoViewer>(args.xml, args.glfw_library);
    std::unique_ptr<RobotBackend> backend;
    UnitreeSdk2Backend* unitree_backend = nullptr;
    if (args.backend == "mujoco") {
      MuJoCoBackendConfig config;
      config.xml_path = args.xml;
      config.policy_anchor_body_name = "torso_link";
      config.substeps_per_command = sim_substeps_and_validate(args.xml, json);
      config.initial_dof_position = standing_pose();
      backend = std::make_unique<MuJoCoBackend>(std::move(config));
    } else {
      configure_unitree_dds(args.net_if, args.dds_participant_index);
      UnitreeSdk2BackendConfig config;
      config.network_interface = args.net_if;
      config.enable_odometry = true;
      config.max_odometry_age = std::chrono::milliseconds(100);
      config.use_lowstate_imu = true;
      config.imu_is_torso = false;
      config.damp_unused_dex3_hands = true;
      config.root_position_without_odometry = {0.0, 0.0, 0.793};
      auto sdk = std::make_unique<UnitreeSdk2Backend>(std::move(config));
      unitree_backend = sdk.get();
      backend = std::move(sdk);
    }
    ControllerManager controllers;
    if (!controllers.add(std::make_unique<MetatronController>(
            manifest.action_map(),
            [&task](Inputs& inputs, const RobotState& state) { task.build(inputs, state); },
            [&engine](const Inputs& inputs) { return engine.run(inputs); })))
      throw std::runtime_error("cannot register Metatron v1 controller");
    SafetyLimits limits;
    limits.max_abs_command = 5.0;
    limits.max_abs_stiffness = 200.0;
    limits.max_abs_damping = 30.0;
    limits.max_cycle_interval = std::chrono::milliseconds(100);
    // Hold at most two missed 50 Hz observations; never infer on a repeated tick.
    limits.repeated_state_grace = std::chrono::milliseconds(60);
    DeploymentRuntime runtime(*backend, controllers, SafetySupervisor(limits));
    RobotState initial;
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    while (Clock::now() < deadline && !backend->read_state(initial))
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (initial.sequence == 0) {
      if (unitree_backend) {
        const auto s = unitree_backend->sensor_status();
        std::cerr << "startup sensors: lowstate=" << s.lowstate_packets
                  << " crc_errors=" << s.lowstate_crc_errors
                  << " lowstate_age_ms=" << s.lowstate_age_ms
                  << " odometry=" << s.odometry_packets
                  << " odometry_age_ms=" << s.odometry_age_ms << '\n';
      }
      throw std::runtime_error("no complete fresh LowState + odometry pair; no safe hold attempted");
    }
    if (!runtime.start())
      throw std::runtime_error("startup safe hold or state validation failed: " +
                               runtime.last_fault_reason());
    std::signal(SIGINT, signal_stop);
    std::signal(SIGTERM, signal_stop);
    std::cout << "Metatron v1 ready; X=damping, Y=revive idle, A=steering, RB=idle. "
                 "Starts in safe hold. backend=" << args.backend << "\n";
    std::uint32_t previous = 0;
    bool active = false, steering = false;
    float heading_goal = 0.0F;
    auto next = Clock::now();
    int count = 0;
    while (!g_stop && (args.steps == 0 || count < args.steps)) {
      next += std::chrono::milliseconds(20);
      InputState input;
      bool input_ready = false;
      if (args.scripted_sim) {
        input.connected = true;
        input.timestamp = Clock::now();
        input.buttons = count == 10 ? xbox_button::kY :
                        count == 40 ? xbox_button::kA :
                        count == 60 || count == 130 || count == 140 ? xbox_button::kY :
                        count == 100 ? xbox_button::kRightBumper :
                        count == 120 ? xbox_button::kX : 0;
        input.axes[3] = count >= 41 && count < 100 ? 0.3F : 0.0F;
        input_ready = true;
      } else input_ready = xbox.poll(input);
      if (!input_ready ||
          Clock::now() - input.timestamp > std::chrono::milliseconds(250)) {
        runtime.emergency_stop();
        active = false;
      } else {
        const auto pressed = input.buttons & ~previous;
        previous = input.buttons;
        // X is a level-triggered safety override: holding X while tapping Y
        // must never revive the policy. Repeated Y while active must not
        // reset policy history, heading, or the current control mode.
        if (input.buttons & xbox_button::kX) {
          if (active || runtime.lifecycle() != LifecycleState::kFault)
            runtime.emergency_stop();
          active = false;
          if (pressed & xbox_button::kX) std::cout << "X: damping hold\n";
        } else if (pressed & xbox_button::kY) {
          if (!active) {
            if (runtime.lifecycle() == LifecycleState::kFault && !runtime.revive())
              throw std::runtime_error("safe revive rejected");
            task.reset();
            if (!runtime.request_controller("metatron_v1"))
              throw std::runtime_error("scene controller selection rejected");
            active = true;
            steering = false;
            std::cout << "Y: idle requested\n";
          } else {
            std::cout << "Y: already active; control unchanged\n";
          }
        } else if (active && (pressed & xbox_button::kRightBumper)) {
          steering = false;
          task.set_idle();
          std::cout << "RB: idle\n";
        } else if (active && (pressed & xbox_button::kA)) {
          steering = true;
          heading_goal = yaw(runtime.last_state().base_orientation_xyzw);
          std::cout << "A: steering\n";
        }
        if (active && steering) {
          const float lx = input.axes[0], ly = input.axes[1];
          const float magnitude = std::min(1.0F, std::hypot(lx, ly));
          if (magnitude > 0.15F) {
            const float amount = (magnitude - 0.15F) / 0.85F;
            const float angle = std::atan2(-lx, ly) * amount;
            const float quantum = 3.14159265358979323846F / 18.0F;
            heading_goal = wrap(yaw(runtime.last_state().base_orientation_xyzw) +
                                std::round(angle / quantum) * quantum);
          }
          const float rx = input.axes[2], ry = input.axes[3];
          const float level = std::min(1.0F, std::hypot(rx, ry));
          const float speed = level <= 0.15F ? 0.0F :
              std::min(1.0F, (level - 0.15F) / 0.85F);
          const float direction = heading_goal + std::atan2(-rx, ry);
          task.set_steering(heading_goal,
                            {speed * std::cos(direction), speed * std::sin(direction)});
        }
      }
      const auto result = runtime.run_cycle(Clock::now());
      if (result == CycleResult::kSafetyFault || result == CycleResult::kBackendFault ||
          result == CycleResult::kNoState) {
        active = false;
        std::cerr << "safe fault: " << runtime.last_fault_reason() << '\n';
        if (unitree_backend && result == CycleResult::kNoState) {
          const auto s = unitree_backend->sensor_status();
          std::cerr << "sensor status: lowstate=" << s.lowstate_packets
                    << " age_ms=" << s.lowstate_age_ms
                    << " crc_errors=" << s.lowstate_crc_errors
                    << " odometry=" << s.odometry_packets
                    << " age_ms=" << s.odometry_age_ms << '\n';
        }
      }
      if (args.scripted_sim && count % 25 == 0) {
        const auto& state = runtime.last_state();
        std::cout << "sim cycle=" << count << " result=" << static_cast<int>(result)
                  << " root_z=" << state.root_position[2]
                  << " recovery=" << static_cast<int>(task.recovery_phase()) << '\n';
      }
      if (viewer) {
        viewer->publish(runtime.last_state());
        if (viewer->closed()) break;
      }
      ++count;
      std::this_thread::sleep_until(next);
    }
    if (!runtime.stop()) std::cerr << "warning: final safe hold was not confirmed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Metatron native launch failed: " << error.what() << '\n';
    return 1;
  }
}
