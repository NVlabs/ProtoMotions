// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// One native 39-input launcher. Only RobotBackend differs across simulation
// and hardware; task goals, FK, history, TensorRT and safety are shared.
#include <mujoco/mujoco.h>
#include <nlohmann/json.hpp>
#include <net/if.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>

extern char** environ;  // for posix_spawn; declared by unistd.h only with _GNU_SOURCE

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "protomotions/deployment/controller_manager.hpp"
#include "protomotions/deployment/dex3_trigger_control.hpp"
#include "protomotions/deployment/evdev_xbox_input.hpp"
#include "protomotions/deployment/metatron_v1_arm_ik.hpp"
#include "protomotions/deployment/metatron_v1_controller.hpp"
#include "protomotions/deployment/metatron_v1_engine.hpp"
#include "protomotions/deployment/metatron_v1_fk_engine.hpp"
#include "protomotions/deployment/metatron_v1_manifest.hpp"
#include "protomotions/deployment/metatron_v1_task.hpp"
#include "protomotions/deployment/metatron_v1_teleop.hpp"
#include "protomotions/deployment/metrics_stream.hpp"
#include "protomotions/deployment/motion_library.hpp"
#include "metatron_v1_async_viewer.hpp"
#include "protomotions/deployment/mujoco_backend.hpp"
#include "protomotions/deployment/runtime.hpp"
#include "protomotions/deployment/teleop_settings.hpp"
#include "protomotions/deployment/unitree_dds_config.hpp"
#include "protomotions/deployment/unitree_sdk2_backend.hpp"
#include "protomotions/deployment/unitree_mode_voice.hpp"
#if PROTOMOTIONS_HAVE_PICO_INPUT
#include "protomotions/deployment/pico_input.hpp"
#endif

namespace {
using namespace protomotions::deployment;
using namespace protomotions::deployment::metatron_v1;
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t g_stop = 0;
void signal_stop(int) { g_stop = 1; }

constexpr float kPi = 3.14159265358979323846F;
constexpr float kControlDt = 0.02F;

struct SimPush {
  double time_s = 0.0;
  std::array<double, 3> force{};
  double duration_s = 0.1;
  bool applied = false;
};

// A scripted operator for simulation: what the sticks and the headset would
// have said at each moment, so the same loop runs without a human.
// The dashboard is a separate process on this machine: the metrics stream goes
// to it over loopback on a fixed port, and its settings (speed and turn caps,
// the selected motion) come back through a small JSON file it replaces
// atomically. The launcher starts the dashboard itself, so both sides agree on
// port and path without an operator typing anything.
constexpr const char* kDashboardHost = "127.0.0.1";
constexpr std::uint16_t kMetricsPort = 9870;

// Starts the dashboard when a person is at the controls and a display exists.
// Scripted runs (qualification grids) never get one. Failure to start is
// reported and otherwise ignored: the dashboard is a window, not a dependency.
pid_t launch_dashboard(const std::string& title, const std::filesystem::path& settings_file) {
  const char* display = std::getenv("DISPLAY");
  const char* wayland = std::getenv("WAYLAND_DISPLAY");
  if ((!display || !*display) && (!wayland || !*wayland)) {
    std::cout << "dashboard: no display, not started (run tools/live_metrics_viewer.py --udp "
              << kDashboardHost << ":" << kMetricsPort << " --settings-file " << settings_file.string()
              << " elsewhere)\n";
    return -1;
  }
  const std::string udp = std::string(kDashboardHost) + ":" + std::to_string(kMetricsPort);
  std::vector<std::string> args = {PROTOMOTIONS_DASHBOARD_PYTHON, PROTOMOTIONS_DASHBOARD_SCRIPT,
                                   "--udp", udp, "--settings-file", settings_file.string(), "--title", title};
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = -1;
  const int status = ::posix_spawn(&pid, PROTOMOTIONS_DASHBOARD_PYTHON, nullptr, nullptr, argv.data(), environ);
  if (status != 0) {
    std::cout << "dashboard: could not start " << PROTOMOTIONS_DASHBOARD_PYTHON << " (" << std::strerror(status) << ")\n";
    return -1;
  }
  std::cout << "dashboard: started (pid " << pid << ") on " << udp << ", settings in " << settings_file.string() << "\n";
  return pid;
}

struct ScriptSegment {
  double start_s = 0.0, end_s = 0.0;
  std::string mode = "idle";  // idle | steer | teleop | track
  float heading_rel = 0.0F;   // radians relative to the heading when the script starts
  std::array<float, 2> velocity{};  // heading-frame m/s
  std::optional<std::array<float, 3>> left, right;  // forward, lateral, height above floor
  std::optional<float> torso;  // commanded torso height above the floor
  std::string motion;          // track: name of the motion to play
};

struct Options {
  std::string backend = "mujoco";
  std::string xml, engine, fk_engine, manifest, recovery_plan, xbox_device, net_if;
  std::string glfw_library;
  std::string teleop_script, metrics_file;
  std::string motions;      // tracking motion bundle (web demo layout); empty = no tracking mode
  bool arm = false;
  bool viewer = false;
  bool scripted_sim = false;
  bool pico = false;
  bool hand_ik = true;
  int steps = 0;
  int dds_participant_index = 0;
  double sim_friction = 0.0, sim_hand_mass = 0.0;
  double operator_height = 0.0;  // standing headset height (m); 0 = trust the headset's floor origin
  std::array<double, 2> sim_odometry_drift{};
  std::vector<SimPush> sim_pushes;
};

SimPush parse_push(const std::string& text) {
  // "time,fx,fy,fz[,duration]" in seconds, newtons, seconds.
  SimPush push;
  std::vector<double> values;
  std::string item;
  for (char c : text + ",") {
    if (c == ',') { values.push_back(std::stod(item)); item.clear(); } else item += c;
  }
  if (values.size() != 4 && values.size() != 5)
    throw std::invalid_argument("--sim-push wants time,fx,fy,fz[,duration]");
  push.time_s = values[0];
  push.force = {values[1], values[2], values[3]};
  if (values.size() == 5) push.duration_s = values[4];
  return push;
}

Options parse(int argc, char** argv) {
  Options args;
  for (int index = 1; index < argc; ++index) {
    const std::string flag(argv[index]);
    if (flag == "--arm") args.arm = true;
    else if (flag == "--viewer") args.viewer = true;
    else if (flag == "--scripted-sim") args.scripted_sim = true;
    else if (flag == "--pico") args.pico = true;
    else if (flag == "--no-hand-ik") args.hand_ik = false;
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
      else if (flag == "--teleop-script") args.teleop_script = value;
      else if (flag == "--metrics-file") args.metrics_file = value;
      else if (flag == "--motions") args.motions = value;
      else if (flag == "--sim-friction") args.sim_friction = std::stod(value);
      else if (flag == "--sim-odometry-drift") {
        const auto comma = value.find(',');
        if (comma == std::string::npos) throw std::invalid_argument("--sim-odometry-drift wants vx,vy in m/s");
        args.sim_odometry_drift = {std::stod(value.substr(0, comma)), std::stod(value.substr(comma + 1))};
      }
      else if (flag == "--sim-hand-mass") args.sim_hand_mass = std::stod(value);
      else if (flag == "--operator-height") args.operator_height = std::stod(value);
      else if (flag == "--sim-push") args.sim_pushes.push_back(parse_push(value));
      else throw std::invalid_argument("unknown option " + flag);
    }
  }
  const bool scripted = args.scripted_sim || !args.teleop_script.empty();
  if ((args.backend != "mujoco" && args.backend != "unitree") ||
      args.xml.empty() ||
      args.engine.empty() || args.fk_engine.empty() ||
      args.manifest.empty() || args.recovery_plan.empty() ||
      (args.xbox_device.empty() && !scripted && !args.pico) || args.steps < 0)
    throw std::invalid_argument("required: --backend mujoco|unitree --xml --engine --fk-engine --manifest --recovery-plan and one of --xbox-device, --pico, --teleop-script (--xml is the robot model for arm kinematics; MuJoCo also steps it)");
  if (args.backend == "unitree" &&
      (!args.arm || args.net_if.empty() || if_nametoindex(args.net_if.c_str()) == 0))
    throw std::invalid_argument("hardware requires --arm and an existing --net-if");
  if (scripted && args.backend != "mujoco")
    throw std::invalid_argument("scripted input is simulation-only");
  if ((args.sim_friction > 0.0 || args.sim_hand_mass > 0.0 || !args.sim_pushes.empty() ||
       args.sim_odometry_drift[0] != 0.0 || args.sim_odometry_drift[1] != 0.0) &&
      args.backend != "mujoco")
    throw std::invalid_argument("--sim-friction, --sim-hand-mass, --sim-push and --sim-odometry-drift are simulation-only");
  if (args.viewer && (args.backend != "mujoco" || args.glfw_library.empty()))
    throw std::invalid_argument("--viewer needs --backend mujoco and --glfw-lib");
#if !PROTOMOTIONS_HAVE_PICO_INPUT
  if (args.pico) throw std::invalid_argument("--pico needs a build with PROTOMOTIONS_BUILD_PICO_INPUT");
#endif
  return args;
}

std::vector<ScriptSegment> load_script(const std::string& path) {
  const auto json = nlohmann::json::parse(std::ifstream(path));
  std::vector<ScriptSegment> segments;
  for (const auto& item : json.at("segments")) {
    ScriptSegment segment;
    segment.start_s = item.at("start").get<double>();
    segment.end_s = item.at("end").get<double>();
    segment.mode = item.value("mode", "idle");
    segment.heading_rel = item.value("heading_rel", 0.0F);
    if (item.contains("velocity")) segment.velocity = item.at("velocity").get<std::array<float, 2>>();
    if (item.contains("left")) segment.left = item.at("left").get<std::array<float, 3>>();
    if (item.contains("right")) segment.right = item.at("right").get<std::array<float, 3>>();
    if (item.contains("torso")) segment.torso = item.at("torso").get<float>();
    segment.motion = item.value("motion", std::string());
    if (segment.mode != "idle" && segment.mode != "steer" && segment.mode != "teleop" && segment.mode != "track")
      throw std::invalid_argument("teleop script mode must be idle, steer, teleop or track");
    if (segment.mode == "track" && segment.motion.empty())
      throw std::invalid_argument("a track segment names its motion");
    segments.push_back(segment);
  }
  return segments;
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

// Joint ranges in policy order from the robot MJCF, for the limit metrics.
std::vector<std::array<double, 2>> joint_ranges(const std::string& xml,
                                                const std::vector<std::string>& names) {
  std::vector<std::array<double, 2>> ranges;
  char error[1024]{};
  mjModel* model = mj_loadXML(xml.c_str(), nullptr, error, sizeof(error));
  if (!model) return ranges;
  for (const auto& name : names) {
    const int joint = mj_name2id(model, mjOBJ_JOINT, name.c_str());
    if (joint < 0 || !model->jnt_limited[joint]) ranges.push_back({-1e9, 1e9});
    else ranges.push_back({model->jnt_range[2 * joint], model->jnt_range[2 * joint + 1]});
  }
  mj_deleteModel(model);
  return ranges;
}

MetatronTask::HandTarget hand_target(const HandGoal& goal) {
  MetatronTask::HandTarget target;
  target.enabled = goal.enabled;
  target.rotation = goal.rotation;
  target.position = goal.position;
  target.rotation_xyzw = goal.rotation_xyzw;
  return target;
}

SupportFrame support_frame(const MetatronTask& task) {
  const auto support = task.hand_frame();
  SupportFrame frame;
  frame.center = support.center;
  frame.yaw = support.yaw;
  frame.valid = support.valid;
  return frame;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options args = parse(argc, argv);
    const auto json = nlohmann::json::parse(std::ifstream(args.manifest));
    const auto manifest = Manifest::load(args.manifest);
    // Identity comes from this artifact, not the launcher's executable name.
    // Existing exports have checkpoint metadata but no policy.name field.
    const std::string policy_name = [&]() -> std::string {
      const auto policy = json.find("policy");
      if (policy != json.end() && policy->is_object()) {
        const auto name = policy->find("name");
        if (name != policy->end() && name->is_string() && !name->get_ref<const std::string&>().empty())
          return name->get<std::string>();
      }
      const auto checkpoint = json.find("checkpoint");
      if (checkpoint != json.end() && checkpoint->is_string() && !checkpoint->get_ref<const std::string&>().empty()) {
        const std::filesystem::path path(checkpoint->get<std::string>());
        return (path.parent_path().filename() / path.filename()).string();
      }
      const auto digest = json.find("artifact_sha256");
      if (digest != json.end() && digest->is_string() && !digest->get_ref<const std::string&>().empty())
        return "sha256:" + digest->get<std::string>();
      return "manifest: " + std::filesystem::path(args.manifest).filename().string();
    }();
    const auto body_names = json.at("robot_contract").at("body_names").get<std::vector<std::string>>();
    const auto joint_names = json.at("robot_contract").at("joint_names").get<std::vector<std::string>>();
    // Joint order is the policy's and, on hardware, the motor order, so one
    // list names every per-joint and per-motor metric. It rides in every
    // frame so a viewer that attaches late can still name what it sees.
    std::string joint_names_csv;
    for (const auto& name : joint_names) joint_names_csv += (joint_names_csv.empty() ? "" : ",") + name;
    // Both engines always run as captured CUDA graphs: at batch one the policy
    // is launch-bound, and the graph takes about half a millisecond off every
    // cycle (measured 2.15 -> 1.68 ms for the fp32 ep6410 plan on an RTX 5090).
    FkEngine fk(args.fk_engine, /*cuda_graph=*/true);
    Engine engine(args.engine, /*cuda_graph=*/true);
    if (!fk.cuda_graph_enabled() || !engine.cuda_graph_enabled())
      throw std::runtime_error("CUDA graphs did not capture for both engines");
    RestPose rest = make_rest_pose(fk);
    std::optional<MotionLibrary> motions;
    if (!args.motions.empty()) {
      motions.emplace(args.motions);
      if (motions->body_count() != body_names.size())
        throw std::runtime_error("tracking motions describe a different body count than the policy");
    }
    MetatronTask task(fk, rest, args.recovery_plan, body_names, motions ? &*motions : nullptr);
    engine.warmup(10);
    std::unique_ptr<InputSource> operator_input;
    if (!args.xbox_device.empty())
      operator_input = std::make_unique<EvdevXboxInputSource>(args.xbox_device);
#if PROTOMOTIONS_HAVE_PICO_INPUT
    if (args.pico) operator_input = std::make_unique<PicoInputSource>();
#endif
    std::vector<ScriptSegment> script;
    if (!args.teleop_script.empty()) script = load_script(args.teleop_script);
    std::unique_ptr<AsyncMujocoViewer> viewer;
    if (args.viewer)
      viewer = std::make_unique<AsyncMujocoViewer>(args.xml, args.glfw_library);
    std::unique_ptr<RobotBackend> backend;
    UnitreeSdk2Backend* unitree_backend = nullptr;
    MuJoCoBackend* mujoco_backend = nullptr;
    if (args.backend == "mujoco") {
      MuJoCoBackendConfig config;
      config.xml_path = args.xml;
      config.policy_anchor_body_name = "torso_link";
      config.substeps_per_command = sim_substeps_and_validate(args.xml, json);
      config.initial_dof_position = standing_pose();
      config.floor_friction = args.sim_friction;
      config.odometry_drift_mps = args.sim_odometry_drift;
      if (args.sim_hand_mass > 0.0) {
        config.body_mass_additions = {{"left_rubber_hand", args.sim_hand_mass},
                                      {"right_rubber_hand", args.sim_hand_mass}};
      }
      auto sim = std::make_unique<MuJoCoBackend>(std::move(config));
      mujoco_backend = sim.get();
      backend = std::move(sim);
    } else {
      configure_unitree_dds(args.net_if, args.dds_participant_index);
      UnitreeSdk2BackendConfig config;
      config.network_interface = args.net_if;
      config.enable_odometry = true;
      config.max_odometry_age = std::chrono::milliseconds(100);
      config.use_lowstate_imu = true;
      config.imu_is_torso = false;
      config.enable_dex3_hands = args.pico;
      config.damp_unused_dex3_hands = !args.pico;
      config.root_position_without_odometry = {0.0, 0.0, 0.793};
      auto sdk = std::make_unique<UnitreeSdk2Backend>(std::move(config));
      unitree_backend = sdk.get();
      backend = std::move(sdk);
    }
    // Arm IK is optional so policy-only hand tracking can be compared against
    // the bounded residual. The MJCF is used for arm kinematics, not physics.
    std::unique_ptr<ArmIkCorrection> arm_ik;
    if (args.hand_ik) {
      std::array<float, kDofCount> scale{}, offset{};
      for (std::size_t j = 0; j < kDofCount; ++j) {
        // Recover the affine map from the manifest JSON: it is the policy's own.
        scale[j] = json.at("action_map").at("scale")[j].get<float>();
        offset[j] = json.at("action_map").at("offset")[j].get<float>();
      }
      arm_ik = std::make_unique<ArmIkCorrection>(args.xml, joint_names, scale, offset, ArmIkSettings{});
    }
    ControllerManager controllers;
    // Stage timings for the dashboard: the policy engine alone, and the
    // observation build (task goals, sensor FK engine, history) that feeds it.
    float observe_ms = 0.0F, policy_ms = 0.0F;
    auto controller = std::make_unique<MetatronController>(
        manifest.action_map(),
        [&task, &observe_ms](Inputs& inputs, const RobotState& state) {
          const auto begin = Clock::now();
          task.build(inputs, state);
          observe_ms = std::chrono::duration<float, std::milli>(Clock::now() - begin).count();
        },
        [&engine, &policy_ms](const Inputs& inputs) {
          const auto begin = Clock::now();
          auto outputs = engine.run(inputs);
          policy_ms = std::chrono::duration<float, std::milli>(Clock::now() - begin).count();
          return outputs;
        });
    if (arm_ik) {
      controller->set_refiner([&arm_ik](Outputs& outputs, const RobotState& state) {
        arm_ik->apply(outputs, state, kControlDt);
      });
    }
    if (!controllers.add(std::move(controller)))
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
    // Speech is asynchronous and only announces active, successfully commanded
    // modes. Construction after runtime startup also ensures DDS is initialized.
    std::unique_ptr<UnitreeModeVoice> mode_voice;
    if (unitree_backend) mode_voice = std::make_unique<UnitreeModeVoice>();
    std::signal(SIGINT, signal_stop);
    std::signal(SIGTERM, signal_stop);

    MetricsPublisher metrics;
    metrics.open_udp(kDashboardHost, kMetricsPort);  // non-blocking; nobody listening costs nothing
    if (!args.metrics_file.empty()) metrics.open_file(args.metrics_file);
    const auto limits_table = args.xml.empty() ? std::vector<std::array<double, 2>>{}
                                               : joint_ranges(args.xml, joint_names);
    MetricsFrame frame;
    // Operator settings from the dashboard: polled by modification time once a
    // cycle, parsed only on change, capped at the launcher's ceilings, and a
    // malformed file leaves the current settings alone. Nothing in it starts,
    // stops or switches a motion: it names what the operator's button would play.
    TeleopSettingsFile settings_file(default_teleop_settings_path(), MetatronTask::settings_schema());
    TeleopSettings settings = defaults_from(settings_file.schema());
    {
      // A launch starts from the task's defaults. The file outlives a dashboard
      // restart on purpose, but a cap pulled past the qualified value in an
      // earlier session must not be waiting for the next operator.
      std::error_code ignored;
      std::filesystem::remove(settings_file.path(), ignored);
    }
    const std::string settings_schema = schema_json(settings_file.schema());
    std::string settings_error;
    // Only a person at the controls gets a window; scripts run blind.
    const pid_t dashboard = operator_input ? launch_dashboard(policy_name, settings_file.path()) : -1;
    std::optional<std::size_t> selected_motion;
    const ScriptSegment* script_track_segment = nullptr;
    if (motions && motions->motion_count() > 0) selected_motion = 0;
    std::string available_motions;
    if (motions)
      for (const auto& name : motions->names()) available_motions += (available_motions.empty() ? "" : ",") + name;
    TeleopHandMapping mapping;
    OperatorHeight operator_height;  // learned from the headset: people stand tallest
    // Headset origin to operator floor, set on the B press that turns teleop on
    // when --operator-height is given; zero trusts the headset's own floor.
    float pico_floor_offset = 0.0F;
    // Set on the first planted cycle after teleop turns on, so the robot's hands
    // stay where they are until the operator moves.
    HandAnchor hand_anchor;
    PicoHandClutch hand_clutch;
    bool hand_anchor_pending = false;
    const float robot_upright = rest.upright_height;

    std::cout << "Metatron v1 ready; X=damping, Y=revive idle, A=steering, B=hands (teleop), RB=idle. "
                 "Starts in safe hold. backend=" << args.backend
              << " hand_ik=" << (args.hand_ik ? "on" : "off")
              << (args.pico ? " Pico triggers=matching Dex-3 fingers (hold to close, release to open); grips=arm clutch." : std::string())
              << (motions ? " LB=track selected motion, D-pad left/right=select (" + available_motions + ")" : std::string())
              << "\n";
    std::uint32_t previous = 0;
    // PROTOMOTIONS_INPUT_TRACE=1 logs every button change: the way to tell a
    // press the controller never delivered from one the launcher ignored.
    const bool trace_input = std::getenv("PROTOMOTIONS_INPUT_TRACE") != nullptr;
    bool active = false, steering = false, teleop = false;
    Dex3TriggerControl fingers;
    std::array<bool, 2> last_fingers{};
    bool hand_position_mode = false;
    bool hand_transport_ready = false;
    auto last_hand_send = Clock::time_point::min();
    auto last_thumb_log = Clock::time_point::min();
    float heading_goal = 0.0F;
    std::array<float, 2> commanded_velocity{};
    float script_heading = 0.0F;
    bool script_heading_ready = false;
    std::vector<SimPush> pushes = args.sim_pushes;
    double energy_joules = 0.0;
    std::vector<double> previous_velocity, previous_acceleration;  // for joint jerk
    std::uint64_t missed_states = 0, previous_sequence = 0;
    auto next = Clock::now();
    const auto started = next;
    int count = 0;
    while (!g_stop && (args.steps == 0 || count < args.steps)) {
      next += std::chrono::milliseconds(20);
      const double time_s = count * static_cast<double>(kControlDt);
      InputState input;
      bool xbox_kick_triggered = false;
      bool input_ready = false;
      std::array<HandGoal, 2> hand_goals{};
      bool script_hands = false;
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
      } else if (!script.empty()) {
        // The script speaks in task terms; translate to the same button and
        // stick semantics the operator path uses, so one control path is tested.
        input.connected = true;
        input.timestamp = Clock::now();
        input_ready = true;
        const ScriptSegment* segment = nullptr;
        for (const auto& candidate : script)
          if (time_s >= candidate.start_s && time_s < candidate.end_s) { segment = &candidate; break; }
        if (count == 5) input.buttons |= xbox_button::kY;  // revive into idle once
        if (segment && active && segment->mode == "track" && motions) {
          // The script selects the motion the way the UI would, then presses LB
          // once per segment; a motion that ends early leaves the robot idle.
          const auto index = motions->index_of(segment->motion);
          if (!index) throw std::runtime_error("teleop script names an unknown motion: " + segment->motion);
          if (script_track_segment != segment) {
            script_track_segment = segment;
            selected_motion = index;
            input.buttons |= xbox_button::kLeftBumper;
          }
        } else if (segment && active) {
          const bool wants_motion = segment->mode != "idle";
          if (task.mode() == MetatronTask::Mode::kTracking) input.buttons |= xbox_button::kLeftBumper;  // stop
          if (wants_motion && !steering) input.buttons |= xbox_button::kA;
          if (!wants_motion && steering) input.buttons |= xbox_button::kRightBumper;
          if (segment->mode == "teleop" && !teleop) input.buttons |= xbox_button::kB;
          if (segment->mode != "teleop" && teleop) input.buttons |= xbox_button::kB;
          if (wants_motion) {
            if (!script_heading_ready) {
              script_heading = yaw(runtime.last_state().base_orientation_xyzw);
              script_heading_ready = true;
            }
            heading_goal = wrap(script_heading + segment->heading_rel);
            // Sticks are read below; feed the velocity directly instead.
            input.axes[2] = -segment->velocity[1];
            input.axes[3] = segment->velocity[0];
          }
          if (segment->mode == "teleop") {
            script_hands = true;
            task.set_torso_height(segment->torso.has_value(), segment->torso.value_or(0.0F));
            const auto support = support_frame(task);
            for (int side = 0; side < 2; ++side) {
              const auto& spec = side == 0 ? segment->left : segment->right;
              if (!spec || !support.valid) continue;
              hand_goals[side].enabled = true;
              hand_goals[side].translation = true;
              hand_goals[side].position = hand_goal_from_operator(
                  support, {(*spec)[0], (*spec)[1], 0.0F}, (*spec)[2], mapping);
            }
          }
        }
      } else if (operator_input) {
        input_ready = operator_input->poll(input);
      }
      if (operator_input) {
        std::string error;
        if (const auto next = settings_file.poll(settings, error)) {
          if (next->values != settings.values) {
            std::cout << "settings v" << next->version << ":";
            for (const auto& range : settings_file.schema())
              std::cout << " " << range.key << "=" << next->get(range.key) << " " << range.unit;
            std::cout << "\n";
          }
          if (motions && !next->selected_motion.empty()) {
            const auto index = motions->index_of(next->selected_motion);
            if (index && selected_motion != index) {
              // A selection is only that: the controller's button plays it. A
              // motion already playing is never interrupted by the dashboard.
              selected_motion = index;
              std::cout << "settings: motion " << motions->names()[*index] << " selected\n";
            }
          }
          settings = *next;
          settings_error.clear();
        } else if (!error.empty() && error != settings_error) {
          settings_error = error;
          std::cout << "settings: " << error << "; keeping the current settings\n";
        }
      }
      if (!input_ready ||
          Clock::now() - input.timestamp > std::chrono::milliseconds(250)) {
        runtime.emergency_stop();
        active = false;
      } else {
        // Sources that see every event latch taps in `pressed`; the snapshot
        // edge still serves sources that only report state.
        const auto pressed = input.pressed | (input.buttons & ~previous);
        if (trace_input && (pressed || input.buttons != previous))
          std::cout << "input: buttons=0x" << std::hex << input.buttons << " pressed=0x" << pressed << std::dec
                    << " seq=" << input.sequence << " active=" << active << " mode=" << task.mode_name() << "\n";
        previous = input.buttons;
        // A motion that finished on its own left the task idle; keep the
        // launcher's view of the mode in step with it.
        if (active && task.mode() == MetatronTask::Mode::kIdle) steering = teleop = false;
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
            hand_clutch.reset();
            if (arm_ik) arm_ik->reset();
            if (!runtime.request_controller("metatron_v1"))
              throw std::runtime_error("scene controller selection rejected");
            active = true;
            steering = false;
            teleop = false;
            std::cout << "Y: idle requested\n";
          } else {
            std::cout << "Y: already active; control unchanged\n";
          }
        } else if (active && (pressed & xbox_button::kRightBumper)) {
          steering = false;
          teleop = false;
          task.set_idle();
          std::cout << "RB: idle\n";
        } else if (active && (pressed & xbox_button::kA)) {
          steering = true;
          teleop = false;
          heading_goal = yaw(runtime.last_state().base_orientation_xyzw);
          std::cout << "A: steering\n";
        } else if (active && motions && (pressed & xbox_button::kLeftBumper)) {
          if (task.mode() == MetatronTask::Mode::kTracking) {
            task.set_idle();
            std::cout << "LB: tracking stopped\n";
          } else if (selected_motion) {
            steering = teleop = false;
            if (arm_ik) arm_ik->reset();
            task.start_tracking(*selected_motion);
            std::cout << "LB: tracking " << motions->names()[*selected_motion] << "\n";
            xbox_kick_triggered = !args.xbox_device.empty() &&
                motions->names()[*selected_motion] == "kick";
          }
        } else if (active && motions && selected_motion && (pressed & (xbox_button::kDpadLeft | xbox_button::kDpadRight))) {
          const auto n = motions->motion_count();
          selected_motion = (pressed & xbox_button::kDpadRight) ? (*selected_motion + 1) % n : (*selected_motion + n - 1) % n;
          std::cout << "D-pad: motion " << motions->names()[*selected_motion] << " selected\n";
        }
        if (active && steering && (pressed & xbox_button::kB)) {
          teleop = !teleop;
          if (!teleop && arm_ik) arm_ik->reset();
          std::cout << (teleop ? "B: hand teleop on\n" : "B: hand teleop off\n");
          hand_anchor = {};
          hand_clutch.reset();
          hand_anchor_pending = teleop && input.has_pico;
          if (teleop && args.operator_height > 0.0 && input.has_pico && input.pico.heading_valid) {
            // The operator stands upright to press B, so this reading is their
            // standing headset height in headset coordinates.
            pico_floor_offset = input.pico.headset_height - static_cast<float>(args.operator_height);
            operator_height.standing = static_cast<float>(args.operator_height);
            std::cout << "pico: floor set " << pico_floor_offset << " m below the headset origin\n";
          }
        }
        if (active && steering) {
          if (script.empty()) {
            const float lx = input.axes[0], ly = input.axes[1];
            const float magnitude = std::min(1.0F, std::hypot(lx, ly));
            if (magnitude > 0.15F) {
              const float amount = (magnitude - 0.15F) / 0.85F;
              const float angle = std::atan2(-lx, ly) * amount;
              const float quantum = kPi / 18.0F;
              const float desired = wrap(yaw(runtime.last_state().base_orientation_xyzw) +
                                         std::round(angle / quantum) * quantum);
              // The goal turns toward where the stick points no faster than the
              // yaw-rate cap, so the cap bounds how fast the robot is asked to
              // turn; a goal that jumped would leave the turn rate to the policy.
              const float step = settings.get("max_yaw_rate_rps") * kControlDt;
              heading_goal = wrap(heading_goal + std::clamp(wrap(desired - heading_goal), -step, step));
            }
          }
          const float rx = input.axes[2], ry = input.axes[3];
          const float level = std::min(1.0F, std::hypot(rx, ry));
          const float speed = (level <= 0.15F ? 0.0F : std::min(1.0F, (level - 0.15F) / 0.85F)) *
                              settings.get("max_speed_mps");
          const float direction = heading_goal + std::atan2(-rx, ry);
          const std::array<float, 2> requested_velocity = {speed * std::cos(direction), speed * std::sin(direction)};
          // A full stick deflection should not make the gait targets jump to
          // full speed in a single policy step. Scripts remain exact fixtures.
          const std::array<float, 2> velocity = script.empty()
              ? slew_velocity_xy(commanded_velocity, requested_velocity, 2.0F * kControlDt)
              : requested_velocity;
          commanded_velocity = velocity;
          if (task.mode() == MetatronTask::Mode::kTracking) {
            // The motion owns the goals; sticks and hands wait until it ends.
          } else if (teleop) {
            if (!script_hands && input.has_pico) {
              const PicoInputState pico = level_to_floor(input.pico, pico_floor_offset);
              const SupportFrame support = support_frame(task);
              if (hand_anchor_pending && support.valid && pico.pose_valid && task.measured_hands_ready()) {
                hand_anchor = anchor_hands(support, pico, task.measured_hands(), task.measured_hand_rotations(),
                                           mapping);
                hand_anchor_pending = false;
                std::cout << "pico: hands anchored to the robot's current pose\n";
              }
              // Until anchored the hands keep their pose rather than jumping to
              // an absolute mapping the operator did not ask for.
              hand_goals = hand_anchor_pending ? std::array<HandGoal, 2>{}
                                               : hand_clutch.update(support, pico, mapping, hand_anchor);
              if (pico.pose_valid && pico.heading_valid) {
                // The operator's crouch comes from the headset, not from where
                // the hands are: squatting with hands at the hips should lower
                // the robot too.
                operator_height.observe(pico.headset_height);
                task.set_torso_height(operator_height.valid(),
                    torso_height_from_operator(pico.headset_height, operator_height, robot_upright));
              }
            }
            task.set_teleop(heading_goal, velocity, hand_target(hand_goals[0]), hand_target(hand_goals[1]));
          } else {
            task.set_steering(heading_goal, velocity);
          }
          if (arm_ik) {
            // The residual follows the same goals the policy sees, and only on
            // planted feet, where the task applies them.
            const bool planted = task.metrics().stationary && task.support().valid;
            for (int side = 0; side < 2; ++side) {
              HandGoal goal = hand_goals[side];
              goal.enabled = goal.enabled && teleop && planted;
              arm_ik->set_goal(side, goal);
            }
          }
        }
        if (!(active && steering)) commanded_velocity = {};
      }
      if (mujoco_backend) {
        for (auto& push : pushes) {
          if (!push.applied && time_s >= push.time_s) {
            push.applied = mujoco_backend->apply_push("torso_link", push.force, push.duration_s);
            std::cout << "sim push at " << time_s << " s: " << push.force[0] << "," << push.force[1]
                      << "," << push.force[2] << " N for " << push.duration_s << " s\n";
          }
        }
      }
      const auto cycle_start = Clock::now();
      const auto result = runtime.run_cycle(cycle_start);
      if (unitree_backend && args.pico) {
        const bool ready = unitree_backend->dex3_hands_ready();
        if (ready != hand_transport_ready) {
          std::cout << "Pico Dex-3: " << unitree_backend->dex3_hand_status()
                    << (ready ? "; finger triggers enabled in Pico steering/teleop\n"
                              : "; finger commands suspended\n");
          hand_transport_ready = ready;
        }
        // An explicit Y revive opens the hands even in idle. Startup, X and
        // faults remain damping-only; triggers are accepted only with a fresh
        // Pico pose in steering/teleop.
        const bool hands_active = ready && input_ready && input.connected &&
            active && result == CycleResult::kSent &&
            (!steering || (input.has_pico && input.pico.pose_valid));
        const bool refresh_due = last_hand_send == Clock::time_point::min() ||
            cycle_start - last_hand_send >= std::chrono::milliseconds(100);
        if (hands_active) {
          if (steering)
            fingers.update(input.pico.left_trigger, input.pico.right_trigger);
          else
            fingers.reset();
          if (!hand_position_mode || fingers.closed != last_fingers || refresh_due) {
            if (unitree_backend->send_dex3_position_command(
                    fingers.target(0), fingers.target(1))) {
              if (!hand_position_mode || fingers.closed != last_fingers)
                std::cout << "Pico Dex-3: left=" << (fingers.closed[0] ? "closed" : "open")
                          << " right=" << (fingers.closed[1] ? "closed" : "open") << '\n';
              hand_position_mode = true;
              last_fingers = fingers.closed;
              last_hand_send = cycle_start;
            } else {
              hand_position_mode = false;
            }
          }
          if (last_thumb_log == Clock::time_point::min() ||
              cycle_start - last_thumb_log >= std::chrono::seconds(1)) {
            if (const auto measured = unitree_backend->dex3_right_hand_position()) {
              const auto& target = fingers.target(1);
              std::cout << "Pico right thumb target=[" << target[0] << ','
                        << target[1] << ',' << target[2] << "] measured=["
                        << (*measured)[0] << ',' << (*measured)[1] << ','
                        << (*measured)[2] << "]\n";
            }
            last_thumb_log = cycle_start;
          }
        } else {
          if (ready && (hand_position_mode || refresh_due)) {
            if (unitree_backend->send_dex3_damping_hold())
              last_hand_send = cycle_start;
          }
          hand_position_mode = false;
          fingers.reset();
        }
      }
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
      if (mode_voice) {
        VoiceMode spoken = VoiceMode::kDamping;
        if (active && result == CycleResult::kSent) {
          switch (task.mode()) {
            case MetatronTask::Mode::kIdle: spoken = VoiceMode::kIdle; break;
            case MetatronTask::Mode::kSteering: spoken = VoiceMode::kSteering; break;
            case MetatronTask::Mode::kTeleop: spoken = VoiceMode::kTeleop; break;
            case MetatronTask::Mode::kTracking: spoken = VoiceMode::kTracking; break;
          }
        }
        mode_voice->update(spoken);
        if (xbox_kick_triggered && active && result == CycleResult::kSent)
          mode_voice->kick();
      }
      if (metrics.enabled()) {
        const auto& state = runtime.last_state();
        const auto now = Clock::now();
        frame.reset(static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(now - started).count()),
                    static_cast<std::uint64_t>(count));
        frame.set_text("policy/name", policy_name);
        frame.set_text("joint/names", joint_names_csv);
        frame.set("cycle/result", static_cast<float>(result));
        frame.set("cycle/compute_ms", std::chrono::duration<float, std::milli>(now - cycle_start).count());
        {
          // Where the cycle went: sensor read, observation build, policy engine,
          // and the command send (for MuJoCo that send steps the physics).
          const auto& timing = runtime.last_cycle_timing();
          if (timing.valid()) {
            const auto ms = [](auto duration) {
              return std::chrono::duration<float, std::milli>(duration).count();
            };
            frame.set("cycle/read_ms", ms(timing.state_ready - cycle_start));
            frame.set("cycle/observe_ms", observe_ms);
            frame.set("cycle/policy_ms", policy_ms);
            frame.set("cycle/send_ms", ms(timing.send_done - timing.command_ready));
          }
        }
        frame.set("cycle/state_age_ms", std::chrono::duration<float, std::milli>(now - state.timestamp).count());
        frame.set("cycle/held", static_cast<float>(runtime.held_cycle_count()));
        if (previous_sequence != 0 && state.sequence > previous_sequence + 1)
          missed_states += state.sequence - previous_sequence - 1;
        previous_sequence = state.sequence;
        frame.set("comm/missed_states", static_cast<float>(missed_states));
        if (unitree_backend) {
          const auto s = unitree_backend->sensor_status();
          frame.set("comm/lowstate_packets", static_cast<float>(s.lowstate_packets));
          frame.set("comm/crc_errors", static_cast<float>(s.lowstate_crc_errors));
          frame.set("comm/lowstate_age_ms", static_cast<float>(s.lowstate_age_ms));
          frame.set("comm/odometry_age_ms", static_cast<float>(s.odometry_age_ms));
        }
        frame.set_vector("joint/q", state.dof_position);
        frame.set_vector("joint/dq", state.dof_velocity);
        if (!state.dof_torque.empty()) {
          frame.set_vector("joint/torque", state.dof_torque);
          double power = 0.0;
          for (std::size_t j = 0; j < state.dof_torque.size() && j < state.dof_velocity.size(); ++j)
            power += std::abs(state.dof_torque[j] * state.dof_velocity[j]);
          energy_joules += power * kControlDt;
          frame.set("power/mechanical_w", static_cast<float>(power));
          frame.set("power/energy_j", static_cast<float>(energy_joules));
          // Jerk of the measured joints, the third derivative of position by
          // finite differences of the reported velocity: what a jittery joint,
          // a noisy action or a hard contact looks like in one number per joint.
          const auto& velocity = state.dof_velocity;
          if (previous_velocity.size() == velocity.size() && previous_acceleration.size() == velocity.size()) {
            double sum_squares = 0.0;
            for (std::size_t j = 0; j < velocity.size(); ++j) {
              const double acceleration = (velocity[j] - previous_velocity[j]) / kControlDt;
              const double jerk = (acceleration - previous_acceleration[j]) / kControlDt;
              previous_acceleration[j] = acceleration;
              frame.set("joint/jerk/" + std::to_string(j), static_cast<float>(jerk));
              sum_squares += jerk * jerk;
            }
            frame.set("joint/jerk_rms", static_cast<float>(std::sqrt(sum_squares / static_cast<double>(velocity.size()))));
          } else if (previous_velocity.size() == velocity.size()) {
            previous_acceleration.resize(velocity.size());
            for (std::size_t j = 0; j < velocity.size(); ++j)
              previous_acceleration[j] = (velocity[j] - previous_velocity[j]) / kControlDt;
          }
          previous_velocity.assign(velocity.begin(), velocity.end());
        }
        if (!limits_table.empty() && state.dof_position.size() == limits_table.size()) {
          int hits = 0;
          double min_margin = std::numeric_limits<double>::infinity();
          std::size_t nearest = 0;
          for (std::size_t j = 0; j < limits_table.size(); ++j) {
            const double margin = std::min(state.dof_position[j] - limits_table[j][0],
                                           limits_table[j][1] - state.dof_position[j]);
            if (margin < min_margin) {
              min_margin = margin;
              nearest = j;
            }
            if (margin < 0.05) ++hits;  // within ~3 degrees of a hard limit
          }
          frame.set("limits/joints_near_limit", static_cast<float>(hits));
          frame.set("limits/min_margin_rad", static_cast<float>(min_margin));
          if (nearest < joint_names.size()) frame.set_text("limits/nearest_joint", joint_names[nearest]);
        }
        // Motor temperatures come only from hardware telemetry (Unitree
        // LowState); simulation reports none, and nothing is invented for it.
        // Every reading that exists is emitted (non-finite ones as null) so a
        // hot motor stays visible even when another sensor has dropped out;
        // temperature_valid says whether the set is complete. The viewer draws
        // caution bands from these keys; they are display aids, not the
        // launcher's stop thresholds.
        // A simulation without a temperature sensor says nothing, so the viewer
        // shows no thermal tab; hardware always reports the family, even with no
        // reading, so a dead sensor shows as unavailable rather than unsupported.
        if (const auto& temps = state.motor_temperature; args.backend == "unitree" || !temps.empty()) {
          const float nan = std::numeric_limits<float>::quiet_NaN();
          bool complete = !temps.empty() && temps.size() == state.dof_position.size();
          float max_winding = -std::numeric_limits<float>::infinity();
          float max_driver = -std::numeric_limits<float>::infinity();
          for (std::size_t motor = 0; motor < temps.size(); ++motor) {
            const double winding = temps[motor][0], driver = temps[motor][1];
            const bool winding_ok = std::isfinite(winding), driver_ok = std::isfinite(driver);
            if (!winding_ok || !driver_ok) complete = false;
            frame.set("motor/winding_c/" + std::to_string(motor), winding_ok ? static_cast<float>(winding) : nan);
            frame.set("motor/driver_c/" + std::to_string(motor), driver_ok ? static_cast<float>(driver) : nan);
            if (winding_ok) max_winding = std::max(max_winding, static_cast<float>(winding));
            if (driver_ok) max_driver = std::max(max_driver, static_cast<float>(driver));
          }
          frame.set("motor/temperature_valid", complete ? 1.0F : 0.0F);
          frame.set("motor/temperature_count", static_cast<float>(temps.size()));
          if (std::isfinite(max_winding)) frame.set("motor/max_winding_c", max_winding);
          if (std::isfinite(max_driver)) frame.set("motor/max_driver_c", max_driver);
        }
        frame.set_vector("pose/root", state.root_position);
        frame.set_vector("pose/root_quat", state.root_orientation_xyzw);
        frame.set("pose/torso_z", static_cast<float>(state.anchor_position[2]));
        frame.set("pose/tilt_deg", static_cast<float>(
            std::acos(std::clamp(-state.projected_gravity[2], -1.0, 1.0)) * 180.0 / kPi));
        const auto& tm = task.metrics();
        frame.set("task/mode", static_cast<float>(task.mode()));
        // The viewer titles the task tab and the mode readout from these, so
        // another task's launcher names itself here and needs no viewer change.
        frame.set_text("task/name", "Metatron v1");
        frame.set_text("task/mode_name", task.mode_name());
        frame.set("task/active", active ? 1.0F : 0.0F);
        frame.set("task/recovery_phase", static_cast<float>(task.recovery_phase()));
        frame.set("task/heading_error_deg", tm.heading_error_rad * 180.0F / kPi);
        frame.set("task/speed_requested", tm.requested_speed_mps);
        frame.set("task/speed_measured", tm.measured_speed_mps);
        frame.set("task/velocity_error_x", tm.velocity_error_mps[0]);
        frame.set("task/velocity_error_y", tm.velocity_error_mps[1]);
        frame.set("task/support_slip_m", tm.support_slip_m);
        frame.set("task/replants", static_cast<float>(tm.replants));
        frame.set("task/torso_height_goal_m", tm.torso_height_goal_m);
        frame.set("task/torso_height_m", tm.torso_height_m);
        frame.set("task/prompt_bias_left_m", tm.prompt_bias_m[0]);
        frame.set("task/prompt_bias_right_m", tm.prompt_bias_m[1]);
        if (motions) {
          frame.set_text("tracking/available", available_motions);
          frame.set_text("tracking/selected", selected_motion ? motions->names()[*selected_motion] : std::string());
          const auto* player = task.tracking();
          const bool playing = player && player->active();
          frame.set_text("tracking/motion", playing ? motions->names()[player->motion()] : std::string());
          frame.set("tracking/elapsed_s", tm.tracking_elapsed_s);
          frame.set("tracking/duration_s", tm.tracking_duration_s);
          frame.set("tracking/progress", tm.tracking_duration_s > 0.0F ? tm.tracking_elapsed_s / tm.tracking_duration_s : 0.0F);
          frame.set("tracking/mean_body_error_m", playing ? tm.tracking.mean_body_error_m : std::numeric_limits<float>::quiet_NaN());
          frame.set("tracking/anchor_height_error_m", playing ? tm.tracking.anchor_height_error_m : std::numeric_limits<float>::quiet_NaN());
          frame.set("tracking/mean_rotation_error_deg", playing ? tm.tracking.mean_rotation_error_deg : std::numeric_limits<float>::quiet_NaN());
          if (playing) {
            // Reference positions of the conditionable bodies, for the replay renderer.
            for (int c = 0; c < 5; ++c)
              frame.set_vector("tracking/ref/" + std::to_string(c),
                               std::array<float, 3>{tm.tracking_reference[c * 3], tm.tracking_reference[c * 3 + 1],
                                                    tm.tracking_reference[c * 3 + 2]});
          }
        }
        frame.set("task/hand_error_left_m", tm.hand_error_m[0]);
        frame.set("task/hand_error_right_m", tm.hand_error_m[1]);
        if (operator_input) {
          // What the launcher is actually running with, so the dashboard shows
          // applied values, not the ones it last wrote.
          frame.set("settings/version", static_cast<float>(settings.version));
          frame.set_text("settings/schema", settings_schema);
          for (const auto& range : settings_file.schema()) frame.set("settings/" + range.key, settings.get(range.key));
          frame.set_text("settings/error", settings_error);
        }
        const auto support = task.support();
        frame.set("support/valid", support.valid ? 1.0F : 0.0F);
        frame.set_vector("support/center", support.center);
        frame.set("support/yaw", support.yaw);
        for (int side = 0; side < 2; ++side) {
          const std::string name = side == 0 ? "goal/left" : "goal/right";
          // The flag is written in both states so a viewer can plot it; the
          // position only while it means something.
          frame.set(name + "/enabled", hand_goals[side].enabled ? 1.0F : 0.0F);
          if (hand_goals[side].enabled) frame.set_vector(name, hand_goals[side].position);
        }
        if (arm_ik) {
          const auto& diag = arm_ik->diagnostics();
          frame.set("ik/offset_left_rad", diag.offset_norm_rad[0]);
          frame.set("ik/offset_right_rad", diag.offset_norm_rad[1]);
          frame.set("ik/solve_ms", diag.solve_milliseconds);
        }
        metrics.publish(frame);
      }
      if ((args.scripted_sim || !script.empty()) && count % 25 == 0) {
        const auto& state = runtime.last_state();
        std::cout << "sim cycle=" << count << " result=" << static_cast<int>(result)
                  << " root_z=" << state.root_position[2]
                  << " recovery=" << static_cast<int>(task.recovery_phase());
        if (teleop) std::cout << " hand_err=" << task.metrics().hand_error_m[0] << "/" << task.metrics().hand_error_m[1];
        std::cout << '\n';
      }
      if (viewer) {
        viewer->publish(runtime.last_state());
        if (viewer->closed()) break;
      }
      ++count;
      std::this_thread::sleep_until(next);
    }
    if (unitree_backend && args.pico && unitree_backend->dex3_hands_ready() &&
        !unitree_backend->send_dex3_damping_hold())
      std::cerr << "warning: final Dex-3 damping hold was not confirmed\n";
    if (!runtime.stop()) std::cerr << "warning: final safe hold was not confirmed\n";
    metrics.close();
    if (dashboard > 0) {
      ::kill(dashboard, SIGTERM);
      ::waitpid(dashboard, nullptr, 0);
    }
    if (metrics.dropped() > 0)
      std::cerr << "metrics: " << metrics.udp_dropped() << " UDP frames and " << metrics.file_dropped()
                << " file frames dropped (telemetry only; the control loop never waits on them)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Metatron native launch failed: " << error.what() << '\n';
    return 1;
  }
}
