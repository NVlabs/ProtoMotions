// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "protomotions/deployment/metatron_v1_task.hpp"
#include "protomotions/deployment/teleop_settings.hpp"

using namespace protomotions::deployment;

namespace {

void write(const std::filesystem::path& path, const std::string& text) {
  // Write-then-rename, as the dashboard does; the mtime moves with the rename.
  const auto temporary = path.string() + ".tmp";
  {
    std::ofstream file(temporary);
    file << text;
  }
  std::filesystem::rename(temporary, path);
}

}  // namespace

int main() {
  const TeleopSettingsSchema schema = {{"max_speed_mps", "Max speed", "m/s", 1.0F, 1.0F, 2.0F},
                                       {"max_yaw_rate_rps", "Max yaw rate", "rad/s", 3.0F, 3.0F, 6.0F}};
  TeleopSettings current = defaults_from(schema);
  assert(current.get("max_speed_mps") == 1.0F && current.get("max_yaw_rate_rps") == 3.0F);
  std::string error;

  // 1. Absent keys keep the current values; present ones apply; the version is carried.
  auto parsed = parse_teleop_settings(R"({"version": 4, "max_speed_mps": 0.4})", current, schema, error);
  assert(parsed && error.empty());
  assert(parsed->version == 4 && std::abs(parsed->get("max_speed_mps") - 0.4F) < 1e-6F);
  assert(parsed->get("max_yaw_rate_rps") == 3.0F && parsed->selected_motion.empty());

  // 2. Past the qualified value is allowed (the dashboard shows it in red); the hard bound caps.
  parsed = parse_teleop_settings(R"({"max_speed_mps": 1.5, "max_yaw_rate_rps": 9.0})", current, schema, error);
  assert(parsed && std::abs(parsed->get("max_speed_mps") - 1.5F) < 1e-6F && parsed->get("max_yaw_rate_rps") == 6.0F);

  // 3. One malformed field rejects the whole document, so nothing half-applies; unknown keys are ignored.
  assert(!parse_teleop_settings(R"({"max_speed_mps": 0.3, "max_yaw_rate_rps": "fast"})", current, schema, error));
  assert(error.find("max_yaw_rate_rps") != std::string::npos);
  assert(!parse_teleop_settings(R"({"max_speed_mps": -1})", current, schema, error));
  assert(!parse_teleop_settings(R"({"version": 1.5})", current, schema, error));
  assert(!parse_teleop_settings("not json", current, schema, error));
  assert(!parse_teleop_settings("[1, 2]", current, schema, error));
  assert(!parse_teleop_settings(R"({"selected_motion": 3})", current, schema, error));
  parsed = parse_teleop_settings(R"({"selected_motion": "kick", "other": "x"})", current, schema, error);
  assert(parsed && parsed->selected_motion == "kick");

  // 4. The schema travels to the dashboard as JSON with the owner's ranges.
  const auto json = nlohmann::json::parse(schema_json(schema));
  assert(json.is_array() && json.size() == 2 && json[0]["key"] == "max_speed_mps");
  assert(json[0]["qualified_max"] == 1.0F && json[0]["hard_max"] == 2.0F && json[1]["unit"] == "rad/s");

  // 5. The task declares its own ranges, and its default speed is the one the grids qualified.
  const auto task_schema = metatron_v1::MetatronTask::settings_schema();
  assert(task_schema.size() == 2 && task_schema[0].key == "max_speed_mps" && task_schema[0].value == 1.0F);
  assert(task_schema[0].hard_max <= 5.0F);  // the task rejects a steering request faster than that

  // 6. The file poller reads only on change and tolerates a file that is not there yet.
  const auto directory = std::filesystem::temp_directory_path() / "protomotions_teleop_settings_test";
  std::filesystem::create_directories(directory);
  const auto path = directory / "settings.json";
  std::filesystem::remove(path);
  TeleopSettingsFile file(path, schema);
  assert(!file.poll(current, error) && error.empty());  // missing: no settings, no error
  write(path, R"({"version": 1, "max_speed_mps": 0.5})");
  parsed = file.poll(current, error);
  assert(parsed && parsed->version == 1 && std::abs(parsed->get("max_speed_mps") - 0.5F) < 1e-6F);
  current = *parsed;
  assert(!file.poll(current, error) && error.empty());  // unchanged: nothing new
  // A later write with a newer mtime is picked up; a broken one reports and keeps the current settings.
  std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(2));
  assert(file.poll(current, error));
  write(path, "{broken");
  std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(4));
  assert(!file.poll(current, error) && !error.empty());
  assert(std::abs(current.get("max_speed_mps") - 0.5F) < 1e-6F);
  std::filesystem::remove_all(directory);

  // 7. The default path sits in shared memory when the machine has it, per user.
  const auto location = default_teleop_settings_path();
  assert(location.filename().string().rfind("protomotions_teleop_settings_", 0) == 0);

  std::cout << "teleop settings: schema, parse, bounds, rejection, file polling and default path checks pass\n";
  return 0;
}
