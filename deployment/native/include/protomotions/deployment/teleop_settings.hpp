// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nlohmann/json.hpp>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace protomotions::deployment {

// One operator-tunable value, declared by whoever owns its meaning (a task,
// a controller, a policy): its default, the largest value that was qualified,
// and the hard bound past which the owner refuses to run. The dashboard draws
// the slider from this, so the owner, not the UI, says where "less reliable"
// starts, and a new tunable needs no dashboard change.
struct SettingRange {
  std::string key;      // settings-file key and metrics name, e.g. "max_speed_mps"
  std::string label;    // what the slider says
  std::string unit;
  float value;          // the default: what runs when the file says nothing
  float qualified_max;  // above this the dashboard shows the value in red
  float hard_max;       // the slider's end and the parser's cap
};
using TeleopSettingsSchema = std::vector<SettingRange>;

// Operator settings the dashboard writes and the launcher reads: a small JSON
// file replaced atomically (write-then-rename) by the writer, so a reader never
// sees half a document. Settings are state, not events: a value stays until the
// next change, survives a dashboard restart (not a launcher start, which
// removes the file), and can be inspected or edited by hand. Anything that must happen at a moment (start a motion, revive) stays on
// the controller, because a remembered command would fire again after a restart.
struct TeleopSettings {
  int version = 0;                       // bumped by the writer on every change; echoed back on the metrics stream
  std::map<std::string, float> values;   // one per declared SettingRange
  std::string selected_motion;           // the motion the controller's button would play; empty = unchanged

  float get(const std::string& key) const { return values.at(key); }
};

inline TeleopSettings defaults_from(const TeleopSettingsSchema& schema) {
  TeleopSettings settings;
  for (const auto& range : schema) settings.values[range.key] = range.value;
  return settings;
}

// The schema as the dashboard reads it off the metrics stream.
inline std::string schema_json(const TeleopSettingsSchema& schema) {
  nlohmann::json out = nlohmann::json::array();
  for (const auto& range : schema)
    out.push_back({{"key", range.key}, {"label", range.label}, {"unit", range.unit}, {"default", range.value},
                   {"qualified_max", range.qualified_max}, {"hard_max", range.hard_max}});
  return out.dump();
}

// Parses a settings document on top of `current`: keys that are absent keep
// their value, numbers are capped at their hard bound, and any malformed field
// rejects the whole document (the caller keeps `current`), so a typo cannot
// half-apply. Keys the schema does not declare are ignored.
inline std::optional<TeleopSettings> parse_teleop_settings(std::string_view text, const TeleopSettings& current,
                                                           const TeleopSettingsSchema& schema, std::string& error) {
  const auto document = nlohmann::json::parse(text, nullptr, false);
  if (!document.is_object()) {
    error = "settings file is not a JSON object";
    return std::nullopt;
  }
  TeleopSettings next = current;
  if (const auto it = document.find("version"); it != document.end()) {
    if (!it->is_number_integer()) {
      error = "version is not an integer";
      return std::nullopt;
    }
    next.version = it->get<int>();
  }
  for (const auto& range : schema) {
    const auto it = document.find(range.key);
    if (it == document.end()) continue;
    if (!it->is_number()) {
      error = range.key + " is not a number";
      return std::nullopt;
    }
    const float value = it->get<float>();
    if (!std::isfinite(value) || value < 0.0F) {
      error = range.key + " must be finite and non-negative";
      return std::nullopt;
    }
    next.values[range.key] = std::min(value, range.hard_max);
  }
  if (const auto it = document.find("selected_motion"); it != document.end()) {
    if (!it->is_string()) {
      error = "selected_motion is not a string";
      return std::nullopt;
    }
    next.selected_motion = it->get<std::string>();
  }
  error.clear();
  return next;
}

// Where the launcher and the dashboard meet. /dev/shm keeps the read off the
// disk, so a busy drive cannot stall the control loop; the uid keeps two
// operators' sessions on one machine apart.
inline std::filesystem::path default_teleop_settings_path() {
  const std::string name = "protomotions_teleop_settings_" + std::to_string(::getuid()) + ".json";
  std::error_code ignored;
  const std::filesystem::path shm = "/dev/shm";
  return (std::filesystem::is_directory(shm, ignored) ? shm : std::filesystem::temp_directory_path(ignored)) / name;
}

// Polls the file once per control cycle by modification time and parses it
// only when it changed. A missing file is not an error: the dashboard may not
// have written yet, or there is no dashboard.
class TeleopSettingsFile {
 public:
  TeleopSettingsFile(std::filesystem::path path, TeleopSettingsSchema schema)
      : path_(std::move(path)), schema_(std::move(schema)) {}

  const std::filesystem::path& path() const noexcept { return path_; }
  const TeleopSettingsSchema& schema() const noexcept { return schema_; }

  // New settings when the file changed and parsed, otherwise nullopt; `error`
  // carries the reason when it changed and did not parse, else it is empty.
  std::optional<TeleopSettings> poll(const TeleopSettings& current, std::string& error) {
    error.clear();
    std::error_code status;
    const auto stamp = std::filesystem::last_write_time(path_, status);
    if (status || stamp == last_seen_) return std::nullopt;
    last_seen_ = stamp;
    std::ifstream file(path_);
    if (!file) {
      error = "settings file could not be opened";
      return std::nullopt;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return parse_teleop_settings(buffer.str(), current, schema_, error);
  }

 private:
  std::filesystem::path path_;
  TeleopSettingsSchema schema_;
  std::filesystem::file_time_type last_seen_{};
};

}  // namespace protomotions::deployment
