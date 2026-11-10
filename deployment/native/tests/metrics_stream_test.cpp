// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metrics_stream.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cassert>
#include <limits>
#include <string>

using protomotions::deployment::MetricsFrame;

int main() {
  MetricsFrame frame;
  frame.reset(12, 34);
  frame.set("numeric", 1.25F);
  frame.set("nan", std::numeric_limits<float>::quiet_NaN());
  frame.set("infinity", std::numeric_limits<float>::infinity());
  frame.set_vector("vector", std::array<float, 2>{-2.0F, 3.5F});
  const std::string numeric_json = frame.to_json();
  assert(numeric_json == "{\"t_ns\":12,\"cycle\":34,\"numeric\":1.25,\"nan\":null,\"infinity\":null,\"vector/0\":-2,\"vector/1\":3.5}");
  assert(frame.values().size() == 5);

  const std::string name = "policy/\"name\\\n\t";
  const std::string value = "Policy \"你好\" 🤖\\checkpoint\nline\r\t";
  frame.set_text(name, value);
  frame.set_text("policy/name", "policy_ep6410/last.ckpt");
  frame.set_text("empty", "");
  std::string controls;
  for (char ch = 0; ch < 0x20; ++ch) controls += ch;
  frame.set_text("controls", controls);
  frame.set_text(std::string("nul\0key", 7), std::string("nul\0value", 9));
  frame.set("numeric/\"escaped\\\n", 7.0F);
  const auto line = frame.to_json();
  assert(line.find('\n') == std::string::npos);
  assert(line.find('\r') == std::string::npos);
  assert(line.find('\0') == std::string::npos);
  const auto parsed = nlohmann::json::parse(line);
  assert(parsed.at(name).get<std::string>() == value);
  assert(parsed.at("policy/name") == "policy_ep6410/last.ckpt");
  assert(parsed.at("empty") == "");
  assert(parsed.at("controls").get<std::string>() == controls);
  assert(parsed.at(std::string("nul\0key", 7)).get<std::string>() == std::string("nul\0value", 9));
  assert(parsed.at("numeric/\"escaped\\\n") == 7.0F);
  const auto numeric = nlohmann::json::parse(numeric_json);
  for (const auto& [key, original] : numeric.items()) assert(parsed.at(key) == original);
  // Text does not enter the numeric values API used by existing producers.
  assert(frame.values().size() == 6);

  frame.reset(99, 100);
  assert(frame.timestamp_ns() == 99 && frame.cycle() == 100);
  assert(frame.values().empty());
  assert(frame.to_json() == "{\"t_ns\":99,\"cycle\":100}");
  frame.set_text("policy/name", "new policy");
  frame.set("numeric", 2.0F);
  const auto reused = nlohmann::json::parse(frame.to_json());
  assert(reused.size() == 4);
  assert(reused.at("policy/name") == "new policy");
  assert(reused.at("numeric") == 2.0F);
}
