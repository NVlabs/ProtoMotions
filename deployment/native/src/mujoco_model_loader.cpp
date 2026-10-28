// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/mujoco_model_loader.hpp"

#include <mujoco/mujoco.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace protomotions::deployment {
namespace {

std::string read_text(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot open MuJoCo XML: " + path.string());
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

void resolve_meshdir(std::string& xml,
                     const std::filesystem::path& source_directory) {
  const auto compiler = xml.find("<compiler");
  if (compiler == std::string::npos) return;
  const auto compiler_end = xml.find('>', compiler);
  if (compiler_end == std::string::npos) return;
  const auto key = xml.find("meshdir=", compiler);
  if (key == std::string::npos || key >= compiler_end) return;
  const auto quote = xml.find_first_of("\"'", key + 8);
  if (quote == std::string::npos || quote >= compiler_end) return;
  const auto close = xml.find(xml[quote], quote + 1);
  if (close == std::string::npos || close > compiler_end) return;
  const std::filesystem::path configured(xml.substr(quote + 1, close - quote - 1));
  if (configured.is_absolute()) return;
  const auto absolute =
      std::filesystem::weakly_canonical(source_directory / configured);
  xml.replace(quote + 1, close - quote - 1, absolute.string());
}

bool remove_contact_section(std::string& xml) {
  const auto begin = xml.find("<contact");
  if (begin == std::string::npos) return false;
  const auto end_tag = xml.find("</contact>", begin);
  if (end_tag == std::string::npos) {
    throw std::runtime_error("malformed MuJoCo contact section");
  }
  xml.erase(begin, end_tag + std::string("</contact>").size() - begin);
  return true;
}

}  // namespace

mjModel* load_kinematics_model(const std::string& xml_path,
                               std::string* load_note) {
  std::array<char, 2048> error{};
  if (mjModel* model =
          mj_loadXML(xml_path.c_str(), nullptr, error.data(), error.size())) {
    if (load_note) *load_note = "loaded source XML directly";
    return model;
  }

  const std::filesystem::path source =
      std::filesystem::absolute(xml_path).lexically_normal();
  std::string xml = read_text(source);
  if (!remove_contact_section(xml)) {
    throw std::runtime_error(
        std::string("MuJoCo XML load failed: ") +
        (error[0] ? error.data() : "unknown error"));
  }
  resolve_meshdir(xml, source.parent_path());

  error.fill('\0');
  mjSpec* spec =
      mj_parseXMLString(xml.c_str(), nullptr, error.data(), error.size());
  if (!spec) {
    throw std::runtime_error(
        std::string("kinematics-only MuJoCo XML parse failed: ") +
        (error[0] ? error.data() : "unknown error"));
  }
  mjModel* model = mj_compile(spec, nullptr);
  if (!model) {
    const char* compile_error = mjs_getError(spec);
    const std::string message =
        compile_error ? compile_error : "unknown compile error";
    mj_deleteSpec(spec);
    throw std::runtime_error(
        "kinematics-only MuJoCo XML compile failed: " + message);
  }
  mj_deleteSpec(spec);
  if (load_note) {
    *load_note =
        "loaded kinematics-only view; external scene contact pairs ignored";
  }
  return model;
}

}  // namespace protomotions::deployment
