// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdlib>
#include <iostream>
#include <net/if.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace protomotions::deployment {

inline void configure_unitree_dds(const std::string& net_if, int participant_index) {
  if (if_nametoindex(net_if.c_str()) == 0)
    throw std::runtime_error("network interface does not exist: " + net_if);
  if (participant_index < 0 || participant_index >= 120)
    throw std::invalid_argument("DDS participant index must be in [0,119]");
  const char* existing = std::getenv("CYCLONEDDS_URI");
  if (existing && existing[0]) {
    std::cout << "startup: respecting existing CYCLONEDDS_URI; verify its participant index and interface\n";
    return;
  }
  std::ostringstream xml;
  xml << "<CycloneDDS><Domain Id=\"any\"><Discovery><ParticipantIndex>"
      << participant_index << "</ParticipantIndex></Discovery></Domain></CycloneDDS>";
  if (setenv("CYCLONEDDS_URI", xml.str().c_str(), 0) != 0)
    throw std::runtime_error("could not configure CYCLONEDDS_URI");
  const int port = 7410 + 2 * participant_index;
  std::cout << "startup: CycloneDDS participant_index=" << participant_index
            << " host UDP ports=" << port << '-' << port + 1 << '\n';
}

}  // namespace protomotions::deployment
