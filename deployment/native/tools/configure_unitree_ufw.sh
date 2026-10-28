#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

interface=""
participant_index=0
robot_subnet="192.168.123.0/24"

usage() {
  echo "usage: sudo $0 --interface IF [--participant-index N] [--robot-subnet CIDR]" >&2
}

while (($#)); do
  case "$1" in
    --interface)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      interface="$2"
      shift 2
      ;;
    --participant-index)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      participant_index="$2"
      shift 2
      ;;
    --robot-subnet)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      robot_subnet="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "run this one-time provisioning command with sudo" >&2
  exit 1
fi
if [[ -z "$interface" || ! "$interface" =~ ^[A-Za-z0-9_.:-]+$ ]]; then
  echo "--interface must name one network interface" >&2
  exit 2
fi
if [[ ! "$participant_index" =~ ^[0-9]+$ ]] ||
   ((participant_index < 0 || participant_index >= 120)); then
  echo "--participant-index must be in [0,119]" >&2
  exit 2
fi
if [[ ! -e "/sys/class/net/$interface" ]]; then
  echo "network interface does not exist: $interface" >&2
  echo "run 'ip -brief address' and retry with the current robot adapter name" >&2
  exit 1
fi
if ! command -v ufw >/dev/null 2>&1; then
  echo "ufw is not installed" >&2
  exit 1
fi

unicast_start=$((7410 + 2 * participant_index))
unicast_end=$((unicast_start + 1))

# CycloneDDS domain 0 uses multicast discovery/data ports 7400-7401. A fixed
# ParticipantIndex makes this host's unicast metadata/data ports deterministic.
# Limit both rules to the selected robot NIC and robot subnet; never disable UFW.
ufw allow in on "$interface" proto udp from "$robot_subnet" to any port 7400:7401 \
  comment "ProtoMotions Unitree DDS multicast"
ufw allow in on "$interface" proto udp from "$robot_subnet" to any \
  port "$unicast_start:$unicast_end" \
  comment "ProtoMotions Unitree DDS participant $participant_index"
ufw reload

echo "Unitree DDS firewall rules installed on $interface for $robot_subnet."
echo "Participant $participant_index uses host UDP $unicast_start-$unicast_end."
echo "UFW remains enabled. If the USB NIC name changes, rerun for the new name."
ufw status verbose
