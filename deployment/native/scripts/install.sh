#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
  cat <<'EOF'
Install the ProtoMotions deployment runtime.

Usage: scripts/install.sh [options]

Options:
  --prefix DIR       Install prefix (default: $HOME/.local)
  --build-dir DIR    CMake build directory (default: <repo>/build-release)
  --build-type TYPE  CMake build type (default: Release)
  --unitree-sdk2-root DIR
                     Enable and build the optional Unitree SDK2 adapter
  --no-tests         Skip CTest during installation
  -h, --help         Show this help

The core install requires a C++20 compiler, CMake and nlohmann/json headers. Policy
backends such as ONNX Runtime, TensorRT, and CUDA graphs are optional
application adapters and are not downloaded by this script.
EOF
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
default_prefix="${HOME:-$repo_root}/.local"
prefix="$default_prefix"
build_dir="$repo_root/build-release"
build_type=Release
unitree_sdk2_root=""
run_tests=1
install_bindir=bin
install_libdir=lib
install_includedir=include

while (($#)); do
  case "$1" in
    --prefix)
      (($# >= 2)) || { echo "--prefix needs a directory" >&2; exit 2; }
      prefix=$2
      shift 2
      ;;
    --build-dir)
      (($# >= 2)) || { echo "--build-dir needs a directory" >&2; exit 2; }
      build_dir=$2
      shift 2
      ;;
    --build-type)
      (($# >= 2)) || { echo "--build-type needs a value" >&2; exit 2; }
      build_type=$2
      shift 2
      ;;
    --unitree-sdk2-root)
      (($# >= 2)) || { echo "--unitree-sdk2-root needs a directory" >&2; exit 2; }
      unitree_sdk2_root=$2
      shift 2
      ;;
    --no-tests)
      run_tests=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command -v cmake >/dev/null || { echo "cmake is required" >&2; exit 1; }
if ! command -v c++ >/dev/null && ! command -v g++ >/dev/null && ! command -v clang++ >/dev/null; then
  echo "a C++ compiler is required" >&2
  exit 1
fi

cmake_args=(
  -S "$repo_root"
  -B "$build_dir"
  -DCMAKE_BUILD_TYPE="$build_type"
  -DBUILD_TESTING=ON
  -DCMAKE_INSTALL_BINDIR="$install_bindir"
  -DCMAKE_INSTALL_LIBDIR="$install_libdir"
  -DCMAKE_INSTALL_INCLUDEDIR="$install_includedir"
  -DCMAKE_INSTALL_PREFIX="$prefix"
)
if [[ -n "$unitree_sdk2_root" ]]; then
  cmake_args+=(
    -DPROTOMOTIONS_BUILD_UNITREE_SDK2=ON
    "-DUNITREE_SDK2_ROOT=$unitree_sdk2_root"
  )
fi
cmake "${cmake_args[@]}"
cmake --build "$build_dir" --parallel
if ((run_tests)); then
  ctest --test-dir "$build_dir" --output-on-failure
fi
cmake --install "$build_dir"

installed_demo="$prefix/$install_bindir/deployment_demo"
"$installed_demo" >/dev/null

cat <<EOF
Installed ProtoMotions deployment runtime.
  prefix:    $prefix
  library:   $prefix/$install_libdir
  headers:   $prefix/$install_includedir
  executable: $installed_demo
  tests:     $([ "$run_tests" -eq 1 ] && echo run || echo skipped)
  unitree-sdk2: $([ -n "$unitree_sdk2_root" ] && echo enabled || echo disabled)
EOF
