# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(protomotions_deployment VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# The manifest/recovery readers require JSON even without vendor backends.
# Fail during configuration, and support non-system installations explicitly.
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp REQUIRED)
include_directories(SYSTEM "${NLOHMANN_JSON_INCLUDE_DIR}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

function(protomotions_enable_hardening target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${target} PRIVATE
      -fstack-protector-strong
      -Wformat-security
    )
    target_compile_definitions(${target} PRIVATE _FORTIFY_SOURCE=2)
  endif()
endfunction()

function(protomotions_configure_target target)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  protomotions_enable_hardening(${target})
endfunction()

option(PROTOMOTIONS_BUILD_UNITREE_SDK2
       "Build the optional Unitree SDK2 G1 backend adapter" OFF)
option(PROTOMOTIONS_BUILD_SDL2_INPUT
       "Build the optional SDL2 Xbox controller input adapter" OFF)
option(PROTOMOTIONS_BUILD_MUJOCO
       "Build the optional headless MuJoCo validation backend" OFF)
option(PROTOMOTIONS_BUILD_METATRON_NATIVE
       "Build the 39-input Metatron TensorRT engine and benchmark" OFF)
option(PROTOMOTIONS_BUILD_GPC_TENSORRT
       "Build the native GPC TensorRT policy and timing benchmark" OFF)
option(PROTOMOTIONS_BUILD_GPC_TRT_PLUGIN
       "Build the exact top-p TensorRT plugin for the GPC policy" OFF)
set(UNITREE_DDS_ROOT "" CACHE PATH
    "Optional DDS root containing lib/libddsc.so and lib/libddscxx.so")
set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation root for the native benchmark")
set(TENSORRT_LIBRARY_ROOT "" CACHE PATH "Optional TensorRT runtime library root when headers and runtime are separate")
set(CUDA_RUNTIME_ROOT "" CACHE PATH "CUDA runtime installation root for the native benchmark")
set(CUDA_TOOLKIT_INCLUDE_ROOT "" CACHE PATH "Optional full CUDA toolkit include root for the native benchmark")
set(CUDA_NVCC_EXECUTABLE "" CACHE FILEPATH
    "Optional nvcc executable used to build the GPC TensorRT plugin")
set(CUDA_CUDART_ROOT "" CACHE PATH
    "CUDA toolkit cudart library directory used by the GPC plugin nvcc link")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  option(PROTOMOTIONS_BUILD_EVDEV_INPUT
         "Build the dependency-free Linux evdev Xbox controller adapter" ON)
else()
  option(PROTOMOTIONS_BUILD_EVDEV_INPUT
         "Build the dependency-free Linux evdev Xbox controller adapter" OFF)
endif()
option(PROTOMOTIONS_BUILD_PICO_INPUT
       "Build the optional native XRoboToolkit/Pico input adapter" OFF)
set(XROBOTOOLKIT_ROOT "" CACHE PATH
    "XRoboToolkit SDK root containing include/PXREARobotSDK.h and lib/libPXREARobotSDK.so")

# Python loads the JSON; native code embeds the same data without a runtime
# parser/dependency or an installed-source-path requirement.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/assets/styles/crawling.json")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/assets/styles/crawling.json" CRAWLING_JSON)
foreach(field poseSamples velocitySamples)
  string(JSON count LENGTH "${CRAWLING_JSON}" presets crawling ${field})
  if((field STREQUAL "poseSamples" AND NOT count EQUAL 225) OR
     (field STREQUAL "velocitySamples" AND NOT count EQUAL 150))
    message(FATAL_ERROR "Invalid crawling style ${field} length: ${count}")
  endif()
  math(EXPR last "${count} - 1")
  set(values "")
  foreach(index RANGE 0 ${last})
    string(JSON value GET "${CRAWLING_JSON}" presets crawling ${field} ${index})
    if(NOT value MATCHES "[.eE]")
      string(APPEND value ".0")
    endif()
    string(APPEND values "${value}F,")
  endforeach()
  if(field STREQUAL "poseSamples")
    set(CRAWLING_POSE_VALUES "${values}")
  else()
    set(CRAWLING_VELOCITY_VALUES "${values}")
  endif()
endforeach()
configure_file(src/crawling_style_data.hpp.in
  generated/crawling_style_data.hpp @ONLY)

add_library(protomotions_deployment
  src/controller_manager.cpp
  src/metatron_v1_contract.cpp
  src/metatron_v1_manifest.cpp
  src/metatron_v1_recovery.cpp
  src/runtime.cpp
  src/safety.cpp
  src/types.cpp
  src/motion_library.cpp
  src/native_time_series_recorder.cpp
  src/steering_style.cpp
)
target_include_directories(protomotions_deployment PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(protomotions_deployment PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_features(protomotions_deployment PUBLIC cxx_std_20)
protomotions_configure_target(protomotions_deployment)
set_target_properties(protomotions_deployment PROPERTIES
  EXPORT_NAME Deployment
  POSITION_INDEPENDENT_CODE ON
)

# Stable C ABI for thin language bridges. The callbacks remain on the
# caller's control thread; this target does not pull Python or vendor SDKs
# into the safety core.
add_library(protomotions_deployment_c SHARED src/c_api.cpp)
target_link_libraries(protomotions_deployment_c PRIVATE protomotions_deployment)
target_include_directories(protomotions_deployment_c PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
protomotions_configure_target(protomotions_deployment_c)
set_target_properties(protomotions_deployment_c PROPERTIES
  OUTPUT_NAME protomotions_deployment_c
  VERSION ${PROJECT_VERSION}
  # The callback signatures and timestamp contract are intentionally versioned
  # as a new ABI; do not let an old client load this shared object silently.
  SOVERSION 2
)

add_executable(deployment_demo tests/deployment_demo.cpp)
target_link_libraries(deployment_demo PRIVATE protomotions_deployment)
protomotions_configure_target(deployment_demo)

add_executable(runtime_benchmark bench/runtime_benchmark.cpp)
target_link_libraries(runtime_benchmark PRIVATE protomotions_deployment)
protomotions_configure_target(runtime_benchmark)

add_executable(mailbox_benchmark bench/mailbox_benchmark.cpp)
target_link_libraries(mailbox_benchmark PRIVATE protomotions_deployment)
protomotions_configure_target(mailbox_benchmark)

if(PROTOMOTIONS_BUILD_UNITREE_SDK2)
  if(NOT UNITREE_SDK2_ROOT)
    message(FATAL_ERROR
            "UNITREE_SDK2_ROOT must point to the Unitree SDK2 checkout")
  endif()

  find_package(Threads REQUIRED)
  find_library(UNITREE_SDK2_LIBRARY
               NAMES unitree_sdk2
               PATHS "${UNITREE_SDK2_ROOT}/lib/${CMAKE_SYSTEM_PROCESSOR}"
               NO_DEFAULT_PATH REQUIRED)
  find_library(UNITREE_DDS_LIBRARY
               NAMES ddsc
               PATHS
                 "${UNITREE_DDS_ROOT}/lib"
                 "${UNITREE_SDK2_ROOT}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}"
               NO_DEFAULT_PATH REQUIRED)
  find_library(UNITREE_DDSXX_LIBRARY
               NAMES ddscxx
               PATHS
                 "${UNITREE_DDS_ROOT}/lib"
                 "${UNITREE_SDK2_ROOT}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}"
               NO_DEFAULT_PATH REQUIRED)

  add_library(protomotions_unitree_sdk2 src/unitree_sdk2_backend.cpp src/unitree_mode_voice.cpp)
  target_include_directories(protomotions_unitree_sdk2 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  target_include_directories(protomotions_unitree_sdk2 PRIVATE
    "${UNITREE_SDK2_ROOT}/include"
    "${UNITREE_SDK2_ROOT}/thirdparty/include"
    "${UNITREE_SDK2_ROOT}/thirdparty/include/ddscxx"
  )
  target_link_libraries(protomotions_unitree_sdk2 PUBLIC
    protomotions_deployment
  )
  target_link_libraries(protomotions_unitree_sdk2 PRIVATE
    "${UNITREE_SDK2_LIBRARY}"
    "${UNITREE_DDSXX_LIBRARY}"
    "${UNITREE_DDS_LIBRARY}"
    Threads::Threads
  )
  protomotions_configure_target(protomotions_unitree_sdk2)
  add_executable(unitree_sensor_probe examples/unitree_sensor_probe.cpp)
  target_link_libraries(unitree_sensor_probe PRIVATE protomotions_unitree_sdk2)
  protomotions_configure_target(unitree_sensor_probe)
endif()

if(PROTOMOTIONS_BUILD_SDL2_INPUT)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(SDL2 REQUIRED IMPORTED_TARGET sdl2)
  add_library(protomotions_sdl2_input src/sdl2_xbox_input.cpp)
  set_target_properties(protomotions_sdl2_input PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(protomotions_sdl2_input PRIVATE
    protomotions_deployment PkgConfig::SDL2
  )
  protomotions_configure_target(protomotions_sdl2_input)
endif()

if(PROTOMOTIONS_BUILD_EVDEV_INPUT)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "PROTOMOTIONS_BUILD_EVDEV_INPUT requires Linux")
  endif()
  add_library(protomotions_evdev_input src/evdev_xbox_input.cpp)
  set_target_properties(protomotions_evdev_input PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(protomotions_evdev_input PRIVATE protomotions_deployment)
  protomotions_configure_target(protomotions_evdev_input)
  # C ABI used by the Python MuJoCo viewer and other language launchers. The
  # production Unitree executable can use the C++ adapter directly; the ABI
  # keeps qualification tooling from reimplementing evdev decoding in Python.
  add_library(protomotions_evdev_input_c SHARED src/input_c_api.cpp)
  target_link_libraries(protomotions_evdev_input_c PRIVATE
    protomotions_evdev_input protomotions_deployment
  )
  target_include_directories(protomotions_evdev_input_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  protomotions_configure_target(protomotions_evdev_input_c)
  set_target_properties(protomotions_evdev_input_c PROPERTIES
    OUTPUT_NAME protomotions_evdev_input_c
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
  )
endif()

if(PROTOMOTIONS_BUILD_PICO_INPUT)
  if(NOT XROBOTOOLKIT_ROOT)
    message(FATAL_ERROR
            "XROBOTOOLKIT_ROOT must point to the XRoboToolkit SDK checkout")
  endif()
  find_package(Threads REQUIRED)
  find_path(PXREA_INCLUDE_DIR PXREARobotSDK.h
            PATHS "${XROBOTOOLKIT_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
  find_path(XROBOTOOLKIT_JSON_INCLUDE_DIR nlohmann/json.hpp
            PATHS "${XROBOTOOLKIT_ROOT}/include" "/usr/include" REQUIRED)
  find_library(PXREA_LIBRARY NAMES PXREARobotSDK
               PATHS "${XROBOTOOLKIT_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  add_library(protomotions_pico_input src/pico_input.cpp)
  set_target_properties(protomotions_pico_input PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(protomotions_pico_input PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  target_include_directories(protomotions_pico_input PRIVATE
    "${PXREA_INCLUDE_DIR}"
    "${XROBOTOOLKIT_JSON_INCLUDE_DIR}"
  )
  target_link_libraries(protomotions_pico_input PRIVATE
    protomotions_deployment
    "${PXREA_LIBRARY}"
    Threads::Threads
  )
  set_target_properties(protomotions_pico_input PROPERTIES
    BUILD_RPATH "${XROBOTOOLKIT_ROOT}/lib"
    INSTALL_RPATH "${XROBOTOOLKIT_ROOT}/lib"
  )
  protomotions_configure_target(protomotions_pico_input)
  add_executable(pico_input_demo tools/pico_input_demo.cpp)
  target_link_libraries(pico_input_demo PRIVATE protomotions_pico_input)
  set_target_properties(pico_input_demo PROPERTIES
    BUILD_RPATH "${XROBOTOOLKIT_ROOT}/lib"
    INSTALL_RPATH "${XROBOTOOLKIT_ROOT}/lib"
  )
  protomotions_configure_target(pico_input_demo)
endif()

if(PROTOMOTIONS_BUILD_MUJOCO)
  if(NOT MUJOCO_ROOT)
    message(FATAL_ERROR "MUJOCO_ROOT must point to a MuJoCo distribution")
  endif()
  find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
            PATHS "${MUJOCO_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
  find_library(MUJOCO_LIBRARY NAMES mujoco
               PATHS "${MUJOCO_ROOT}" "${MUJOCO_ROOT}/lib"
               NO_DEFAULT_PATH)
  if(NOT MUJOCO_LIBRARY)
    file(GLOB MUJOCO_LIBRARY_CANDIDATES
         "${MUJOCO_ROOT}/libmujoco.so*"
         "${MUJOCO_ROOT}/lib/libmujoco.so*")
    list(SORT MUJOCO_LIBRARY_CANDIDATES COMPARE NATURAL ORDER DESCENDING)
    list(LENGTH MUJOCO_LIBRARY_CANDIDATES MUJOCO_LIBRARY_COUNT)
    if(MUJOCO_LIBRARY_COUNT GREATER 0)
      list(GET MUJOCO_LIBRARY_CANDIDATES 0 MUJOCO_LIBRARY)
    else()
      message(FATAL_ERROR
              "Could not find a versioned MuJoCo shared library under ${MUJOCO_ROOT}")
    endif()
  endif()
  add_library(protomotions_mujoco
    src/mujoco_backend.cpp
    src/mujoco_model_loader.cpp)
  target_include_directories(protomotions_mujoco PRIVATE "${MUJOCO_INCLUDE_DIR}")
  target_link_libraries(protomotions_mujoco PUBLIC
    protomotions_deployment "${MUJOCO_LIBRARY}"
  )
  protomotions_configure_target(protomotions_mujoco)
  add_executable(mujoco_smoke tests/mujoco_smoke.cpp)
  target_link_libraries(mujoco_smoke PRIVATE protomotions_mujoco)
  protomotions_configure_target(mujoco_smoke)
  add_test(NAME mujoco_smoke
           COMMAND mujoco_smoke
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/two_joint.xml)
  add_executable(mujoco_model_loader_test
    tests/mujoco_model_loader_test.cpp)
  target_include_directories(mujoco_model_loader_test PRIVATE
    "${MUJOCO_INCLUDE_DIR}")
  target_link_libraries(mujoco_model_loader_test PRIVATE protomotions_mujoco)
  protomotions_configure_target(mujoco_model_loader_test)
  add_test(NAME mujoco_model_loader
           COMMAND mujoco_model_loader_test
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/two_joint_external_floor.xml)
  if(PROTOMOTIONS_G1_MJCF AND EXISTS "${PROTOMOTIONS_G1_MJCF}")
    add_executable(metatron_mujoco_sensor_test tests/metatron_mujoco_sensor_test.cpp)
    target_link_libraries(metatron_mujoco_sensor_test PRIVATE protomotions_mujoco)
    target_compile_options(metatron_mujoco_sensor_test PRIVATE -UNDEBUG)
    protomotions_configure_target(metatron_mujoco_sensor_test)
    add_test(NAME metatron_mujoco_sensor COMMAND metatron_mujoco_sensor_test
      "${PROTOMOTIONS_G1_MJCF}")
  endif()
endif()

if(PROTOMOTIONS_BUILD_GPC_TENSORRT)
  if(NOT TENSORRT_ROOT OR NOT CUDA_RUNTIME_ROOT)
    message(FATAL_ERROR
            "TENSORRT_ROOT and CUDA_RUNTIME_ROOT are required for the native GPC TensorRT target")
  endif()
  if(NOT TENSORRT_LIBRARY_ROOT)
    set(TENSORRT_LIBRARY_ROOT "${TENSORRT_ROOT}")
  endif()
  find_path(GPC_TENSORRT_INCLUDE_DIR NvInfer.h
            PATHS "${TENSORRT_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
  find_file(GPC_TENSORRT_LIBRARY
            NAMES libnvinfer.so.10 libnvinfer.so
            PATHS "${TENSORRT_LIBRARY_ROOT}/lib" "${TENSORRT_LIBRARY_ROOT}"
            NO_DEFAULT_PATH REQUIRED)
  find_path(GPC_CUDA_RUNTIME_INCLUDE_DIR cuda_runtime_api.h
            PATHS "${CUDA_RUNTIME_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
  find_path(GPC_CUDA_CRT_INCLUDE_DIR crt/host_defines.h
            PATHS "${CUDA_TOOLKIT_INCLUDE_ROOT}/include"
                  "/usr/local/cuda/include"
                  "/usr/local/cuda-13.1/targets/x86_64-linux/include"
            REQUIRED)
  find_file(GPC_CUDA_RUNTIME_LIBRARY
            NAMES libcudart.so.12 libcudart.so
            PATHS "${CUDA_RUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

  add_library(protomotions_gpc_policy src/gpc_policy.cpp)
  target_include_directories(protomotions_gpc_policy PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    "${GPC_TENSORRT_INCLUDE_DIR}"
    "${GPC_CUDA_RUNTIME_INCLUDE_DIR}"
    "${GPC_CUDA_CRT_INCLUDE_DIR}"
  )
  target_link_libraries(protomotions_gpc_policy PUBLIC
    "${GPC_TENSORRT_LIBRARY}"
    "${GPC_CUDA_RUNTIME_LIBRARY}"
    ${CMAKE_DL_LIBS}
  )
  target_compile_features(protomotions_gpc_policy PUBLIC cxx_std_20)
  protomotions_configure_target(protomotions_gpc_policy)
  set_target_properties(protomotions_gpc_policy PROPERTIES POSITION_INDEPENDENT_CODE ON)

  add_executable(gpc_tensorrt_native_benchmark
    bench/gpc_tensorrt_native_benchmark.cpp)
  target_link_libraries(gpc_tensorrt_native_benchmark PRIVATE protomotions_gpc_policy)
  protomotions_configure_target(gpc_tensorrt_native_benchmark)

  if(PROTOMOTIONS_BUILD_GPC_TRT_PLUGIN)
    if(NOT CUDA_CUDART_ROOT)
      message(FATAL_ERROR
              "CUDA_CUDART_ROOT is required for the GPC plugin nvcc link")
    endif()
    if(CUDA_NVCC_EXECUTABLE)
      set(GPC_CUDA_NVCC "${CUDA_NVCC_EXECUTABLE}")
    else()
      find_program(GPC_CUDA_NVCC nvcc REQUIRED)
    endif()
    set(GPC_PLUGIN_OUTPUT
        "${CMAKE_CURRENT_BINARY_DIR}/libprotomotions_gpc_plugins.so")
    add_custom_command(
      OUTPUT "${GPC_PLUGIN_OUTPUT}"
      COMMAND "${GPC_CUDA_NVCC}"
              -std=c++17
              -O3
              -cudart shared
              -gencode arch=compute_120,code=sm_120
              -Xcompiler -fPIC
              -shared
              "${CMAKE_CURRENT_SOURCE_DIR}/src/gpc_top_p_plugin.cu"
              "-I${GPC_TENSORRT_INCLUDE_DIR}"
              "-I${GPC_CUDA_RUNTIME_INCLUDE_DIR}"
              "-I${GPC_CUDA_CRT_INCLUDE_DIR}"
              "-L${TENSORRT_LIBRARY_ROOT}"
              -Xlinker -l:libnvinfer.so.10
              "-L${CUDA_CUDART_ROOT}"
              -Xlinker -rpath -Xlinker "${TENSORRT_LIBRARY_ROOT}"
              -Xlinker -rpath -Xlinker "${CUDA_CUDART_ROOT}"
              -Xlinker -rpath -Xlinker "${CUDA_RUNTIME_ROOT}/lib"
              -o "${GPC_PLUGIN_OUTPUT}"
      DEPENDS src/gpc_top_p_plugin.cu
      VERBATIM
    )
    add_custom_target(protomotions_gpc_plugins DEPENDS "${GPC_PLUGIN_OUTPUT}")
  endif()
endif()

if(PROTOMOTIONS_BUILD_METATRON_NATIVE)
  if(NOT TENSORRT_ROOT OR NOT CUDA_RUNTIME_ROOT)
    message(FATAL_ERROR "TENSORRT_ROOT and CUDA_RUNTIME_ROOT are required for Metatron native")
  endif()
  if(NOT TENSORRT_LIBRARY_ROOT)
    set(TENSORRT_LIBRARY_ROOT "${TENSORRT_ROOT}")
  endif()
  find_path(METATRON_TENSORRT_INCLUDE NvInfer.h
            PATHS "${TENSORRT_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
  find_file(METATRON_TENSORRT_LIBRARY NAMES libnvinfer.so.10 libnvinfer.so
            PATHS "${TENSORRT_LIBRARY_ROOT}/lib" "${TENSORRT_LIBRARY_ROOT}"
            NO_DEFAULT_PATH REQUIRED)
  find_path(METATRON_CUDA_INCLUDE cuda_runtime_api.h
            PATHS "${CUDA_RUNTIME_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
  find_path(METATRON_CUDA_CRT_INCLUDE crt/host_defines.h
            PATHS "${CUDA_TOOLKIT_INCLUDE_ROOT}/include"
                  "/usr/local/cuda/include"
                  "/usr/local/cuda-13.1/targets/x86_64-linux/include" REQUIRED)
  find_file(METATRON_CUDA_LIBRARY NAMES libcudart.so.12 libcudart.so
            PATHS "${CUDA_RUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  add_library(protomotions_metatron_v1_engine
    src/metatron_v1_engine.cpp
    src/metatron_v1_fk_engine.cpp
    src/metatron_v1_rest.cpp
    src/metatron_v1_task.cpp)
  target_include_directories(protomotions_metatron_v1_engine PUBLIC
    "${METATRON_TENSORRT_INCLUDE}" "${METATRON_CUDA_INCLUDE}"
    "${METATRON_CUDA_CRT_INCLUDE}")
  target_link_libraries(protomotions_metatron_v1_engine PUBLIC
    protomotions_deployment "${METATRON_TENSORRT_LIBRARY}"
    "${METATRON_CUDA_LIBRARY}")
  protomotions_configure_target(protomotions_metatron_v1_engine)
  add_executable(metatron_v1_inference_benchmark
    bench/metatron_v1_inference_benchmark.cpp)
  target_link_libraries(metatron_v1_inference_benchmark PRIVATE
    protomotions_metatron_v1_engine)
  protomotions_configure_target(metatron_v1_inference_benchmark)
  add_executable(metatron_v1_parity bench/metatron_v1_parity.cpp)
  target_link_libraries(metatron_v1_parity PRIVATE
    protomotions_metatron_v1_engine)
  protomotions_configure_target(metatron_v1_parity)
  add_executable(metatron_v1_fk_parity bench/metatron_v1_fk_parity.cpp)
  target_link_libraries(metatron_v1_fk_parity PRIVATE
    protomotions_metatron_v1_engine)
  protomotions_configure_target(metatron_v1_fk_parity)
  add_executable(metatron_v1_rest_dump
    bench/metatron_v1_rest_dump.cpp)
  target_link_libraries(metatron_v1_rest_dump PRIVATE
    protomotions_metatron_v1_engine)
  protomotions_configure_target(metatron_v1_rest_dump)
  add_executable(metatron_v1_task_dump
    bench/metatron_v1_task_dump.cpp)
  target_link_libraries(metatron_v1_task_dump PRIVATE
    protomotions_metatron_v1_engine)
  protomotions_configure_target(metatron_v1_task_dump)
  if(PROTOMOTIONS_BUILD_UNITREE_SDK2 AND PROTOMOTIONS_BUILD_MUJOCO
     AND PROTOMOTIONS_BUILD_EVDEV_INPUT)
    add_executable(metatron_v1_deploy_cpp
      examples/metatron_v1_deploy.cpp)
    target_include_directories(metatron_v1_deploy_cpp PRIVATE
      "${MUJOCO_INCLUDE_DIR}")
    target_link_libraries(metatron_v1_deploy_cpp PRIVATE
      protomotions_metatron_v1_engine
      protomotions_unitree_sdk2
      protomotions_mujoco
      protomotions_evdev_input)
    protomotions_configure_target(metatron_v1_deploy_cpp)
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(motion_library_test tests/motion_library_test.cpp)
  target_link_libraries(motion_library_test PRIVATE protomotions_deployment)
  protomotions_configure_target(motion_library_test)
  target_compile_options(motion_library_test PRIVATE -UNDEBUG)
  add_test(NAME motion_library COMMAND motion_library_test
    "${CMAKE_CURRENT_BINARY_DIR}/motion_library_test.pmmot")
  add_executable(metatron_v1_contract_test
    tests/metatron_v1_contract_test.cpp)
  target_link_libraries(metatron_v1_contract_test PRIVATE protomotions_deployment)
  protomotions_configure_target(metatron_v1_contract_test)
  add_test(NAME metatron_v1_contract COMMAND metatron_v1_contract_test)
  add_executable(metatron_v1_goals_test tests/metatron_v1_goals_test.cpp)
  target_link_libraries(metatron_v1_goals_test PRIVATE protomotions_deployment)
  protomotions_configure_target(metatron_v1_goals_test)
  add_test(NAME metatron_v1_goals COMMAND metatron_v1_goals_test)
  add_executable(metatron_v1_recovery_test tests/metatron_v1_recovery_test.cpp)
  target_link_libraries(metatron_v1_recovery_test PRIVATE protomotions_deployment)
  protomotions_configure_target(metatron_v1_recovery_test)
  if(EXISTS "${CMAKE_SOURCE_DIR}/artifacts/metatron-ep6410/recovery_plan.json")
    add_test(NAME metatron_v1_recovery COMMAND metatron_v1_recovery_test
      "${CMAKE_SOURCE_DIR}/artifacts/metatron-ep6410/recovery_plan.json")
  endif()
  add_executable(metatron_v1_action_map_test tests/metatron_v1_action_map_test.cpp)
  target_link_libraries(metatron_v1_action_map_test PRIVATE protomotions_deployment)
  protomotions_configure_target(metatron_v1_action_map_test)
  add_test(NAME metatron_v1_action_map COMMAND metatron_v1_action_map_test)
  add_executable(metatron_v1_sensor_test tests/metatron_v1_sensor_test.cpp)
  target_link_libraries(metatron_v1_sensor_test PRIVATE protomotions_deployment)
  protomotions_configure_target(metatron_v1_sensor_test)
  add_test(NAME metatron_v1_sensor COMMAND metatron_v1_sensor_test)
  add_executable(metatron_v1_command_test tests/metatron_v1_command_test.cpp)
  target_link_libraries(metatron_v1_command_test PRIVATE protomotions_deployment)
  protomotions_configure_target(metatron_v1_command_test)
  add_test(NAME metatron_v1_command COMMAND metatron_v1_command_test)
  add_executable(metatron_v1_controller_test tests/metatron_v1_controller_test.cpp)
  target_link_libraries(metatron_v1_controller_test PRIVATE protomotions_deployment)
  protomotions_configure_target(metatron_v1_controller_test)
  add_test(NAME metatron_v1_controller COMMAND metatron_v1_controller_test)
  if(EXISTS "${CMAKE_SOURCE_DIR}/artifacts/metatron-ep6410/manifest.json")
    add_executable(metatron_v1_manifest_test tests/metatron_v1_manifest_test.cpp)
    target_link_libraries(metatron_v1_manifest_test PRIVATE protomotions_deployment)
    protomotions_configure_target(metatron_v1_manifest_test)
    add_test(NAME metatron_v1_manifest COMMAND metatron_v1_manifest_test
      "${CMAKE_SOURCE_DIR}/artifacts/metatron-ep6410/manifest.json")
    target_compile_options(metatron_v1_manifest_test PRIVATE -UNDEBUG)
  endif()
  # These focused tests use assert; keep assertions active in Release builds.
  foreach(target IN ITEMS metatron_v1_contract_test metatron_v1_goals_test
      metatron_v1_recovery_test metatron_v1_action_map_test
      metatron_v1_sensor_test metatron_v1_command_test
      metatron_v1_controller_test)
    target_compile_options(${target} PRIVATE -UNDEBUG)
  endforeach()

  add_executable(steering_style_dump tests/steering_style_dump.cpp)
  target_link_libraries(steering_style_dump PRIVATE protomotions_deployment)
  protomotions_configure_target(steering_style_dump)
  add_test(NAME steering_style_contract COMMAND steering_style_dump)

  add_executable(deployment_contract_tests tests/deployment_contract_tests.cpp)
  add_executable(motor_temperature_tests tests/motor_temperature_tests.cpp)
  target_link_libraries(motor_temperature_tests PRIVATE protomotions_deployment)
  protomotions_configure_target(motor_temperature_tests)
  add_test(NAME motor_temperature_tests COMMAND motor_temperature_tests)
  add_executable(mode_voice_tests tests/mode_voice_tests.cpp)
  target_link_libraries(mode_voice_tests PRIVATE protomotions_deployment)
  protomotions_configure_target(mode_voice_tests)
  add_test(NAME mode_voice_tests COMMAND mode_voice_tests)
  target_link_libraries(deployment_contract_tests PRIVATE protomotions_deployment)
  protomotions_configure_target(deployment_contract_tests)
  add_test(NAME deployment_contract_tests COMMAND deployment_contract_tests)
  add_executable(c_api_smoke tests/c_api_smoke.cpp)
  target_link_libraries(c_api_smoke PRIVATE protomotions_deployment_c)
  protomotions_configure_target(c_api_smoke)
  add_test(NAME c_api_smoke COMMAND c_api_smoke)
  if(PROTOMOTIONS_BUILD_UNITREE_SDK2)
    # The SDK target is optional and cannot be built on machines without the
    # vendor SDK. When enabled, keep its compile/link check visible in CTest so
    # CI and hardware qualification jobs fail loudly instead of silently
    # skipping the real-robot adapter.
    add_test(
      NAME unitree_sdk2_compile
      COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
              --target protomotions_unitree_sdk2
              --config $<CONFIG>
    )
    add_executable(unitree_g1_sensor_contract_test
      tests/unitree_g1_sensor_contract_test.cpp)
    target_link_libraries(unitree_g1_sensor_contract_test PRIVATE
      protomotions_unitree_sdk2)
    protomotions_configure_target(unitree_g1_sensor_contract_test)
    add_test(NAME unitree_g1_sensor_contract
             COMMAND unitree_g1_sensor_contract_test)
  endif()
endif()

install(TARGETS protomotions_deployment protomotions_deployment_c deployment_demo
  EXPORT ProtoMotionsDeploymentTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
if(PROTOMOTIONS_BUILD_GPC_TENSORRT)
  install(TARGETS protomotions_gpc_policy gpc_tensorrt_native_benchmark
    EXPORT ProtoMotionsDeploymentTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
  if(PROTOMOTIONS_BUILD_GPC_TRT_PLUGIN)
    install(FILES "${GPC_PLUGIN_OUTPUT}"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
  endif()
endif()
if(PROTOMOTIONS_BUILD_UNITREE_SDK2)
  install(TARGETS protomotions_unitree_sdk2
    EXPORT ProtoMotionsDeploymentTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()
if(PROTOMOTIONS_BUILD_SDL2_INPUT)
  install(TARGETS protomotions_sdl2_input
    EXPORT ProtoMotionsDeploymentTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()
if(PROTOMOTIONS_BUILD_EVDEV_INPUT)
  install(TARGETS protomotions_evdev_input
    EXPORT ProtoMotionsDeploymentTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
  install(TARGETS protomotions_evdev_input_c
    EXPORT ProtoMotionsDeploymentTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()
if(PROTOMOTIONS_BUILD_MUJOCO)
  install(TARGETS protomotions_mujoco mujoco_smoke
    EXPORT ProtoMotionsDeploymentTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${PROJECT_NAME}
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/ProtoMotionsDeploymentConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ProtoMotionsDeploymentConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ProtoMotionsDeploymentConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ProtoMotionsDeployment
)
install(EXPORT ProtoMotionsDeploymentTargets
  FILE ProtoMotionsDeploymentTargets.cmake
  NAMESPACE ProtoMotions::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ProtoMotionsDeployment
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/ProtoMotionsDeploymentConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ProtoMotionsDeploymentConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ProtoMotionsDeployment
)
