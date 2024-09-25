"""Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Imports base_interface dependencies.  This module is for internal use and should be imported by each public module."""

import ctypes
import os
import sys


def _import_deps():

    # make sure that torch was not imported before isaacgym modules
    if "torch" in sys.modules:
        raise ImportError(
            "PyTorch was imported before isaacgym modules.  Please import torch after isaacgym modules."
        )

    # get the directory that contains the native libs
    lib_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "_bindings"))

    if os.name == "nt":
        platform = "windows-x86_64"
    else:
        platform = "linux-x86_64"

    lib_dir = os.path.join(lib_dir, platform)

    # preload some tricky libs
    if os.name == "nt":
        try:
            # physx libs
            ctypes.WinDLL(os.path.join(lib_dir, "PhysXDevice64.dll"))
            ctypes.WinDLL(os.path.join(lib_dir, "PhysXGpu_64.dll"))
        except OSError:
            print("*** Warning: failed to preload PhysX libs")
    else:
        try:
            # load CUDA lib before PhysX
            ctypes.CDLL("libcuda.so", ctypes.RTLD_GLOBAL)
        except OSError:
            print("*** Warning: failed to preload CUDA lib")
        try:
            # physx
            ctypes.CDLL(os.path.join(lib_dir, "libPhysXGpu_64.so"))
        except OSError:
            print("*** Warning: failed to preload PhysX libs")
        try:
            # usd libs
            ctypes.CDLL(os.path.join(lib_dir, "libboost_system.so"))
            ctypes.CDLL(os.path.join(lib_dir, "libboost_thread.so"))
            ctypes.CDLL(os.path.join(lib_dir, "libarch.so"))
            ctypes.CDLL(os.path.join(lib_dir, "libtf.so"))
            ctypes.CDLL(os.path.join(lib_dir, "libmem_filesys.so"))
        except OSError:
            print("*** Warning: failed to preload USD libs")


_import_deps()
