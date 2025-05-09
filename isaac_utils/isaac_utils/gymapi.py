"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Gym bindings wrapper module
"""

from __future__ import print_function, division, absolute_import

import importlib
import json
import sys
import os

from isaac_utils import gymdeps


def _format_path(pathstr):
    if os.name == "nt":
        # need to flip backslashes and convert to lower case
        return pathstr.replace("\\", "/").lower()
    else:
        return pathstr


def _import_active_version():
    major = sys.version_info[0]
    minor = sys.version_info[1]

    # print("Running with Python %d.%d" % (major, minor))

    # get the directory that contains the native libs
    lib_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "_bindings"))

    if os.name == "nt":
        platform = "windows-x86_64"
        ext = "pyd"
    else:
        platform = "linux-x86_64"
        ext = "so"

    module_name = "gym_%d%d" % (major, minor)
    module_dir = os.path.join(lib_dir, platform)
    module_path = os.path.join(module_dir, "%s.%s" % (module_name, ext))
    package_path = "isaacgym._bindings.%s.%s" % (platform, module_name)

    # print(module_name)
    # print(module_dir)
    # print(module_path)
    # print(package_path)

    if os.path.isfile(module_path):
        # tell Carbonite where the plugins are
        os.environ["CARB_APP_PATH"] = _format_path(module_dir)

        print("Importing module '%s' (%s)" % (module_name, module_path))
        module = importlib.import_module(package_path)
        # https://stackoverflow.com/questions/28826127/dynamically-load-all-names-from-module-in-python
        # Now extract the attributes into the globals() namespace, as 'from ... import *' would do.
        # A module can define __all__ to explicitly allow all symbols to be imported.
        # Otherwise, we import all names that do not start with an underscore.
        if hasattr(module, "__all__"):
            attrs = {key: getattr(module, key) for key in module.__all__}
        else:
            attrs = {key: value for key, value in module.__dict__.items() if key[0] != "_"}
        globals().update(attrs)

        # initialize the carbonite framework
        # jcfg = {
        #    "log": {
        #        "level": "info"
        #    }
        # }
        # jcfg["pluginSearchPaths"] = [ _format_path(module_dir) ]
        # cfg_str = json.dumps(jcfg, indent=2)
        # print("Config:\n%s" % cfg_str)
        # carb_init(cfg_str)

        # use shared carb.config.json
        carb_init()

        # set custom path to USD plugInfo.json
        usd_plug_info_path = os.path.join(lib_dir, platform, "usd", "plugInfo.json")
        if os.path.isfile(usd_plug_info_path):
            os.environ["GYM_USD_PLUG_INFO_PATH"] = usd_plug_info_path
            print("Setting GYM_USD_PLUG_INFO_PATH to %s" % usd_plug_info_path)
        else:
            print("Warning: Failed to find USD plugInfo file (%s)" % usd_plug_info_path)

        # add module search path for USD and related bindings for specific Python versions
        if major == 3 and minor == 6:
            sys.path.append(os.path.join(lib_dir, platform, "py36"))

    else:
        raise RuntimeError("No gym module found for the active version of Python (%d.%d)" % (major, minor))


_import_active_version()
