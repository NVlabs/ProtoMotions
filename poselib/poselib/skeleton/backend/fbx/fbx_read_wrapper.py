"""
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property and proprietary 
rights in and to this software, related documentation and any modifications thereto. Any 
use, reproduction, disclosure or distribution of this software and related documentation 
without an express license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

"""
Script that reads in fbx files from python 2

This requires a configs file, which contains the command necessary to switch conda
environments to run the fbx reading script from python 2
"""

from ....core import logger

import inspect
import os

import numpy as np

# Get the current folder to import the config file
current_folder = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])
)


def fbx_to_array(fbx_file_path, fbx_configs, root_joint, fps):
    """
    Reads an fbx file to an array.

    Currently reading of the frame time is not supported. 120 fps is hard coded TODO

    :param fbx_file_path: str, file path to fbx
    :return: tuple with joint_names, parents, transforms, frame time
    """

    # Ensure the file path is valid
    fbx_file_path = os.path.abspath(fbx_file_path)
    assert os.path.exists(fbx_file_path)

    # Switch directories to the env_utils folder to ensure the reading works
    previous_cwd = os.getcwd()
    os.chdir(current_folder)

    # Call the python 2.7 script
    temp_file_path = os.path.join(current_folder, fbx_configs["tmp_path"])
    python_path = fbx_configs["fbx_py27_path"]
    logger.info("executing python script to read fbx data using Autodesk FBX SDK...")
    command = '{} fbx_py27_backend.py "{}" "{}" "{}" "{}"'.format(
        python_path, fbx_file_path, temp_file_path, root_joint, fps
    )
    logger.debug("executing command: {}".format(command))
    os.system(command)
    logger.info(
        "executing python script to read fbx data using Autodesk FBX SDK... done"
    )

    with open(temp_file_path, "rb") as f:
        data = np.load(f)
        output = (
            data["names"],
            data["parents"],
            data["transforms"],
            data["fps"],
        )

    # Remove the temporary file
    os.remove(temp_file_path)

    # Return the os to its previous cwd, otherwise reading multiple files might fail
    os.chdir(previous_cwd)
    return output
