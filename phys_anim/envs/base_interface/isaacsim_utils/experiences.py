import os


def get_experience(headless, enable_livestream, enable_recording=False):
    base_path = os.path.dirname(__file__) + '/apps/'
    if headless:
        if enable_recording:
            experience = base_path + 'omni.isaac.sim.python.headless.camera.kit'
        else:
            experience = base_path + 'omni.isaac.sim.python.headless.kit'
    else:
        if enable_recording:
            experience = base_path + 'omni.isaac.sim.python.camera.kit'
        else:
            experience = base_path + 'omni.isaac.sim.python.kit'
    return experience
