import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import protomotions.robot_configs as robot_configs

_BUILTIN_DIR = Path(robot_configs.__file__).parent

_PROBE = """
import json, sys
from protomotions.robot_configs.factory import registered_robot_names, robot_config
names = registered_robot_names()
files = {}
for name in sys.argv[1:]:
    module = type(robot_config(name)).__module__
    files[name] = [module, sys.modules[module].__file__]
print(json.dumps({"names": list(names), "files": files}))
"""


def _probe(robot_config_path, *robots):
    env = dict(os.environ)
    env.pop("PROTOMOTIONS_ROBOT_CONFIG_PATH", None)
    if robot_config_path is not None:
        env["PROTOMOTIONS_ROBOT_CONFIG_PATH"] = robot_config_path
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, *robots],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_unset_path_keeps_the_builtin_registry():
    builtin = sorted(
        p.stem
        for p in _BUILTIN_DIR.glob("*.py")
        if p.stem not in {"base", "factory", "motor_classes", "__init__"}
    )
    assert _probe(None)["names"] == builtin
    assert _probe("")["names"] == builtin


def test_path_directories_register_outside_robots(tmp_path):
    outside = tmp_path / "robots"
    outside.mkdir()
    shutil.copy(_BUILTIN_DIR / "amp.py", outside / "outside_amp.py")
    shutil.copy(_BUILTIN_DIR / "amp.py", outside / "g1.py")
    empty = tmp_path / "empty"
    empty.mkdir()

    result = _probe(os.pathsep.join([str(empty), str(outside)]), "outside_amp", "g1")

    assert "outside_amp" in result["names"]
    assert result["files"]["outside_amp"] == [
        "protomotions.robot_configs.outside_amp",
        str(outside / "outside_amp.py"),
    ]
    assert result["files"]["g1"] == [
        "protomotions.robot_configs.g1",
        str(_BUILTIN_DIR / "g1.py"),
    ]
