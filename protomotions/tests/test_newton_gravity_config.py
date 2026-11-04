from types import SimpleNamespace

import numpy as np
import pytest

from protomotions.simulator.base_simulator.config import ProjectileConfig
from protomotions.simulator.newton.config import (
    NewtonSimParams,
    NewtonSimulatorConfig,
)

_BALL = """<mujoco>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_default_gravity_matches_the_previous_constant():
    assert NewtonSimParams().gravity == -9.81


@pytest.mark.parametrize("gravity", [None, -1.62, 0.0])
def test_newton_model_takes_gravity_from_the_sim_config(tmp_path, gravity):
    wp = pytest.importorskip("warp")
    pytest.importorskip("newton")
    from protomotions.simulator.newton.simulator import NewtonSimulator

    asset = tmp_path / "ball.xml"
    asset.write_text(_BALL)
    sim_params = NewtonSimParams() if gravity is None else NewtonSimParams(gravity=gravity)

    simulator = object.__new__(NewtonSimulator)
    simulator.config = NewtonSimulatorConfig(
        num_envs=2, headless=True, sim=sim_params, experiment_name="gravity"
    )
    simulator.num_envs = 2
    simulator.terrain = None
    simulator.robot_config = SimpleNamespace(
        asset=SimpleNamespace(fix_base_link=False, self_collisions=False),
        number_of_actions=0,
    )
    simulator._prepared_robot_asset_path = str(asset)
    simulator._proj_config = ProjectileConfig(num_projectiles=0)
    simulator._configure_builder_joint_properties = lambda: None

    with wp.ScopedDevice("cpu"):
        simulator._create_envs()

    expected = -9.81 if gravity is None else gravity
    np.testing.assert_allclose(
        np.asarray(simulator.model.gravity.numpy()).reshape(-1, 3),
        np.tile([0.0, 0.0, expected], (2, 1)),
        rtol=0,
        atol=1e-6,
    )
