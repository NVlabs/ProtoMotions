# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motor classes: one physical actuator type, its gains derived from armature.

A robot's PD gains are not hand-typed per joint; they follow from the motor
driving the joint. A :class:`MotorClass` names one actuator (its rotor
``armature`` and torque/speed limits) and derives the critically-damped PD
gains from a natural frequency and damping ratio, exactly as BeyondMimic does:

    stiffness = armature * (2*pi*f)**2
    damping   = 2 * damping_ratio * armature * (2*pi*f)

``control_info(scale)`` returns a :class:`ControlInfo` for one joint. ``scale``
multiplies the gains **and** the armature (two identical motors in parallel on
one joint scale the inertia and gains, not the speed limit), so a double-motor
ankle is ``MotorClass.control_info(scale=2)``. The effort and velocity limits
are the motor's by default; a joint that carries a smaller actuator than its
gain class overrides them with ``effort_limit=`` / ``velocity_limit=``.

The literal ``3.1415926535`` matches the value the per-robot formula blocks
used, so the derived gains are bit-identical to the hand-expanded constants.
"""

from dataclasses import dataclass
from typing import Optional

from protomotions.components.pose_lib import ControlInfo

# The exact pi literal the BeyondMimic gain blocks used; kept verbatim so the
# derived stiffness/damping are bit-identical to the pre-refactor constants.
_PI = 3.1415926535


@dataclass(frozen=True)
class MotorClass:
    """One actuator type: armature plus torque/speed limits, gains derived."""

    armature: float
    effort_limit: float
    velocity_limit: float
    natural_freq_hz: float = 10.0
    damping_ratio: float = 2.0

    def control_info(
        self,
        scale: float = 1.0,
        *,
        effort_limit: Optional[float] = None,
        velocity_limit: Optional[float] = None,
    ) -> ControlInfo:
        """PD gains and limits for one joint driven by this motor.

        ``scale`` multiplies the armature and the derived stiffness/damping
        (parallel motors), leaving the speed limit alone. ``effort_limit`` /
        ``velocity_limit`` override the motor's limits for a joint whose
        actuator differs from its gain class; ``None`` keeps the motor's value.
        """
        omega = self.natural_freq_hz * 2.0 * _PI
        return ControlInfo(
            stiffness=scale * (self.armature * omega**2),
            damping=scale * (2.0 * self.damping_ratio * self.armature * omega),
            armature=scale * self.armature,
            effort_limit=(self.effort_limit if effort_limit is None else effort_limit),
            velocity_limit=(
                self.velocity_limit if velocity_limit is None else velocity_limit
            ),
        )


# Unitree G1 actuators (BeyondMimic). Names encode the motor size.
G1_5020 = MotorClass(armature=0.003609725, effort_limit=25, velocity_limit=37)
G1_7520_14 = MotorClass(armature=0.010177520, effort_limit=88, velocity_limit=32)
G1_7520_22 = MotorClass(armature=0.025101925, effort_limit=139, velocity_limit=20)
G1_4010 = MotorClass(armature=0.00425, effort_limit=5, velocity_limit=22)

# Unitree H1_2 actuators, named by their torque rating in N·m.
H1_2_300 = MotorClass(armature=0.040, effort_limit=300, velocity_limit=50)
H1_2_200 = MotorClass(armature=0.030, effort_limit=200, velocity_limit=50)
H1_2_60 = MotorClass(armature=0.010, effort_limit=60, velocity_limit=50)
H1_2_40 = MotorClass(armature=0.005, effort_limit=40, velocity_limit=50)
