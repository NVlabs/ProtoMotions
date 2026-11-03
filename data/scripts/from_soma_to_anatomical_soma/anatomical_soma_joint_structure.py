# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Anatomical hinge orders in the SOMA23 basis (+X left, -Y forward, +Z up).

Suffixes identify physical signed axes, not their position in the chain.
Upper arms retain SOMA's T-pose frames and use XZY as recommended by the
trajectory conditioning study; this is not an arms-down robot PRY chain.
"""

JOINT_ORDERS = {
    "Spine1": "zyx",
    "Spine2": "zyx",
    "Chest": "zyx",
    "Neck1": "zyx",
    "Neck2": "zyx",
    "Head": "zyx",
    "Shoulder": "xyz",
    "Arm": "xzy",
    "ForeArm": "xyz",
    "Hand": "yxz",
    "Leg": "xyz",
    "Shin": "xyz",
    "Foot": "xyz",
    "ToeBase": "xyz",
}


def joint_order(body_name):
    part = body_name.removeprefix("Left").removeprefix("Right")
    return JOINT_ORDERS[part]


def reorder_soma_tree(tree, *, serial):
    """Reorder coincident hinges without changing axes, ranges or rest geometry.

    Serial anatomical assets have two zero-length carriers per triple. Ordinary
    generalized templates keep three hinges on each semantic body. Joint and
    actuator names remain physical-axis identifiers across both representations.
    """
    root = tree.getroot()
    world = root.find("worldbody")
    bodies = list(world.iter("body"))
    groups = {}
    for body in bodies:
        for joint in body.findall("joint"):
            if joint.get("type") == "hinge":
                name, suffix = joint.get("name").rsplit("_", 1)
                groups.setdefault(name, []).append((body, joint, suffix))
    for name, entries in groups.items():
        if len(entries) == 1:
            continue
        if len(entries) != 3 or {e[2] for e in entries} != set("xyz"):
            raise ValueError(f"Expected one or three named SOMA hinges: {name}")
        by_axis = {suffix: joint for _, joint, suffix in entries}
        ordered = [by_axis[axis] for axis in joint_order(name)]
        if serial:
            if len({id(body) for body, _, _ in entries}) != 3:
                raise ValueError(f"Expected serial carriers: {name}")
            for body, joint, _ in entries:
                body.remove(joint)
            for (body, _, _), joint in zip(entries, ordered):
                body.insert(0, joint)
        else:
            body = entries[0][0]
            if any(b is not body for b, _, _ in entries):
                raise ValueError(f"Expected one semantic body: {name}")
            index = min(list(body).index(joint) for _, joint, _ in entries)
            for _, joint, _ in entries:
                body.remove(joint)
            for offset, joint in enumerate(ordered):
                body.insert(index + offset, joint)
    # Simulator actuator order must agree with the new depth-first DOF order.
    actuator = root.find("actuator")
    if actuator is not None:
        by_joint = {a.get("joint"): a for a in actuator}
        if len(by_joint) != len(actuator):
            raise ValueError("Expected one joint actuator per hinge")
        actuator[:] = [
            by_joint[j.get("name")]
            for b in world.iter("body")
            for j in b.findall("joint")
            if j.get("name") in by_joint
        ]
    return tree
