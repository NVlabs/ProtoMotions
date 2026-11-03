# From SOMA to Anatomical SOMA

Offline tools for preparing `anatomic_soma` assets and motion data.
These do not replace or modify the existing `soma23` robot.
Run the commands below from the repository root.

## Which Script Do I Need?

| Script | Purpose | When to use it |
| --- | --- | --- |
| `retarget_motions.py` | Fit packed SOMA motions to anatomical SOMA's ordered hinges. | Preparing motions for the anatomical robot. |
| `convert_soma_scene.py` | Transform associated scene poses into the same coordinate convention. | The source motions interact with scene objects. |
| `build_soma_joint_structure.py` | Reorder hinges and actuators in an existing MJCF geometry template. | Developing or regenerating robot assets, not normal training. |

For training with the checked-in anatomical asset, skip the asset builder.
Use motions and checkpoints made for `anatomic_soma`; SOMA23 joint coordinates
and policies are not interchangeable with it.

The offline helpers live here too: `anatomical_soma_joint_structure.py` defines
hinge ordering, and `soma23_to_anatomical_soma_retargeting.py` implements
coordinate conversion, hinge initialization and quality metrics. They are not
part of the training runtime.

The JAX solver has a separate environment and lives under
[`pyroki/from_soma_to_anatomical_soma`](../../../pyroki/from_soma_to_anatomical_soma/README.md).

## Retarget Motions

Run the driver in the ProtoMotions Python environment. It launches the solver
using a separate Python environment containing PyRoki and its dependencies.
The [retargeting guide](../../../docs/source/tutorials/workflows/retargeting_pyroki.rst)
documents the solver environment.

```bash
python data/scripts/from_soma_to_anatomical_soma/retarget_motions.py \
  --input source.pt \
  --output anatomical.pt \
  --pyroki-python /path/to/pyroki/python \
  --cache-dir /path/to/jax-cache
```

The driver preserves clip identities and timing, fits joint trajectories and
recomputes velocities. It writes audit information alongside the packed output.
It refuses to overwrite existing outputs. Solver acceptance is not a guarantee
of collision-free motion or successful physical tracking.

## Convert Paired Scenes

```bash
python data/scripts/from_soma_to_anatomical_soma/convert_soma_scene.py \
  --input source_scenes.pt \
  --output anatomical_scenes.pt
```

Original SOMA faces -Y; anatomical SOMA faces +X, with Z up in both.
Motion conversion and scene conversion apply the same world-basis change.
The scene converter transforms object poses and scene offsets, preserving
object-local geometry and motion IDs. Keep the converted scenes paired with
their corresponding converted motions.

## Rebuild Joint Ordering (Asset Development Only)

```bash
python data/scripts/from_soma_to_anatomical_soma/build_soma_joint_structure.py \
  --source protomotions/data/assets/mjcf/anatomic_soma.xml \
  --output /tmp/anatomic_soma_rebuilt.xml \
  --serial
```

On the shipped asset this reproduces the file byte for byte, so it doubles as
a consistency check after editing the asset.

`--serial` expects an existing serial-hinge template with intermediate carrier
bodies. It does not create those bodies or reduce SOMA23 to 52 hinges.
Without `--serial`, the builder expects grouped hinges on their semantic bodies.

The builder preserves rest geometry, inertias, named axes and individual joint
limits, then applies the body-specific hinge ordering and matching actuator
ordering. Changing rotation order changes non-rest kinematics: retarget motions
again rather than permuting their existing joint coordinates.

For detailed contracts, see the
[anatomical SOMA guide](../../../docs/source/robots/anatomic_soma.rst).
