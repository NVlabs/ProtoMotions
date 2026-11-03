# SOMA23 to Anatomical SOMA: Trajectory Solver

Run this code in the separate PyRoki/JAX environment. The
[ProtoMotions data driver](../../data/scripts/from_soma_to_anatomical_soma/README.md)
launches it automatically when converting a packed motion library.

| File | Responsibility |
| --- | --- |
| `solve_trajectory.py` | Fit an entire motion trajectory to the anatomical hinges and enforce solver acceptance criteria. |
| `analytic_kinematics.py` | Forward kinematics with analytic derivatives for the solver. |
| `banded_trajectory_solver.py` | Solve the temporally coupled least-squares system using its banded structure. |
| `test_*.py` | Regression tests for trajectories, derivatives and linear algebra. |

The math modules are solver implementation details, not alternate robot models.
The checked-in target is `protomotions/data/assets/mjcf/anatomic_soma.xml`.
SOMA23 remains the unchanged source model.

From the repository root, in the PyRoki environment:

```bash
JAX_PLATFORMS=cpu python -m pytest pyroki/from_soma_to_anatomical_soma/ -q
python pyroki/from_soma_to_anatomical_soma/solve_trajectory.py --help
```

See the [anatomical SOMA guide](../../docs/source/robots/anatomic_soma.rst)
for coordinate conventions and retargeting guarantees and limitations.

## Solver Notes

One `TrajectorySolver` is reused across clips; `--cache-dir` enables JAX's disk
compilation cache, and in-memory executables are cleared after sixteen distinct
frame shapes. `--frame-bucket-size N` pads every clip to a multiple of N frames
so fewer shapes compile; padding is masked out of every cost and out of the
convergence test, so clip content and temporal boundaries are unchanged (the
released corpus used 128).

The objective couples three consecutive frames through the acceleration terms,
giving block-pentadiagonal normal equations. Grouping adjacent frames into pairs
turns them block-tridiagonal, solved by Levenberg-Marquardt with a
cyclic-reduction preconditioned conjugate gradient; forward kinematics and its
derivatives are analytic (`analytic_kinematics.py`, checked against autodiff and
finite differences in `test_analytic_kinematics.py`). Arithmetic is FP32 with
full-precision matrix multiplication. When a hinge seed exceeds the rate budgets
a two-frame Gaussian filter smooths only the initial coordinates; source poses,
timing, costs and acceptance budgets are untouched and outputs are never
filtered or clipped.
