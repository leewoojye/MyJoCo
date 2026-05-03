# Modern Robotics chapter examples

These examples were prepared from the local `MR-slides-Nov-24-2020` material.
The available local slide/PDF set contains Ch03, Ch04, Ch05, and Ch06. No Ch07
folder or Ch07 PDF was present in that download, so the Ch07 example follows
the standard Modern Robotics Chapter 7 topic: closed-chain kinematics.

Run examples from this directory or from the repository root:

```bash
/Users/woojyelee/workspace/my_robotics/venv/bin/python modern_robotics/ch03/rigid_body_motions.py
```

Each script saves a PNG into `modern_robotics/outputs/` and opens a Matplotlib
window. For terminal-only checks, pass `--no-show`.

```bash
/Users/woojyelee/workspace/my_robotics/venv/bin/python modern_robotics/ch05/jacobian_statics_manipulability.py --no-show
```

## Chapter map

- `ch03/rigid_body_motions.py`: SO(3), SE(3), exponential/log coordinates,
  twists, screw motion, adjoint transform, and wrench power invariance.
- `ch04/forward_kinematics_poe.py`: product of exponentials in space/body
  frames, home configuration, screw axes, and workspace sampling.
- `ch05/jacobian_statics_manipulability.py`: space/body Jacobians, rank,
  singularities, manipulability ellipses, velocity limits, and statics
  `tau = J.T F`.
- `ch06/inverse_kinematics_newton.py`: numerical inverse kinematics,
  Newton-Raphson updates, damped pseudoinverse, convergence history, and
  multiple IK solutions from different initial guesses.
- `ch07/closed_chain_four_bar.py`: loop-closure equations, constraint
  Jacobian, continuation with Newton solves, velocity constraints, and
  closed-chain singularity indicators.
