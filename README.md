
![Simulator preview](/assets/images/img0.png)

## Simulator Overview

This project is a custom robotics simulator for experimenting with a ROBOTIS FFW-SH5 humanoid upper-body robot, a five-finger hand, and simple object interaction tasks such as pushing, grasping, and lifting a can.

The simulator loads MuJoCo XML assets, converts robot geometry into Open3D meshes, and runs a kinematics-based interaction loop.

⭐️ Check out the more detailed implementation journey here!⭐️ https://leewoojye.github.io/research/2026/05/02/sim_from_scratch.html

## How to Use

Create and activate a conda environment:

```bash
conda create -n my_robotics python=3.12
conda activate my_robotics
```

Install the required Python packages from the project root:

```bash
pip install -r requirements.txt
```

Run the simulator:

```bash
python3 main.py
```

Use the right-side GUI panels to move the right hand target and control the right-hand grasp sliders.

## Core Implementation

<!-- - MuJoCo XML loading and robot model construction.
- Forward kinematics and inverse kinematics for arm and hand control.
- Geometric Jacobian and body twist utilities.
- Collision and contact detection between the robot hand, table, and object.
- Contact point representation with normal, force, penetration depth, and relative velocity.
- Form closure and force closure checks for grasp evaluation.
- Simple contact-based object motion for pushing and grasping experiments. -->

## Additional Implementation

<!-- - Open3D-based rendering and GUI panels.
- Proxy geometry for faster collision distance checks.
- Basic contact force summation and object translation.
- Simple grasp attachment behavior after a valid grasp state.
- Local asset copies under the project-level `assets/` directory. -->

## Future Implementation Plan

## Project Structure

```text
assets/
  objects/              Object XML, mesh, and texture files.
  robots/               Robot XML and mesh assets.
sim/
  controller/           Main simulator control loop.
  model/
    collision/          Collision checking, proxy geometry, and distance logic.
    dynamics/           Basic dynamics and integration utilities.
    grasping/           Contact, form closure, force closure, and contact response.
    kinematics/         FK, IK, Jacobian, and twist calculations.
    math3d/             Lie group, screw, transform, and vector utilities.
    motion/             Trajectory and smoothing utilities.
    robot/              Robot model, state, joints, bodies, and geometry loading.
  view/                 Open3D renderer and GUI panels.
tests/                  Experimental scripts and tests.
docs/                   Notes and study references.
```

## References

- Modern Robotics Official lecture slides: Ch. 2 Configuration Space, Ch. 3 Rigid-Body Motions, Ch. 4 Forward Kinematics, Ch. 5 Velocity Kinematics and Statics, Ch. 6 Inverse Kinematics, Ch. 8 Dynamics of Open Chains, Ch. 9 Trajectory Generation, Ch. 11 Robot Control, and Ch. 12 Grasping and Manipulation.
- MuJoCo XML modeling and computation documentation.
- ROBOTIS MuJoCo Menagerie assets: https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie
- SPOTS object assets: https://github.com/joonhyung-lee/spots

## Tech Stack

- Python
- NumPy
- SciPy
- MuJoCo
- Open3D
- python-fcl
