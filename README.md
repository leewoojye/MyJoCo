
![Simulator preview](/assets/images/img0.png)

## Simulator Overview

This project is a custom robotics simulator for experimenting with a ROBOTIS FFW-SH5 humanoid upper-body robot, a five-finger hand, and simple object interaction tasks such as pushing, grasping, and lifting a can.

The simulator loads MuJoCo XML assets, converts robot geometry into Open3D meshes, and runs a kinematics-based interaction loop.

Check out the more detailed implementation journey here! https://leewoojye.github.io/research/2026/05/02/sim_from_scratch.html

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

#### Robot model

- `BodyNode`, `GeomRecord`, `RobotModel`, `RobotState`로 구조를 분리

#### FK

- PoE 기반 recursive FK와 mesh delta transform
- IK, FK 모듈은 계산만 하고 상태 변화시키지 않는 함수와 계산, 상태변화를 동시에 수행하는 함수로 구분됨

#### IK

- Position IK, pose IK, matrix log error를 구현
- position IK와 pose IK를 구분하고, 본 레포에서는 오른손에 pose IK를, 왼손에는 position IK를 적용함

#### Trajectory

- 0.1초 cubic time scaling 혹은 quantic time scaling을 적용
- 궤적 형성 주체는 타겟 이벤트 핸들러이고, 궤적을 실제로 FK로 옮기는 주체는 tick 이벤트 핸들러임

#### Collision

- Right hand-object/table 중심 pair filter와 FCL proxy distance를 구현
- 로봇팔과 손가락은 capsule primitive로, 테이블은 box primitive로 근사한 뒤 python fcl 라이브러리를 이용해 두 프록시 사이의 거리 계산
- capsule은 cylinder와 비교해 두 프록시 간 거리 계산이 용이해 대부분의 mesh를 capsule로 근사함
- 

#### Contact

- normal, tangent, relative velocity, 트위스트를 필드로 갖는 접촉점 클래스 ContactPoint를 정의
- ContactPoint 생성자는 외부에서 받은 거리를 이용해 관통 거리를 구함

#### Grasp

- Finger alpha interpolation과 form/force closure 조건을 만족하는지 판별하는 함수
- 비침투 조건 판별
- solve_contact_force()를 통해 force closure 조건을 확인하는 동시에 force closure가 참이라면 그때의 접촉점 후보들을 반환

#### GUI

- Target pose panel과 grasp panel 구현

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
- velocity-limited servo
- mass matrix, actuator dynamics
- damping/step limit
- loader_without_mujoco.py

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
    solver/           Contact, form closure, force closure, and contact response.
    kinematics/         FK, IK, Jacobian, and twist calculations.
    math3d/             Lie group, screw, transform, and vector utilities.
    motion/             Trajectory and smoothing utilities.
    robot/              Robot model, state, joints, bodies, and geometry loading.
  view/                 Open3D renderer and GUI panels.
tests/                  Experimental scripts and tests.
docs/                   Notes and study references.
```

## References

- Modern Robotics Official lecture slides: Ch. 2 Configuration Space, Ch. 3 Rigid-Body Motions, Ch. 4 Forward Kinematics, Ch. 5 Velocity Kinematics and Statics, Ch. 6 Inverse Kinematics, Ch. 8 Dynamics of Open Chains, Ch. 9 Trajectory Generation, and Ch. 12 Grasping and Manipulation.
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
