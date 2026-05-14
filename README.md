
![Simulator preview](/assets/images/img2.png)

## DEMO

### 26/05/13
https://youtu.be/gHD_W7J2Uig

캔과 손가락 사이의 접촉, 캔의 밀림이 더 잘 보이는 데모영상입니다.

### 26/05/11
https://youtu.be/q49b6CsoBwk

데모영상 초안입니다.

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

| 영역 | 구현 내용 |
| --- | --- |
| Robot model | BodyNode, GeomRecord, RobotModel, RobotState 클래스 |
| Math utils | rotation/transform/skew/jacobian matrix, hat 연산 등 |
| FK | PoE 기반 recursive FK |
| IK | position IK, pose IK |
| Trajectory | 0.1초 cubic/quantic time scaling |
| Collision | 손가락-캔/테이블 중심 필터링, proxy 클래스 |
| Contact | ContactPoint 클래스, normal/tangent 벡터 계산 |
| Grasp | alpha interpolation, form/force closure 판단 |
| Object update | 캔 가속도, 위치 업데이트 |
| GUI | target pose panel, grasp panel 클래스 |

<!-- #### Robot model

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

- Target pose panel과 grasp panel 구현 -->

<!-- - MuJoCo XML loading and robot model construction.
- Forward kinematics and inverse kinematics for arm and hand control.
- Geometric Jacobian and body twist utilities.
- Collision and contact detection between the robot hand, table, and object.
- Contact point representation with normal, force, penetration depth, and relative velocity.
- Form closure and force closure checks for grasp evaluation.
- Simple contact-based object motion for pushing and grasping experiments. -->

<!-- ## Additional Implementation -->

<!-- - Open3D-based rendering and GUI panels.
- Proxy geometry for faster collision distance checks.
- Basic contact force summation and object translation.
- Simple grasp attachment behavior after a valid grasp state.
- Local asset copies under the project-level `assets/` directory. -->

<!-- ## Future Implementation Plan
- velocity-limited servo
- mass matrix, actuator dynamics
- damping/step limit
- loader_without_mujoco.py -->

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

### Limitation (future implementation plan)
- 현재 접촉점에서 물체의 회전을 고려하지 않고 있습니다.
- 엑추에이터 힘을 별도로 계산하지 않고 고정된 힘 크기를 사용하고 있습니다.
- force closure를 판단하는 과정에서 원뿔을 네 개의 기저벡터로 근사합니다. 이때 각 기저벡터 앞에 붙는 가중치만 반환하고 실제 힘으로 복원하는 과정이 추가되어야 합니다.
- Manipulability을 평가하는 로직을 추가해야 합니다.
- 엔드이펙터를 뜻하는 문자열이 하드코딩되어 있습니다.
- 캔을 xy 평면에 수평한 방향으로 밀 때, z축 힘을 자른다는 아쉬움이 있습니다.
- 중력 및 기타 물리 법칙을 환경에 적용해야 합니다.
- tick loop 및 궤적 형성 과정을 최적화하여 더 부드러운 움직임을 도모해야 합니다.
- 오른손, 왼손에 연속적으로 IK를 적용하는 게 옳은지 검증해봐야 합니다. 7장 Kinematics of Closed Chains을 참고해야 합니다.
- capsule proxy 외에 cylinder, sphere 등 다양한 primitive type을 고려해야 합니다.
- 한 방향으로 힘이 더해질 때 누적되는 힘이 커지는 상황에서 에너지 보존 법칙 활용을 고려하고 있습니다. (레퍼런스: 무조코)
- 충돌 감지 로직에서 후보 state에 대해 hard collision이면 단순 rollback을 수행하고 있으며, 접촉 지점까지라도 FK가 적용될 수 있어야 합니다.
- 

## References

- Modern Robotics Official lecture slides: Ch. 2 Configuration Space, Ch. 3 Rigid-Body Motions, Ch. 4 Forward Kinematics, Ch. 5 Velocity Kinematics and Statics, Ch. 6 Inverse Kinematics, Ch. 8 Dynamics of Open Chains, Ch. 9 Trajectory Generation, and Ch. 12 Grasping and Manipulation.
- MuJoCo XML modeling and computation documentation.
- ROBOTIS MuJoCo Menagerie assets: https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie
- SPOTS object assets: https://github.com/joonhyung-lee/spots

## Tech Stack

- Python
- NumPy
- SciPy
- Open3D
- python-fcl
- MuJoCo
