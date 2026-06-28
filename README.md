
![Simulator preview](/assets/images/img6.png)

<!-- ## DEMO -->

<!-- ### 26/06/22
https://youtu.be/ZT2nsVZF6J0

00:03 ~ 01:14 Kinematic simulator Demo

01:17 ~ 02:32 Dynamic simulator (position) Demo

02:35 ~ 03:29 Dynamic simulator (motor) Demo

### 26/06/12
https://youtu.be/5F9DRPQdj8Y

00:03 ~ 01:03 Kinematic simulator Demo

01:06 ~ 02:30 Dynamic simulator (position) Demo

02:33 ~ 03:31 Dynamic simulator (motor) Demo -->

## Simulator Overview

MyJoCo(feat. MuJoCo)는 로봇 모델의 다른 제어 방식을 이해하고 실험할 수 있는 MuJoCo-based 로봇 시뮬레이터 구축 프로젝트입니다. MJCF을 제외한 MyJoCo 코드는 vibe coding 없이 from scratch로 제작되었습니다.

Check out the more detailed implementation journey here !

https://leewoojye.github.io/robotics/research/2026/06/03/myjoco3.html

[update] A newsletter feature has been added to personal blogs. If you would like to receive the newsletter, please subscribe !

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

Run the DexJoCo teleop demo:

```bash
git clone --depth 1 https://github.com/brave-eai/dexjoco.git temp/dexjoco_src
python3 -m sim_with_mujoco.dexjoco_teleop
```

```bash
# If DexJoCo is already cloned elsewhere:
DEXJOCO_XML_PATH=/path/to/arena_arm_hand_bucket_pick.xml python3 -m sim_with_mujoco.dexjoco_teleop
```

Use the right-side GUI panels to move the Panda-Allegro target and control the thumb/finger grasp sliders.

## Core Implementation

| 엔트리 파일 | 상태 갱신 방식 | 궤적 형성 방식 | ctrl 입력 | 물리 계산 | 용도 |
| --- | --- | --- | --- | --- | --- |
| dexjoco_teleop.py | DexJoCo Panda-Allegro 모델을 로드하고 arm/site 제어와 hand ctrl을 mj_step으로 갱신 | panel target을 attachment_site pose target으로 직접 사용, grasp slider는 open/close alpha로 매핑 | Panda arm은 operational-space torque를 motor ctrl에 입력, Allegro hand는 position actuator qpos target 입력 | mj_step 사용, hand-object 접촉 friction/condim 및 boxed_food 질량을 런타임 조정 | DexJoCo Panda + Allegro hand teleop grasp demo |

| 영역 | 구현 내용 |
| --- | --- |
| Environment, Viewer class | model, data, viewer, mujoco API wrapper를 묶어서 관리하는 Environment class / GLFW와 mujoco rendering API를 묶은 Viewer class, camera 조종 패널 추가, event handler에서 polling 중심 구조로 변경 (reference: dm_control) |
| Kinematic simulator | IK 결과를 data.qpos에 직접 반영하고 mj_forward로 상태를 갱신 |
| Dynamic simulator | IK 결과를 actuator ctrl에 넣고 mj_step으로 mujoco dynamics 진행 |
| 시뮬레이션 공통 | rendering, polling, trajectory generation, simulation 시간축 분리 및 적절한 주기(ex. trajectory_duration, poll_interval) 탐색 |
| Multi target IK | multi target의 jacobian과 error를 쌓는 get_stacked_ik 함수, damped least squares로 IK 계산, 클리핑 로직 최적화 |
| Differential IK | actual state 기준으로 multi target jacobian을 구성하고, least-squares로 qvel target과 다음 q target 계산 |
| Trajectory | pose interpolation(시작점 속도/가속도가 비영으로 부드러운 궤적 전환 도모), joint-space interpolation |
| Dynamics utility | computed torque, PD controller 모듈을 구현하고, mujoco timestep마다 각각 팔과 손가락 제어를 담당 (reference: robosuite) |
| Collision utility | mujoco data.contact 기반 robot-table, finger-object 접촉 판정 |
| mujoco utility | 편의를 위한 mujoco API wrapper (ex. joint id, dof id, actuator id 매핑) |
| Assets | motor actuator로 구성된 ffw MJCF 파일 추가, 손가락 마디 사이에 self-collision을 exclude 태그로 임시 방지 |

<!-- | Planning(현재 미사용) | joint trajectory planner: 궤적들 간 qvel을 공유 + singularity check (reference: ROS2) | -->

### Simulator Structure

```text
sim_with_mujoco/
  dexjoco_teleop.py            DexJoCo Panda arm + Allegro hand teleop entry
  environment/
    env.py                      model, data, viewer wrapper
  mjcf/
    parser.py                   MJCF parser
  utils/
    ik.py                       DLS multi-target IK
    ik_qp.py                    differential IK
    dynamics.py                 CT, PD, task-space control
    kinematics.py               finger / kinematics helper
    collision.py                contact helper
    math3d.py                   transform / twist helper
    mj.py                       mujoco id mapping helper
    planning.py                 waypoint / trajectory helper
  viewer/
    viewer.py                   renderer wrapper
    glfw_panel.py               hand target panel
    gui_panel.py                camera control panel
sim/model/motion/
  trajectory.py                 task / joint trajectory math
temp/
  dexjoco_src/                  external DexJoCo clone used by dexjoco_teleop.py
```

## Limitation (Future Implementation Plan)
 
- IK의 damping과 dq 제한이 singularity, collision 조건까지 만족시키지 않습니다. 별도 planning 모듈을 추가하거나 Differential IK 모듈 제약을 보완해야 합니다.
- 향후에 task-space PD외에 mass matrix, null-space posture control을 포함한 operational space controller를 구현해야 합니다.
- finger interpolation은 grasp synergy를 모델링하지 않고 open/close alpha를 관절 목표로 직접 매핑해서 손의 실제 닫힘을 충분히 표현하지 못합니다.
- 현재 trajectory duration은 고정값인데, 인접 velocity 등을 바탕으로 동적으로 바꿔볼 수 있습니다.
- 현재 충돌 감지 모듈은 kinematic rollback 중심이라 접촉면을 따라 미끄러짐을 표현하지 못합니다. 또한 robot-table hard collision에 초점이 맞춰진 모듈을 확장해야 합니다. 한편 kinematic mode 기능 범위가 헷갈려 사용처를 더 조사해야 합니다.
- DexJoCo hand grasp는 tactile feedback, contact-aware grasp planner, force closure optimization 없이 position actuator target과 MuJoCo contact solver에 의존합니다. 작은 물체를 안정적으로 잡으려면 finger force/impedance control 또는 retargeting 기반 hand posture가 추가로 필요합니다.

<!-- - DexJoCo object의 friction, contact dimension, boxed_food mass는 demo 조작감을 위해 런타임에서 조정됩니다. 실제 물체 물성 검증이나 실기 재현을 주장하려면 질량, 관성, 마찰, pad compliance를 따로 식별해야 합니다.
- DexJoCo asset은 repo 내부에 vendoring하지 않고 `temp/dexjoco_src` 또는 `DEXJOCO_XML_PATH`로 참조합니다. 따라서 fresh clone 환경에서는 How to Use의 clone 단계가 필요합니다.
- 점진적으로 강화학습 데모를 붙여봅니다. -->

## References

- mujoco documentation
- Modern Robotics, Kevin M. Lynch and Frank C. Park: Ch. 3 Rigid-Body Motions, Ch. 5 Velocity Kinematics and Statics, Ch. 6 Inverse Kinematics, Ch. 8 Dynamics of Open Chains, Ch. 9 Trajectory Generation, Ch. 11 Robot Control, Ch. 12 Grasping and Manipulation
- Drake Differential IK: https://drake.mit.edu/doxygen_cxx/group__planning__kinematics.html
- robosuite Controllers: https://robosuite.ai/docs/modules/controllers.html
- dm_control: https://github.com/google-deepmind/dm_control
- DexJoCo: https://github.com/brave-eai/dexjoco

## Tech Stack

- MuJoCo
- GLFW
- SciPy
- matplotlib
- Python
- NumPy
