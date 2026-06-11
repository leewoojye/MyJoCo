
![Simulator preview](/assets/images/img3.png)

## DEMO

### 26/06/11
https://youtu.be/gHD_W7J2Uig

00:00 ~ 01:00 Kinematic Simulator Demo

01:00 ~ 02:00 Dynamic Simulator Demo

## Simulator Overview

Myjoco(feat. mujoco)는 구버전인 mujoco-free 시뮬레이터를 mujoco-based 시뮬레이터로 재탄생한 프로젝트입니다. mjcf을 제외한 Myjoco 코드는 from scratch로 만들어졌으며 제어 및 강화학습으로의 확장을 목표로 합니다.

Check out the more detailed implementation journey here!

https://leewoojye.github.io/research/2026/06/03/myjoco2.html

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

**Dynamic Simulator**

```bash
python3 -m sim_with_mujoco.dynamic_simulator
```

**Kinematic Simulator**

```bash
python3 -m sim_with_mujoco.kinematic_simulator
```

**Joint space computed torque (with motor actuator)**

```bash
python3 -m sim_with_mujoco.test.joint_space_ctorque_test
```

Use the right-side GUI panels to move the right hand target and control the right-hand grasp sliders.

[update] A GUI panel for adjusting the viewpoint has been added within the viewer.

## Core Implementation

| 영역 | 구현 내용 |
| --- | --- |
| Environment class | model, data, viewer, 초기 qpos, step( ), render( )를 묶어서 관리하는 Environment 클래스 |
| Viewer class | GLFW window + MjvScene + MjrContext + mjr_render( )을 통합한 뷰어 클래스, event handler에서 polling 중심으로 변경, GLFW 위에 camera 조종 패널 추가 |
| Kinematic simulation | IK 결과를 data.qpos에 직접 반영하고 mj_forward()로 상태를 갱신 |
| Dynamic simulation | IK 결과를 actuator ctrl에 넣고 mj_step( )으로 MuJoCo dynamics를 진행 |
| 시뮬레이션 공통 | rendering, polling, trajectory generation, simulation 시간축 분리 및 적절한 주기(ex. trajectory_duration, poll_interval) 탐색 |
| Multi IK | 다중 타겟의 jacobian, error를 쌓는 get_stacked_ik( ), damped least squares로 IK 계산, 클리핑 로직 최적화 |
| Trajectory | position, rotation을 동시에 보간하는 interpolate_pose( ), joint space에서 인접한 두 궤적이 qvel을 공유하게 해 부드러움을 도모하는 interpolate_joint_ros( )(reference: ROS) 구현 |
| Dynamics utility | solve_inverse_dynamics, computed torque 모듈 구현 |
| Collision utility | MuJoCo data.contact 기반 robot-table hard collision 판정 |
| MuJoCo utility | 편의를 위한 MuJoCo API wrapper (ex. joint id, dof id, actuator id를 매핑) |
| Experiments files | joint_space_ctorque_test.py, task_space_pd_test.py |

## Project Structure

```text
sim_with_mujoco/
  dynamic_simulator.py          ctrl 입력을 받고 env.step()을 호출하는 entry point
  kinematic_simulator.py        IK 결과를 data.qpos에 넣고 mj_forward()를 호출하는 entry point
  environment/
    env.py                      MjModel, MjData, viewer, step/render wrapper
  mjcf/
    parser.py                   MJCF에서 MjModel, MjData 추출
  utils/
    ik.py                       damped_pseudoinverse를 계산하고 multi-target IK 결과 반환
    kinematics.py               finger interpolation, body/site Jacobian helper
    dynamics.py                 inverse dynamics, computed torque, task-space PD
    collision.py                MuJoCo contact 기반 hard collision check (kinematic_simulator용)
    math3d.py                   body/site transform, body twist calculation
    mj.py                       joint id, dof id, actuator id mapping helper
  viewer/
    viewer.py                   MuJoCo renderer와 GUI panel 통합
    glfw_panel.py               hand target panel
    gui_panel.py                camera control panel
  test/
    joint_space_trajectory_test.py
    joint_space_ctorque_test.py
    pd_task_space_test.py
    dh_params_test.py
sim/model/motion/
    trajectory.py               quintic time-scaling 기반 task/joint space 자세 보간
```

## Limitation (Future Plan)

- IK 행렬이 단순히 stack한 구조라 task 간 우선순위가 없고 직관적으로 position IK를 적용한 왼손의 target error를 과소평가할 것 같습니다. 실제로 시뮬레이션 상에서 왼손은 오른손에 비해 자리를 이탈하는 경우가 많았습니다.
- IK의 damping과 dq 제한이 singularity, collision 조건까지 만족시키지 않습니다. planning 모듈을 추가해야 합니다.
- 향후에 task-space PD외에 mass matrix, null-space posture control을 포함한 operational space controller를 구현해야 합니다.
- finger interpolation은 grasp synergy를 모델링하지 않고 open/close alpha를 관절 목표로 직접 매핑해서 손의 실제 닫힘을 충분히 표현하지 못합니다.
- trajectory generator는 목표 변경을 부드럽게 만들지만 actuator torque, velocity, jerk limit을 동시에 만족하는 time-parameterization은 아닙니다.
- viewer diagnostics는 visual contact와 body BVH에 의존하고 있어 q_des saturation, ctrl saturation, qfrc_bias 보상 여부 같은 controller 내부 상태를 즉시 확인하기 어렵습니다. timestep별 mjData 대시보드도 만들면 좋을 것 같습니다. 생성된 궤적(ref)을 얼마나 준수했는지 평가하는 지표도 시각화할 수 있습니다.
- dm_control과 달리 현재 Environment wrapper는 state snapshot과 restore 경계가 약해서 IK용 임시 MjData, simulation MjData, viewer가 보는 MjData가 섞여있습니다.
- 현재 trajectory duration은 고정값인데, 인접 velocity 등을 바탕으로 동적으로 바꿔볼 수 있습니다.
- 현재 충돌 감지 모듈은 kinematic rollback 중심이라 접촉면을 따라 미끄러짐을 표현하지 못합니다. 또한 robot-table hard collision에 초점이 맞춰진 모듈을 확장해야 합니다. 한편 kinematic mode 기능 범위가 헷갈려 사용처를 더 조사해야 합니다.
- 실제 motor actuator를 상대로 joint-space computed torque를 실험해볼 필요가 있습니다.
- 점진적으로 강화학습 데모를 붙여봅니다.

## References

- MuJoCo XML modeling documentation
- MuJoCo computation and API documentation
- MuJoCo visualization documentation
- Gymnasium MuJoCo environment API
- Modern Robotics, Kevin M. Lynch and Frank C. Park: Ch. 3 Rigid-Body Motions, Ch. 5 Velocity Kinematics and Statics, Ch. 6 Inverse Kinematics, Ch. 8 Dynamics of Open Chains, Ch. 9 Trajectory Generation, Ch. 12 Grasping and Manipulation
- dm_control: https://github.com/google-deepmind/dm_control
- ROBOTIS MuJoCo Menagerie assets: https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie
- robosuite assets: https://github.com/ARISE-Initiative/robosuite

## Tech Stack

- MuJoCo
- Python
- NumPy
- SciPy
- MuJoCo
- GLFW
- matplotlib / mediapy for experimental scripts
