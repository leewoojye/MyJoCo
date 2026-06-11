
![Simulator preview](/assets/images/img3.png)

## DEMO

### 26/06/11
https://youtu.be/gHD_W7J2Uig

00:00 ~ 01:00 Kinematic Simulator Demo

01:00 ~ 02:00 Dynamic Simulator Demo

## Simulator Overview

This project is a custom robotics simulator for experimenting with a ROBOTIS FFW-SH5 humanoid upper-body robot, a five-finger hand, and simple object interaction tasks such as pushing, grasping, and lifting a can.

The simulator loads MuJoCo XML assets, converts robot geometry into Open3D meshes, and runs a kinematics-based interaction loop.

Check out the more detailed implementation journey here!



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

Use the right-side GUI panels to move the right hand target and control the right-hand grasp sliders.
[update] A GUI panel for adjusting the viewpoint has been added within the viewer.

## Core Implementation

| 영역 | 구현 내용 |
| --- | --- |
| Environment wrapper | model, data, viewer, 초기 qpos, step(), render()를 묶어서 관리하는 Environment 클래스 |
| Viewer | GLFW window + MjvScene + MjrContext + mjr_render()을 통합한 뷰어 클래스, event handler에서 polling 중심으로 변경 |
| GUI panel | target pose 슬라이더와 camera 슬라이더를 MuJoCo(GLFW) 위에 그림 |
| Kinematic simulation | IK 결과를 data.qpos에 직접 반영하고 mj_forward()로 상태를 갱신 |
| Dynamic simulation | IK 결과를 actuator ctrl에 넣고 mj_step()으로 MuJoCo dynamics를 진행 |
| 시뮬레이션 공통 | rendering, polling(trajectory generating), simulation 주기 분리 및 적절한 주기(ex. trajectory_duration, poll_interval) 탐색 |
| Multi IK | 오른손은 pose IK, 왼손은 position IK를 적용하는 get_stacked_ik() |
| Trajectory | position, rotation을 동시에 보간하는 interpolate_pose(), joint space에서 인접한 두 궤적이 qvel을 공유하게 해 부드러움을 도모하는 interpolate_joint_ros()(reference: ROS) 구현 |
| Dynamics utility | solve_inverse_dynamics, computed torque 모듈 구현 |
| Collision utility | MuJoCo data.contact 기반 robot-table hard collision 판정 |
| MuJoCo utility | 편의를 위한 MuJoCo API wrapper (ex. joint id, dof id, actuator id를 매핑) |
| Experiments | joint_space_trajectory_test.py, joint_space_ctorque_test.py, pd_task_space_test.py |

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

### Limitation (future plan)
<!-- - 현재 접촉점에서 물체의 회전을 고려하지 않고 있습니다.
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
-  -->

## References

- Modern Robotics Official lecture slides: Ch. 2 Configuration Space, Ch. 3 Rigid-Body Motions, Ch. 4 Forward Kinematics, Ch. 5 Velocity Kinematics and Statics, Ch. 6 Inverse Kinematics, Ch. 8 Dynamics of Open Chains, Ch. 9 Trajectory Generation, Ch. 11 Robot Control, and Ch. 12 Grasping and Manipulation.
- ROS 2 control joint trajectory controller trajectory documentation: https://control.ros.org/master/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html
- dm_control repository: [google-deepmind/dm_control](https://github.com/google-deepmind/dm_control)
- MuJoCo XML modeling and computation documentation.
- ROBOTIS MuJoCo Menagerie assets: https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie
- SPOTS object assets: https://github.com/joonhyung-lee/spots

## Tech Stack

- MuJoCo
- Python
- NumPy
- SciPy
- MuJoCo
- GLFW
- matplotlib / mediapy for experimental scripts
