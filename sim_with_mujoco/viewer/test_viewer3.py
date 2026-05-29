import time

import glfw
import mujoco
import numpy as np

from sim.model.math3d.rotation import rpy2rotation_matrix
from sim.model.motion.trajectory import interpolate_position, interpolate_position_simple
from sim_with_mujoco.environment.env import Environment
from sim_with_mujoco.utils.ik import solve_ik
from sim_with_mujoco.utils.kinematics import interpolate_finger
from sim_with_mujoco.utils.math3d import get_body_T
from sim_with_mujoco.utils.mj import actuator_ids_from_joints

XML_PATH = "/Users/woojyelee/workspace/my_robotics/assets/robots/robotis_ffw/scene_ffw_sh5.xml"


def main():
    initial_qpos = {
        "lift_joint": -0.15,
        "head_joint1": 0.0,
        "head_joint2": 0.0,
        "arm_l_joint1": 0.0,
        "arm_l_joint2": 0.0,
        "arm_l_joint3": 0.0,
        "arm_l_joint4": -1.57,
        "arm_l_joint5": 0.0,
        "arm_l_joint6": 0.0,
        "arm_l_joint7": 0.0,
        "arm_r_joint1": 0.0,
        "arm_r_joint2": 0.0,
        "arm_r_joint3": 0.0,
        "arm_r_joint4": -1.57,
        "arm_r_joint5": 0.0,
        "arm_r_joint6": 0.0,
        "arm_r_joint7": 0.0,
        "finger_l_joint1": 0.3,
        "finger_l_joint2": 1.57,
        "finger_l_joint3": -0.35,
        "finger_l_joint4": -0.25,
        "finger_r_joint1": 0.3,
        "finger_r_joint2": -1.57,
        "finger_r_joint3": 0.35,
        "finger_r_joint4": 0.25,
    }
    env = Environment(XML_PATH, "arm_r_link7")  # MjModel, MjData, viewer 초기화 및 e.e 지정
    env.initial_qpos(initial_qpos)

    ik_joint_names = [
        "lift_joint",
        "arm_r_joint1",
        "arm_r_joint2",
        "arm_r_joint3",
        "arm_r_joint4",
        "arm_r_joint5",
        "arm_r_joint6",
        "arm_r_joint7",
    ]
    env.viewer.init_viewer(env.initial_target_pos)
    initial_target_pos = env.initial_target_pos
    initial_pose = env.initial_pose

    # 궤적 형성 관련 변수
    trajectory_start = initial_target_pos.copy()
    trajectory_goal = initial_target_pos.copy()
    current_target = initial_target_pos.copy()
    trajectory_start_time = None
    trajectory_duration = 0.05  # 고정값이 아닌 target까지 거리와 루프 주기를 고려해 동적으로 변하도록 수정하기 (env.data.time 또는 model.opt.timestep 기반으로 잡기)

    poll_interval = 0.1
    last_poll_time = 0.0
    sim_steps_per_frame = 1
    steps_per_sim = 8

    # 입력 정보 관리
    polled_target = None
    target_R = None
    alpha = np.zeros(2)

    try:
        while not glfw.window_should_close(env.viewer.window):  # 렌더링 루프
            glfw.poll_events()

            # 1. rendering 주기 / 2. poll 주기 / 3. 시뮬레이션 주기 / 4. step() 자체 주기(예. model.opt.timestep) 분리
            # 주기 관리: 반복문 / 조건문
            # poll_interval = 0.1
            # last_poll_time = 0.0
            now = time.time()
            # polled_target = None
            # target_R = None
            # alpha = np.zeros(2)

            polled_target = env.viewer.hand_pose_panel.poll_target(env.viewer.window)  # 매 프레임마다 입력 처리
            if polled_target is not None:
                trajectory_start = current_target.copy()
                trajectory_goal = polled_target[:3].copy()

                # rpy 입력 반영
                target_rpy = polled_target[3:]
                target_rot = rpy2rotation_matrix(target_rpy[0], target_rpy[1], target_rpy[2])
                target_R = get_body_T(env.data, env.ee_body_id)[:3, :3] @ target_rot

                # hand grasp 입력 반영
                alpha[:] = [polled_target[6], polled_target[7]]

                trajectory_start_time = env.data.time
                last_poll_time = now

            step_start_time = time.time()

            for _ in range(sim_steps_per_frame):  # 시뮬레이션 루프
                if trajectory_start_time is not None:
                    t = env.data.time - trajectory_start_time  # t: time-scaling이 입력으로 받는 궤적 시점

                    if t >= trajectory_duration:
                        current_target = trajectory_goal.copy()
                        trajectory_start_time = None
                    else:
                        current_target = interpolate_position(  # cubic time-scaling
                            trajectory_start,
                            trajectory_goal,
                            trajectory_duration,
                            t,
                        )
                        # current_target = interpolate_position_simple(  # 선형 보간
                        #     trajectory_start,
                        #     trajectory_goal,
                        #     trajectory_duration,
                        #     t,
                        # )

                new_target = initial_pose.copy()
                new_target[:3, 3] = current_target
                if target_R is not None:
                    new_target[:3, :3] = target_R

                q_des, joint_ids = solve_ik(env.model, env.data, env.ee_body_id, new_target, True, ik_joint_names)
                actuator_ids = actuator_ids_from_joints(env.model, joint_ids)

                for i in range(1):  # 수정 예정
                    # IK solver result로 qpos(kinematic simulation용) 또는 ctrl(dynamic simulation용)을 갱신
                    # 옵션 1: dynamic update (dynamic simulation)
                    for actuator_id in actuator_ids:
                        jid = env.model.actuator_trnid[actuator_id, 0]
                        qadr = env.model.jnt_qposadr[jid]
                        env.data.ctrl[actuator_id] = q_des[qadr]  # ctrl 유형은 관절 종류에 따라 다름(예. force/qpos 등)
                    # 옵션 2: kinematic update (kinematic simulation)
                    # for joint_id in joint_ids:
                    #     qadr = model.jnt_qposadr[joint_id]
                    #     data.qpos[qadr] = q_des[qadr]

                    # 손가락 위치 보간
                    interpolate_finger(env.model, env.data, alpha)  # data.qpos를 갱신중, ctrl을 갱신하도록 수정?
                    # 옵션 1
                    # mujoco.mj_forward(env.model, env.data)  # qfrc_bias 계산을 위한 forward
                    # env.data.qfrc_applied[:] = 0.0
                    # env.data.qfrc_applied[:] = env.data.qfrc_bias #  중력항 상쇠를 위한 보정항, 현재 joint-space와 달리 task-space에서는 damping이 이루어지지 않고 있음
                    env.step(steps_per_sim)
                    # mujoco.mj_step(env.model, env.data)
                    # 옵션 2
                    # data.qvel[:] = 0.0
                    # mujoco.mj_forward(model, data)

            env.viewer.render()

    finally:
        env.viewer.terminate_viewer()


if __name__ == "__main__":
    main()
