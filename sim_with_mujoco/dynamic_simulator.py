from pyexpat import model
import time

import glfw
import mujoco
import numpy as np

from sim.model.math3d.rotation import rpy2rotation_matrix
from sim.model.motion.trajectory import interpolate_position, interpolate_position_quintic, interpolate_position_simple
from sim_with_mujoco.environment.env import Environment
from sim_with_mujoco.utils.dynamics import pd_controller
from sim_with_mujoco.utils.ik import solve_ik
from sim_with_mujoco.utils.kinematics import interpolate_finger
from sim_with_mujoco.utils.math3d import get_body_T
from sim_with_mujoco.utils.mj import actuator_ids_from_joints, dof_ids_from_joints

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

    # 임시 MjData (forward, step 중복 계산 방지용)
    temp_data = mujoco.MjData(env.model)

    ik_joint_names = [
        "lift_joint",
        "arm_l_joint1",
        "arm_l_joint2",
        "arm_l_joint3",
        "arm_l_joint4",
        "arm_l_joint5",
        "arm_l_joint6",
        "arm_l_joint7",
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

    # 주기 상수
    sim_steps_per_frame = 8
    steps_per_sim = 16
    poll_interval = 1.0 / 60.0
    last_poll_time = time.time()

    # 궤적 형성 관련 변수
    trajectory_start = initial_target_pos.copy()
    trajectory_goal = initial_target_pos.copy()
    current_target = initial_target_pos.copy()
    trajectory_start_time = None
    trajectory_last_time = 0
    # trajectory_duration = (
    #     env.model.opt.timestep * sim_steps_per_frame
    # )  # 고정값이 아닌 target까지 거리와 루프 주기를 고려해 동적으로 변하도록 수정하기 (env.data.time 또는 model.opt.timestep 기반으로 잡기)
    trajectory_duration = 0.5
    # trajectory_duration = poll_interval

    # 입력 정보 관리
    polled_target = None
    target_R = None
    alpha = np.zeros(2)

    try:
        while not glfw.window_should_close(env.viewer.window):  # 렌더링 루프 (프레임 단위)
            glfw.poll_events()

            # 1. rendering 주기 / 2. poll 주기 / 3. 시뮬레이션 주기 / 4. step() 자체 주기(예. model.opt.timestep) 분리
            # 주기 관리: 반복문 / 조건문
            now = time.time()  # 절대 시간 (시뮬레이션 시간은 env.data.time)

            # if now - last_poll_time >= poll_interval: # poll 주기 설정
            last_poll_time = now
            polled_target, polled_camera = env.viewer.poll_target()  # 매 프레임마다 입력 처리
            # env.viewer.render()  # 패널 입력 실시간 반영을 위한 렌더링
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

            for _ in range(sim_steps_per_frame):  # 시뮬레이션 루프
                p_dot_des = np.zeros(3)
                p_dotdot_des = np.zeros(3)

                if trajectory_start_time is not None:
                    t = env.data.time - trajectory_start_time  # t: time-scaling이 입력으로 받는 궤적 시점

                    if t >= trajectory_duration:
                        current_target = trajectory_goal.copy()
                        trajectory_start_time = None
                    else:
                        # current_target, qacc_des = interpolate_position(  # cubic time-scaling
                        #     trajectory_start, trajectory_goal, trajectory_duration, t, True
                        # )
                        # current_target = interpolate_position_simple(  # 선형 보간
                        #     trajectory_start,
                        #     trajectory_goal,
                        #     trajectory_duration,
                        #     t,
                        # )
                        current_target, p_dot_des, p_dotdot_des = interpolate_position_quintic(
                            trajectory_start,
                            trajectory_goal,
                            trajectory_duration,
                            t,
                        )
                        # if trajectory_last_time is not None:
                        #     prev_p, prev_p_dot, prev_p_dotdot = interpolate_position_quintic(
                        #         trajectory_start,
                        #         trajectory_goal,
                        #         trajectory_duration,
                        #         trajectory_last_time,
                        #     )
                        #     trajectory_last_time = t
                        # else:
                        #     trajectory_last_time = 0

                # qdot_des = pinv(J) @ p_dot_des
                # qddot_des = pinv(J) @ (p_ddot_des - Jdot_qdot)

                new_target = initial_pose.copy()
                new_target[:3, 3] = current_target
                if target_R is not None:
                    new_target[:3, :3] = target_R

                left_hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "arm_l_link7")  # 추후 수정
                left_target = get_body_T(env.data, left_hand_id)
                q_des, joint_ids = solve_ik(
                    env.model,
                    env.data,
                    [(env.ee_body_id, new_target), (left_hand_id, left_target)],
                    is_pose=[True, False],
                    joint_names=ik_joint_names,
                    check_collision=False,
                )
                actuator_ids = actuator_ids_from_joints(env.model, joint_ids)

                for i in range(1):  # 수정 예정
                    # IK solver result로 qpos(kinematic simulation용) 또는 ctrl(dynamic simulation용)을 갱신
                    # 옵션 1: dynamic update (dynamic simulation)
                    dof_ids = dof_ids_from_joints(env.model, joint_ids)
                    for actuator_id in actuator_ids:
                        jid = env.model.actuator_trnid[actuator_id, 0]
                        qadr = env.model.jnt_qposadr[jid]
                        env.data.ctrl[actuator_id] = q_des[qadr]  # ctrl 유형은 관절 종류에 따라 다름(예. force/qpos 등)
                    # 옵션 2: kinematic update (kinematic simulation)
                    # for joint_id in joint_ids:
                    #     qadr = env.model.jnt_qposadr[joint_id]
                    #     env.data.qpos[qadr] = q_des[qadr]
                    # for actuator_id in actuator_ids:
                    #     jid = env.model.actuator_trnid[actuator_id, 0]
                    #     qadr = env.model.jnt_qposadr[jid]
                    #     env.data.ctrl[actuator_id] = q_des[qadr]

                    # 손가락 위치 보간
                    interpolate_finger(env.model, env.data, alpha)  # data.qpos를 갱신중, ctrl을 갱신하도록 수정?
                    # 옵션 1
                    mujoco.mj_copyData(temp_data, env.model, env.data)
                    mujoco.mj_forward(env.model, temp_data)  # qfrc_bias 계산을 위한 forward

                    # PD controller
                    q_dot_des, q_dotdot_des = env.task_to_joint_space(p_dot_des, p_dotdot_des, joint_ids)
                    tau_des = pd_controller(env.model, env.data, q_des, q_dot_des, q_dotdot_des, joint_ids)
                    env.data.qfrc_applied[:] = 0.0
                    env.data.qfrc_applied[dof_ids] = tau_des

                    # env.data.qfrc_applied[dof_ids] = 0.0
                    # env.data.qfrc_applied[dof_ids] = temp_data.qfrc_bias[dof_ids]
                    env.step(steps_per_sim)

                    # 옵션 2
                    # env.data.qvel[:] = 0.0
                    # # mujoco.mj_forward(env.model, env.data)
                    # env.step(steps_per_sim)

            env.viewer.render()

    finally:
        env.viewer.terminate_viewer()


if __name__ == "__main__":
    main()
