from pyexpat import model
import time

import glfw
import mujoco
import numpy as np

from sim.model.math3d.rotation import rpy2rotation_matrix
from sim.model.motion.trajectory import (
    interpolate_pose,
    interpolate_position,
    interpolate_position_quintic,
    interpolate_position_simple,
)
from sim_with_mujoco.environment.env import Environment
from sim_with_mujoco.utils.dynamics import pd_joint_space
from sim_with_mujoco.utils.ik import solve_ik
from sim_with_mujoco.utils.kinematics import interpolate_finger
from sim_with_mujoco.utils.math3d import get_body_T
from sim_with_mujoco.utils.mj import actuator_ids_from_joints, dof_ids_from_joints, joint_ids_from_names

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
    mujoco.mj_copyData(temp_data, env.model, env.data)
    mujoco.mj_forward(env.model, temp_data)  # qfrc_bias 계산을 위한 forward

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
    steps_per_sim = 8
    poll_interval = 1.0 / 30.0
    render_interval = 1.0 / 60.0
    last_poll_time = time.time()
    last_render_time = time.time()

    # 궤적 형성 관련 변수
    trajectory_start = initial_pose.copy()
    trajectory_goal = initial_pose.copy()
    T_des = initial_pose.copy()
    trajectory_start_time = None
    trajectory_last_time = 0
    # trajectory_duration = (
    #     env.model.opt.timestep * sim_steps_per_frame
    # )  # 고정값이 아닌 target까지 거리와 루프 주기를 고려해 동적으로 변하도록 수정하기 (env.data.time 또는 model.opt.timestep 기반으로 잡기)
    trajectory_duration = 0.05

    # 입력 정보 관리
    polled_target = None
    alpha = np.zeros(2)

    try:
        while not glfw.window_should_close(env.viewer.window):  # 렌더링 루프 (프레임 단위)
            for _ in range(sim_steps_per_frame):  # 시뮬레이션 루프
                glfw.poll_events()

                now = time.time()

                # polled_target = None
                if now - last_poll_time >= poll_interval:  # poll 주기 설정
                    last_poll_time = now
                    polled_target, polled_camera = env.viewer.poll_target()  # 매 프레임마다 입력 처리

                if polled_target is not None:
                    # trajectory_start = T_des.copy()
                    trajectory_goal = T_des.copy()
                    # xyz 입력 반영
                    trajectory_goal[:3, 3] = polled_target[:3].copy()

                    # rpy 입력 반영
                    target_rpy = polled_target[3:]
                    target_rot = rpy2rotation_matrix(target_rpy[0], target_rpy[1], target_rpy[2])
                    trajectory_goal[:3, :3] = initial_pose[:3, :3] @ target_rot

                    # hand grasp 입력 반영
                    alpha[:] = [polled_target[6], polled_target[7]]

                    # 궤적이 없을 때만 궤적 시발점/출발점 설정, poll 주기마다 start time을 갱신하면 매번 새 궤적이 궤적이 생성되며 t에 변화가 없어짐
                    if trajectory_start_time is None:
                        trajectory_start = T_des.copy()
                        trajectory_start_time = env.data.time

                # for _ in range(sim_steps_per_frame):  # 시뮬레이션 루프
                twist_des = np.zeros(6)
                twistdot_des = np.zeros(6)

                if trajectory_start_time is not None:
                    t = env.data.time - trajectory_start_time  # t: time-scaling이 입력으로 받는 궤적 시점

                    if t >= trajectory_duration:  # 궤적 주기가 끝난 후에는 목표 pose를 고정
                        T_des = trajectory_goal.copy()
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
                        # current_target, p_dot_des, p_dotdot_des = interpolate_position_quintic(
                        #     trajectory_start,
                        #     trajectory_goal,
                        #     trajectory_duration,
                        #     t,
                        # )
                        T_des, twist_des, twistdot_des = interpolate_pose(
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

                new_target = T_des.copy()

                q_des, joint_ids = solve_ik(
                    env.model,
                    env.data,
                    [(env.ee_body_id, new_target), (env.left_hand_id, env.left_initial_T)],
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
                        env.data.ctrl[actuator_id] = q_des[qadr]  # position actuator ctrl = qpos

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
                    interpolate_finger(env.model, env.data, [0, 0], True)  # 왼손 자세 유지
                    # 옵션 1
                    # mujoco.mj_copyData(temp_data, env.model, env.data)
                    # mujoco.mj_forward(env.model, temp_data)  # qfrc_bias 계산을 위한 forward

                    # PD controller
                    q_dot_des, q_dotdot_des = env.task_to_joint_space(twist_des, twistdot_des, joint_ids)
                    # tau_des = pd_controller(env.model, env.data, q_des, q_dot_des, q_dotdot_des, joint_ids)
                    # env.data.qfrc_applied[:] = 0.0
                    # env.data.qfrc_applied[dof_ids] = tau_des

                    all_joint_ids = joint_ids_from_names(env.model, ik_joint_names)
                    all_dof_ids = dof_ids_from_joints(env.model, all_joint_ids)
                    env.data.qfrc_applied[all_dof_ids] = 0.0
                    env.data.qfrc_applied[all_dof_ids] = temp_data.qfrc_bias[all_dof_ids]
                    env.step(steps_per_sim)

                    # 옵션 2
                    # env.data.qvel[:] = 0.0
                    # # mujoco.mj_forward(env.model, env.data)
                    # env.step(steps_per_sim)

                    if now - last_render_time >= render_interval:
                        env.viewer.render()
                        last_render_time = now
                        # glfw.poll_events() # render wrapper에서 이벤트 폴링 및 callback 함수 실행하게 함

    finally:
        env.viewer.terminate_viewer()


if __name__ == "__main__":
    main()
