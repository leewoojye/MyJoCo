from pyexpat import model
import time

import glfw
import mujoco
import numpy as np

from sim.model.math3d.rotation import rpy2rotation_matrix
from sim.model.motion.trajectory import (
    interpolate_pose,
    interpolate_pose_ros,
    interpolate_position_cubic,
    interpolate_position_quintic,
    interpolate_position_simple,
)
from sim_with_mujoco.environment.env import Environment
from sim_with_mujoco.utils.dynamics import ct_joint_space
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
    ref_data = mujoco.MjData(env.model)

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
    initial_q = env.initial_q
    q_des = initial_q.copy()

    # 주기 상수
    sim_steps_per_frame = 1
    steps_per_sim = 8
    poll_interval = 1.0 / 20.0
    render_interval = 1.0 / 60.0
    last_poll_time = 0
    last_render_time = 0

    # 궤적 형성 관련 변수
    trajectory_start = initial_pose.copy()  # 단일 궤적의 시작 지점
    trajectory_goal = initial_pose.copy()  # 단일 궤적의 목표 지점
    T_des = initial_pose.copy()  # 한 시점의 궤적상 위치
    twist_des = np.zeros(6)
    twistdot_des = np.zeros(6)
    trajectory_start_twist = np.zeros(6)
    trajectory_start_twistdot = np.zeros(6)
    # trajectory_duration = (
    #     env.model.opt.timestep * sim_steps_per_frame
    # )  # 고정값이 아닌 target까지 거리와 루프 주기를 고려해 동적으로 변하도록 수정하기 (env.data.time 또는 model.opt.timestep 기반으로 잡기)
    # trajectory_duration = env.model.opt.timestep * 16
    trajectory_start_time = None
    trajectory_duration = 0.06
    trajectory_step = env.model.opt.timestep * steps_per_sim  # 궤적이 시뮬레이션 루프당 진척되는 정도
    traj_plan_interval = poll_interval  # 궤적형성 주기 지정
    last_traj_plan = 0

    # 입력 정보 관리
    polled_target = None  # polled: 이번 폴링에 새로 들어온 입력
    pending_target = None  # pending: 아직 궤적에 반영안된 최신 입력
    alpha = [0, 0]

    try:
        while not glfw.window_should_close(env.viewer.window):  # 렌더링 루프 (프레임 단위)
            for _ in range(sim_steps_per_frame):  # 시뮬레이션 루프, 추후 수정
                glfw.poll_events()

                now = time.time()

                polled_target = None
                if now - last_poll_time >= poll_interval:  # poll 주기 설정
                    last_poll_time = now
                    polled_target, polled_camera = env.viewer.poll_target()  # 매 프레임마다 입력 처리

                if polled_target is not None:
                    pending_target = polled_target.copy()

                if pending_target is not None and now - last_traj_plan >= traj_plan_interval:
                    last_traj_plan = now
                    trajectory_start = T_des.copy()
                    trajectory_start_twist = twist_des.copy()
                    trajectory_start_twistdot = twistdot_des.copy()
                    trajectory_goal = trajectory_start.copy()

                    # xyz 입력 반영
                    trajectory_goal[:3, 3] = pending_target[:3].copy()

                    # rpy 입력 반영
                    target_rpy = pending_target[3:]
                    target_rot = rpy2rotation_matrix(target_rpy[0], target_rpy[1], target_rpy[2])
                    trajectory_goal[:3, :3] = initial_pose[:3, :3] @ target_rot

                    # hand grasp 입력 반영
                    alpha[:] = [pending_target[6], pending_target[7]]
                    trajectory_start_time = env.data.time
                    pending_target = None

                if trajectory_start_time is not None:
                    pos_err = np.linalg.norm(trajectory_goal[:3, 3] - T_des[:3, 3])
                    rot_err = np.linalg.norm(trajectory_goal[:3, :3] - T_des[:3, :3])
                    if pos_err <= 1e-5 and rot_err <= 1e-5:
                        T_des = trajectory_goal.copy()
                        twist_des = np.zeros(6)
                        twistdot_des = np.zeros(6)
                        trajectory_start_time = None
                    else:
                        # T_des, twist_des, twistdot_des = interpolate_pose(
                        #     trajectory_start,
                        #     trajectory_goal,
                        #     trajectory_duration,
                        #     min(t, trajectory_duration),
                        # )
                        T_des, twist_des, twistdot_des = interpolate_pose_ros(
                            trajectory_start,
                            trajectory_goal,
                            trajectory_start_twist,
                            trajectory_start_twistdot,
                            trajectory_duration,
                            min(trajectory_step, trajectory_duration),
                        )
                        trajectory_start = T_des.copy()
                        trajectory_start_twist = twist_des.copy()
                        trajectory_start_twistdot = twistdot_des.copy()
                else:
                    T_des = trajectory_goal.copy()

                new_target = T_des.copy()

                mujoco.mj_copyData(ref_data, env.model, env.data)
                ref_data.qpos[:] = q_des
                ref_data.qvel[:] = 0.0
                mujoco.mj_forward(env.model, ref_data)

                q_des, joint_ids = solve_ik(
                    env.model,
                    ref_data,
                    [(env.ee_body_id, new_target), (env.left_hand_id, env.left_initial_T)],
                    is_pose=[True, False],
                    joint_names=ik_joint_names,
                    check_collision=False,
                )
                actuator_ids = actuator_ids_from_joints(env.model, joint_ids)

                for _ in range(1):  # 수정 예정
                    # IK solver result로 qpos(kinematic simulation용) 또는 ctrl(dynamic simulation용)을 갱신
                    # 옵션 1: dynamic update (dynamic simulation)
                    for actuator_id in actuator_ids:
                        jid = env.model.actuator_trnid[actuator_id, 0]
                        qadr = env.model.jnt_qposadr[jid]
                        env.data.ctrl[actuator_id] = q_des[qadr]

                    # 옵션 2: kinematic update (kinematic simulation)
                    # for joint_id in joint_ids:
                    #     qadr = env.model.jnt_qposadr[joint_id]
                    #     env.data.qpos[qadr] = q_des[qadr]
                    # for actuator_id in actuator_ids:
                    #     jid = env.model.actuator_trnid[actuator_id, 0]
                    #     qadr = env.model.jnt_qposadr[jid]
                    #     env.data.ctrl[actuator_id] = q_des[qadr]

                    interpolate_finger(env.model, env.data, alpha)  # ctrl을 갱신하도록 수정
                    interpolate_finger(env.model, env.data, [0, 0], True)  # 왼손 자세 유지

                    # 옵션 1
                    # mujoco.mj_copyData(temp_data, env.model, env.data)
                    # mujoco.mj_forward(env.model, temp_data)  # qfrc_bias 계산을 위한 forward

                    finger_joint_names = [f"finger_r_joint{i}" for i in range(1, 21)] + [
                        f"finger_l_joint{i}" for i in range(1, 21)
                    ]
                    gravity_joint_names = ik_joint_names + finger_joint_names
                    all_joint_ids = joint_ids_from_names(env.model, gravity_joint_names)
                    all_dof_ids = dof_ids_from_joints(env.model, all_joint_ids)

                    mujoco.mj_forward(env.model, env.data)
                    env.data.qfrc_applied[all_dof_ids] = 0.0
                    env.data.qfrc_applied[all_dof_ids] = env.data.qfrc_bias[all_dof_ids]
                    env.step(steps_per_sim)

                    # 옵션 2
                    # env.data.qvel[:] = 0.0
                    # # mujoco.mj_forward(env.model, env.data)
                    # env.step(steps_per_sim)

            now_ren = time.time()

            # if now_ren - last_render_time >= render_interval:
            #     env.viewer.render()
            #     last_render_time = now_ren
            env.viewer.render()

    finally:
        env.viewer.terminate_viewer()


if __name__ == "__main__":
    main()
