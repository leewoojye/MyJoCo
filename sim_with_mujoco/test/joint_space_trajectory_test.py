from pyexpat import model
import time

import glfw
import mujoco
import numpy as np

from sim.model.math3d.rotation import rpy2rotation_matrix
from sim.model.motion.trajectory import (
    interpolate_joint,
    interpolate_pose,
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
    env = Environment(XML_PATH, "arm_r_link7")
    env.initial_qpos(initial_qpos)

    # 임시 MjData (forward, step 중복 계산 방지용)
    temp_data = mujoco.MjData(env.model)
    mujoco.mj_copyData(temp_data, env.model, env.data)
    mujoco.mj_forward(env.model, temp_data)

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
    trajectory_goal_q = initial_q.copy()

    # 주기 상수
    sim_steps_per_frame = 8
    steps_per_sim = 8
    poll_interval = 1.0 / 20.0
    render_interval = 1.0 / 60.0
    last_poll_time = time.time()
    last_render_time = time.time()

    # 궤적 형성 관련 변수
    trajectory_start = initial_pose.copy()
    trajectory_goal_T = initial_pose.copy()
    T_des = initial_pose.copy()

    joint_ids = joint_ids_from_names(env.model, ik_joint_names)
    q_des = initial_q.copy()
    q_dot_des = np.zeros(len(joint_ids))
    q_dotdot_des = np.zeros(len(joint_ids))
    q_traj_start = q_des.copy()
    q_traj_goal = q_des.copy()
    trajectory_start_time = None
    trajectory_duration = 0.08
    # trajectory_duration = env.model.opt.timestep * 16

    # 입력 정보 관리
    polled_target = None
    alpha = np.zeros(2)

    try:
        while not glfw.window_should_close(env.viewer.window):
            for _ in range(sim_steps_per_frame):  # 시뮬레이션 루프
                glfw.poll_events()

                now = time.time()

                polled_target = None
                if now - last_poll_time >= poll_interval:
                    last_poll_time = now
                    polled_target, polled_camera = env.viewer.poll_target()  # 매 프레임마다 입력 처리

                if polled_target is not None:
                    trajectory_goal_T = T_des.copy()
                    # xyz 입력 반영
                    trajectory_goal_T[:3, 3] = polled_target[:3].copy()

                    # rpy 입력 반영
                    target_rpy = polled_target[3:]
                    target_rot = rpy2rotation_matrix(target_rpy[0], target_rpy[1], target_rpy[2])
                    trajectory_goal_T[:3, :3] = initial_pose[:3, :3] @ target_rot

                    # hand grasp 입력 반영
                    alpha[:] = [polled_target[6], polled_target[7]]

                    trajectory_goal_q, joint_ids = solve_ik(
                        env.model,
                        env.data,
                        [(env.ee_body_id, trajectory_goal_T), (env.left_hand_id, env.left_initial_T)],
                        is_pose=[True, False],
                        joint_names=ik_joint_names,
                        check_collision=False,
                    )

                    if trajectory_start_time is None:
                        # q_traj_start = q_des.copy()
                        q_traj_start = env.data.qpos.copy()
                        q_traj_goal = trajectory_goal_q.copy()
                        trajectory_start_time = env.data.time

                if trajectory_start_time is not None:
                    t = env.data.time - trajectory_start_time

                    if t >= trajectory_duration:
                        T_des = trajectory_goal_T.copy()
                        q_des = q_traj_goal.copy()
                        q_dot_des = np.zeros(len(joint_ids))
                        q_dotdot_des = np.zeros(len(joint_ids))
                        trajectory_start_time = None
                    else:
                        q_des, q_dot_des, q_dotdot_des = interpolate_joint(
                            q_traj_start,
                            q_traj_goal,
                            trajectory_duration,
                            t,
                        )

                # new_target = T_des.copy()

                actuator_ids = actuator_ids_from_joints(env.model, joint_ids)

                for _ in range(1):  # 수정 예정
                    # 옵션 1: dynamic update (dynamic simulation)
                    dof_ids = dof_ids_from_joints(env.model, joint_ids)

                    for actuator_id in actuator_ids:
                        jid = env.model.actuator_trnid[actuator_id, 0]
                        qadr = env.model.jnt_qposadr[jid]
                        env.data.ctrl[actuator_id] = q_des[qadr]

                    interpolate_finger(env.model, env.data, alpha)
                    interpolate_finger(env.model, env.data, [0, 0], True)

                    # PD controller
                    # q_dot_des, q_dotdot_des = env.task_to_joint_space(twist_des, twistdot_des, joint_ids)
                    # tau_des = pd_controller(env.model, env.data, q_des, q_dot_des, q_dotdot_des, joint_ids)

                    all_joint_ids = joint_ids_from_names(env.model, ik_joint_names)
                    all_dof_ids = dof_ids_from_joints(env.model, all_joint_ids)
                    env.data.qfrc_applied[all_dof_ids] = 0.0
                    env.data.qfrc_applied[all_dof_ids] = temp_data.qfrc_bias[all_dof_ids]
                    env.step(steps_per_sim)

                    if now - last_render_time >= render_interval:
                        env.viewer.render()
                        last_render_time = now

    finally:
        env.viewer.terminate_viewer()


if __name__ == "__main__":
    main()
