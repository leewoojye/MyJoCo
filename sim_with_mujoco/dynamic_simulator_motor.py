import time

import glfw
import mujoco
import numpy as np

from sim.model.math3d.rotation import rpy2rotation_matrix
from sim_with_mujoco.environment.env import Environment
from sim_with_mujoco.utils.collision import is_can_finger_contact
from sim_with_mujoco.utils.dynamics import computed_torque_control, finger_pd_control
from sim_with_mujoco.utils.ik_qp import solve_differential_ik
from sim_with_mujoco.utils.kinematics import interpolate_finger, interpolate_finger_motor
from sim_with_mujoco.utils.mj import actuator_ids_from_joints, dof_ids_from_joints, joint_ids_from_names
# from sim_with_mujoco.temp.metrics import MetricRecorder

XML_PATH = "/Users/woojyelee/workspace/my_robotics/assets/robots/robotis_ffw/scene_ffw_sh5_motor_arms_fingers.xml"


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
    # temp_data = mujoco.MjData(env.model)
    # mujoco.mj_copyData(temp_data, env.model, env.data)
    # mujoco.mj_forward(env.model, temp_data)

    # qpos를 q_des로 맞춘 mjData
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
    initial_pose = env.initial_pose
    initial_q = env.initial_q

    # 주기 상수
    sim_steps_per_frame = 1  # 궤적 형성
    steps_per_sim = 8
    poll_interval = 1.0 / 60.0
    last_poll_time = 0

    T_des = initial_pose.copy()

    joint_ids = joint_ids_from_names(env.model, ik_joint_names)
    q_des = initial_q.copy()
    q_dot_des = np.zeros(len(joint_ids))
    q_dotdot_des = np.zeros(len(joint_ids))
    # recorder = MetricRecorder("ik_test2", joint_ids)

    # 입력 정보 관리
    polled_target = None  # polled: 이번 폴링에 새로 들어온 입력
    alpha_target = np.zeros(2)
    alpha_cmd = alpha_target.copy()

    try:
        while not glfw.window_should_close(env.viewer.window):
            for _ in range(sim_steps_per_frame):  # 시뮬레이션 루프
                glfw.poll_events()
                # is_dragging = glfw.get_mouse_button(env.viewer.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS

                now = time.time()

                polled_target = None
                if now - last_poll_time >= poll_interval:
                    last_poll_time = now
                    polled_target, polled_camera = env.viewer.poll_target()

                if polled_target is not None:
                    T_des = initial_pose.copy()
                    T_des[:3, 3] = polled_target[:3].copy()

                    target_rpy = polled_target[3:]
                    target_rot = rpy2rotation_matrix(target_rpy[0], target_rpy[1], target_rpy[2])
                    T_des[:3, :3] = initial_pose[:3, :3] @ target_rot

                    alpha_target[:] = [polled_target[6], polled_target[7]]

                for _ in range(steps_per_sim):  # 충돌 감지, mj_step()을 한 iteration으로 묶음
                    mujoco.mj_copyData(ref_data, env.model, env.data)
                    ref_data.qpos[:] = q_des
                    ref_data.qvel[:] = 0.0
                    mujoco.mj_forward(env.model, ref_data)

                    q_des, q_dot_des, _ = solve_differential_ik(
                        env.model,
                        ref_data,
                        [
                            (env.ee_body_id, T_des, True, 1.0),
                            (env.left_hand_id, env.left_initial_T, False, 4.0),
                        ],  # 에러 벡터 비율에 비례하여 왼손 position IK에 가중치
                        joint_ids,
                        env.model.opt.timestep,
                    )
                    q_dotdot_des[:] = 0.0

                    finger_contact = is_can_finger_contact(  # 추후 수정
                        env.model,
                        env.data,
                    )

                    # finger 목표 alpha를 dt 후 명령 alpha로 변환, differential IK에 맞춰 dt 동안 alpha 변화량을 의도한 것이지만 명시된 forcerange 범위가 작아 효과가 미미함
                    dt = env.model.opt.timestep
                    k_alpha = 30.0
                    alpha_dot_des = k_alpha * (alpha_target - alpha_cmd)
                    alpha_cmd += alpha_dot_des * dt
                    alpha_cmd = np.clip(alpha_cmd, 0.0, 1.0)

                    if finger_contact:  # 오른손
                        # finger_pd_control(env, alpha_cmd)
                        finger_pd_control(env, alpha_cmd, kp=20.0, kd=2.0, tau_max=4.0)
                    else:
                        finger_pd_control(env, alpha_cmd, kp=40.0, kd=3.0, tau_max=4.0)

                    interpolate_finger_motor(env, [0, 0], True)  # 왼손

                    # finger 제외한 active joints의 computed torque 계산
                    tau_des = computed_torque_control(
                        env.model, env.data, q_des, q_dot_des, q_dotdot_des, joint_ids, 35, 12
                    )

                    for i, joint_id in enumerate(joint_ids):
                        actuator_id = None

                        for a in range(env.model.nu):
                            if env.model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                                continue
                            if env.model.actuator_trnid[a, 0] == joint_id:
                                actuator_id = a
                                break

                        if actuator_id is None:
                            continue

                        gear = env.model.actuator_gear[actuator_id, 0]
                        ctrl = tau_des[i] / gear
                        if env.model.actuator_ctrllimited[actuator_id]:
                            lo, hi = env.model.actuator_ctrlrange[actuator_id]
                            ctrl = np.clip(ctrl, lo, hi)
                        env.data.ctrl[actuator_id] = ctrl

                    env.step(1)
                    # recorder.record(env, T_des, q_des, tau_des)

            env.viewer.render()

    finally:
        env.viewer.terminate_viewer()


if __name__ == "__main__":
    main()
