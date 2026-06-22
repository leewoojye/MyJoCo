import mujoco
import numpy as np

from sim.model.kinematics.ik import calculate_twist_error
from sim_with_mujoco.environment.env import Environment
from sim_with_mujoco.utils.ik import damped_pseudoinverse
from sim_with_mujoco.utils.kinematics import get_body_jacobian
from sim_with_mujoco.utils.math3d import get_body_T, get_body_twist
from sim_with_mujoco.utils.mj import dof_ids_from_joints, joint_ids_from_actuators


# <position> actuator:
# ctrl = 목표 관절각, MuJoCo가 force 계산
# <motor> actuator:
# ctrl = 직접 힘/토크 명령
# <velocity> actuator:
# ctrl = 목표 속도, MuJoCo가 속도 오차 기반 force 계산


def solve_inverse_dynamics(model, data, qacc):
    data.qacc = qacc.copy()
    mujoco.mj_inverse(model, data)
    qfrc_inverse = data.qfrc_inverse  # inverse dynamics 결과

    return qfrc_inverse.copy()


def computed_torque_control(model, data, q_des, q_dot_des, q_dotdot_des, joint_ids, kp=300, kd=50):
    inv_data = mujoco.MjData(model)  # inverse dynamic 계산용
    mujoco.mj_copyData(inv_data, model, data)

    dof_ids = dof_ids_from_joints(model, joint_ids)
    qpos_ids = model.jnt_qposadr[joint_ids]
    # kp = model.actuator_gainprm[dof_ids, 0]
    # kp = 60
    # kd = 25

    #  q_dot_des = q_dotdot_des = 0: 컨트롤러가 목표 궤적이 정지해있다고 인식 (computed torque setpoint control)
    # q_dot_des = np.zeros(len(joint_ids))
    # q_dotdot_des = np.zeros(len(joint_ids))

    qacc_des = (
        q_dotdot_des  # task_to_joint_space()로 인해 active actuator에 대한 원소로만 구성
        + kp * (q_des[qpos_ids] - inv_data.qpos[qpos_ids])
        + kd * (q_dot_des - inv_data.qvel[dof_ids])
    )

    inv_data.qacc[:] = 0.0
    inv_data.qacc[dof_ids] = qacc_des
    mujoco.mj_inverse(model, inv_data)  # mj_inverse는 내부적으로 중력항을 고려해 토크를 반환
    # contact constraint는 mj_step이 처리하므로 actuator torque에서 다시 보상하지 않음
    tau = inv_data.qfrc_inverse[dof_ids]

    return tau


# task-space PD 제어 입력을 joint torque로 바꿈
def pd_task_space(env: Environment, x_des, twist_des, twist_dot_des, joint_ids, kp=16, kd=4):
    dof_ids = np.array([env.model.jnt_dofadr[jid] for jid in joint_ids], dtype=int)
    # kp = 100
    # kd = 30

    twist_current = get_body_twist(env.model, env.data, env.ee_body_id, joint_ids)
    twist_error = env.get_twist_error(x_des)
    cmd = twist_dot_des + kd * (twist_des - twist_current) + kp * twist_error

    J = get_body_jacobian(env.model, env.data, env.ee_body_id)[:, dof_ids]
    tau = J.T @ cmd

    return tau


def pid_controller():
    return


# finger 대상 PD 제어기 (position actuator가 썼던 방식)
def finger_pd_control(env, alpha, is_left=False, kp=20.0, kd=2.0, tau_max=2.0):
    model = env.model
    data = env.data
    name = "finger_l" if is_left else "finger_r"
    q_open = 0
    thumb_q_open = [0.3, 1.57, -0.35, -0.25] if is_left else [0.3, -1.57, 0.35, 0.25]
    thumb_q_closed = [0.4, 1.25, -0.8, -0.7] if is_left else [0.4, -1.25, 0.8, 0.7]

    for index, i in enumerate(range(1, 5)):
        joint_name = f"{name}_joint{i}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        qpos_id = model.jnt_qposadr[joint_id]
        dof_id = model.jnt_dofadr[joint_id]

        q_des = (1 - alpha[0]) * thumb_q_open[index] + alpha[0] * thumb_q_closed[index]
        q = data.qpos[qpos_id]
        qdot = data.qvel[dof_id]

        tau = kp * (q_des - q) - kd * qdot

        tau = np.clip(tau, -tau_max, tau_max)  # actuator limits에 맞춤
        if model.actuator_ctrllimited[actuator_id]:
            lo, hi = model.actuator_ctrlrange[actuator_id]
            tau = np.clip(tau, lo, hi)

        data.ctrl[actuator_id] = tau

    for i in range(5, 21):
        joint_name = f"{name}_joint{i}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        qpos_id = model.jnt_qposadr[joint_id]
        dof_id = model.jnt_dofadr[joint_id]

        q_closed = model.jnt_range[joint_id, 1]
        q_des = (1 - alpha[1]) * q_open + alpha[1] * q_closed
        q = data.qpos[qpos_id]
        qdot = data.qvel[dof_id]

        tau = kp * (q_des - q) - kd * qdot

        tau = np.clip(tau, -tau_max, tau_max)  # actuator limits에 맞춤
        if model.actuator_ctrllimited[actuator_id]:
            lo, hi = model.actuator_ctrlrange[actuator_id]
            tau = np.clip(tau, lo, hi)

        data.ctrl[actuator_id] = tau
