import mujoco
import numpy as np

from sim.model.kinematics.ik import calculate_twist_error
from sim_with_mujoco.environment.env import Environment
from sim_with_mujoco.utils.ik import damped_pseudoinverse
from sim_with_mujoco.utils.kinematics import get_body_jacobian
from sim_with_mujoco.utils.math3d import get_body_twist
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


def taskacc_to_jointacc():
    return


def pd_joint_space():
    return


def ct_joint_space(model, data, q_des, q_dot_des, q_dotdot_des, joint_ids, kp=300, kd=50):
    # inverse dynamic 계산용
    inv_data = mujoco.MjData(model)
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
    tau = inv_data.qfrc_inverse[dof_ids] + inv_data.qfrc_constraint[dof_ids]

    # forcerange 기반 클리핑 (전체 최적화로 계산된 토크 균형이 망가질 것으로 우려, 추후 수정)
    # for i, joint_id in enumerate(joint_ids):
    #     joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    #     actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)

    #     if actuator_id >= 0 and model.actuator_forcelimited[actuator_id]:
    #         lo, hi = model.actuator_ctrlrange[actuator_id]  # 모델마다 range명 상이
    #         tau[i] = np.clip(tau[i], lo, hi)

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
