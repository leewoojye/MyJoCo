import mujoco
import numpy as np

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


# position actuator를 위한 임시 PD controller
def pd_controller(model, data, q_des, q_dot_des, q_dotdot_des, joint_ids):  # qpos_ids, joints_id
    # inverse dynamic 계산용
    inv_data = mujoco.MjData(model)
    mujoco.mj_copyData(inv_data, model, data)

    dof_ids = dof_ids_from_joints(model, joint_ids)
    qpos_ids = model.jnt_qposadr[joint_ids]
    kp = model.actuator_gainprm[dof_ids, 0]
    kd = 60

    qacc_des = (
        q_dotdot_des  # task_to_joint_space()로 인해 active actuator에 대한 원소로만 구성
        + kp * (q_des[qpos_ids] - inv_data.qpos[qpos_ids])
        + kd * (q_dot_des - inv_data.qvel[dof_ids])
    )
    inv_data.qacc[dof_ids] = qacc_des
    mujoco.mj_inverse(model, inv_data)
    tau = inv_data.qfrc_inverse[dof_ids]  # qfrc_inverse(inverse 결과 저장용), qfrc_applied(forward, step 입력용)

    # forcerange 기반 클리핑
    for i, joint_id in enumerate(joint_ids):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)

        if actuator_id >= 0 and model.actuator_forcelimited[actuator_id]:
            lo, hi = model.actuator_forcerange[actuator_id]
            tau[i] = np.clip(tau[i], lo, hi)
    return tau


def pid_controller():
    return


def computed_torque():
    return
