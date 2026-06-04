import mujoco
import numpy as np


# joint id: 관절 정보 인덱스 (model.jnt_*)
# dof id: 속도, 가속도, generalized force 인덱스 (data.qvel, data.qacc, data.qfrc_applied, data.qfrc_bias)
# actuator id: 제어 입력 인덱스 (data.ctrl, model.actuator_*)
def actuator_ids_from_joints(model, joint_ids):
    joint_id_set = set(joint_ids)
    actuator_ids = []

    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue

        joint_id = model.actuator_trnid[actuator_id, 0]
        if joint_id in joint_id_set:
            actuator_ids.append(actuator_id)

    return actuator_ids


def joint_ids_from_actuators(model, actuator_ids):
    joint_ids = []

    for actuator_id in actuator_ids:
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue

        joint_ids.append(model.actuator_trnid[actuator_id, 0])

    return joint_ids


def dof_ids_from_joints(model, joint_ids):
    dof_ids = []

    for joint_id in joint_ids:
        # if model.actuator_trntype[joint_id] != mujoco.mjtTrn.mjTRN_JOINT:
        #     continue

        # 회전/직선 관절은 joint id 하나당 dof id 하나
        dof_ids.append(model.jnt_dofadr[joint_id])

    return dof_ids


def joint_ids_from_names(model, joint_names):
    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names], dtype=int) # 인덱스 자료형 명시
    return joint_ids

def joint_ids_from_body(model, body_id):
    jnt_adr = model.body_jntadr[body_id]
    jnt_num = model.body_jntnum[body_id] # body에 달린 joint 개수

    return np.arange(jnt_adr, jnt_adr + jnt_num, dtype=int)
