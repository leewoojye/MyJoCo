import mujoco
import numpy as np


# site/body T 모두 world frame 기준
def get_site_T(data, site_id):
    T = np.eye(4)  # 4x4 단위행렬 생성
    T[:3, 3] = data.xpos[site_id]
    T[:3, :3] = data.xmat[site_id].reshape(3, 3)
    return T


def get_body_T(data, body_id):
    T = np.eye(4)  # 4x4 단위행렬 생성
    T[:3, 3] = data.xpos[body_id]
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    return T


def get_body_twist(model, data, ee_id, joint_ids):
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=int)

    # body jacobian
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)

    J = np.vstack([jacr, jacp])[:, dof_ids]  # active joints에 대해서만 트위스트 계산

    qvel = data.qvel[dof_ids]
    twist_current = J @ qvel
    return twist_current


# active joint 자코비안의 최소 특이값과 전체 특이값을 반환
def get_singular_values(model, data, body_id, joint_ids):
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=int)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    J = np.vstack([jacr, jacp])[:, dof_ids]

    # 자코비안 행렬 특이값 분해
    singular_values = np.linalg.svd(J, compute_uv=False)

    return singular_values[-1], singular_values


# mj_objectVelocity()을 이용한 현재 바디 트위스트 계산
def get_body_twist_general(model, data, ee_body_name):
    twist = np.zeros(6)

    ee_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        ee_body_name,
    )

    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        ee_id,
        twist,
        0,  # 0 or 1: world/body frame orientation
    )

    return twist
