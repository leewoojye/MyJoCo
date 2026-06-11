import mujoco
import numpy as np

from sim.model.math3d.transform import create_transform_matrix
from sim_with_mujoco.utils.mj import joint_ids_from_body


# finger position interpolation: q = (1 - grasp) q_open + grasp q_closed
def interpolate_finger(model, data, alpha, is_left=False):
    # 네 손가락을 위한 초기 자세
    q_open = 0
    # spread_joints = {5, 9, 13, 17}

    # 엄지의 초기 자세는 손바닥과 수직에 가깝고, qpos도 0이 아님
    # 엄지 자세 q를 배열로 표현
    q_open_list = [0.3, -1.57, 0.35, 0.25]
    q_closed_list = [0.4, -1.25, 0.8, 0.7]
    # finger_open = [0.0, 0.45, 0.35, 0.25]

    # 엄지 관절 보간
    for index, i in enumerate(range(1, 5)):  # joint 1부터 4까지 순회
        if is_left:
            joint_name = f"finger_l_joint{i}"
        else:
            joint_name = f"finger_r_joint{i}"

        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)

        value = (1 - alpha[0]) * q_open_list[index] + alpha[0] * q_closed_list[index]
        data.ctrl[actuator_id] = value

    # 네 손가락 관절 보간
    for i in range(5, 21):
        if is_left:
            joint_name = f"finger_l_joint{i}"
        else:
            joint_name = f"finger_r_joint{i}"

        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)

        # if i in spread_joints:
        #     value = 0.0  # 손가락 벌림 고정
        # else:
        #     q_closed = model.jnt_range[joint_id, 1]
        #     value = (1 - alpha[1]) * q_open + alpha[1] * q_closed

        # finger_joint_index = (i - 5) % 4
        # q_open = finger_open[finger_joint_index]
        q_closed = model.jnt_range[joint_id, 1]
        value = (1 - alpha[1]) * q_open + alpha[1] * q_closed
        data.ctrl[actuator_id] = value


def get_dh_params(model, data, body_id):
    joint_id = joint_ids_from_body(model, body_id)[0]
    qpos_id = model.jnt_qposadr[joint_id]
    child_body_id = np.where(model.body_parentid == body_id)[0]  # serial manipulator

    theta = data.qpos[qpos_id]

    pos = model.body_pos[child_body_id]
    quat = model.body_quat[child_body_id]

    R = np.zeros((3, 3))
    mujoco.mju_quat2Mat(R.ravel(), quat)  # quaternion->회전행렬 변환

    T = create_transform_matrix(R, pos)

    a = T[0, 3]
    d = T[2, 3]
    alpha = np.arctan2(T[2, 1], T[2, 2])

    return a, alpha, d, theta


def get_site_jacobian(model, data, site_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacr, jacp])
    return J


def get_body_jacobian(model, data, body_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    J = np.vstack([jacr, jacp])
    return J
