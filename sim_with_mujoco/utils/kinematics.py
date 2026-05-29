import mujoco
import numpy as np


# finger position interpolation: q = (1 - grasp) q_open + grasp q_closed
def interpolate_finger(model, data, alpha):
    # 네 손가락을 위한 초기 자세
    q_open = 0

    # 엄지의 초기 자세는 손바닥과 수직에 가깝고, qpos도 0이 아님
    # 엄지 자세 q를 배열로 표현
    q_open_list = [0.3, -1.57, 0.35, 0.25]
    q_closed_list = [0.4, -1.25, 0.8, 0.7]

    # 엄지 관절 보간
    for index, i in enumerate(range(1, 5)):  # joint 1부터 4까지 순회
        joint_name = f"finger_r_joint{i}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        qadr = model.jnt_qposadr[joint_id]
        value = (1 - alpha[0]) * q_open_list[index] + alpha[0] * q_closed_list[index]
        data.qpos[qadr] = value
        data.ctrl[actuator_id] = value
    # 네 손가락 관절 보간
    for i in range(5, 21):
        joint_name = f"finger_r_joint{i}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        q_closed = model.jnt_range[joint_id, 1]
        qadr = model.jnt_qposadr[joint_id]
        value = (1 - alpha[1]) * q_open + alpha[1] * q_closed
        data.qpos[qadr] = value
        data.ctrl[actuator_id] = value
    return


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
