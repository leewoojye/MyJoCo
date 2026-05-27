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
