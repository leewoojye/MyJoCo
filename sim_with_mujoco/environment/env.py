# dm_control Physics / robosuite MujocoEnv

from typing import Callable, NamedTuple, Optional, Union
import mujoco
import numpy as np
from sim_with_mujoco.mjcf import parser


class Environment:
    # MjModel, MjData
    # body/joint/geom/site/sensor name
    # qpos/qvel/ctrl (named access 기능 추가)
    # getter, setter: tick/data.time
    # forward/step/reset/render etc. wrapper
    # IK solver (보류)
    # contact points 가져오기 (렌더링용)
    # e.e의 site/body id field

    def __init__(self, xml_path, end_effector):
        self.model, self.data = parser(xml_path)
        self.ee_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            end_effector,
        )

    def get_ctrl():
        return

    def set_ctrl():
        return

    def step(self):
        mujoco.mj_step(self.model, self.data)
        return

    def render():
        return

    def get_state():
        return

    def set_state():
        return

    # data.time
    def get_time():
        return

    # def set_time(): # mj_step()에서 관리
    #     return

    def get_tick():
        return

    def set_tick():
        return

    # transformation matrix getter: data.xpos + data.xmat
    def get_space_T(self, site_id):
        T = np.eye(4)  # 4x4 단위행렬 생성
        T[:3, :3] = self.data.xpos[site_id]
        T[:3, 3] = self.data.xmat[site_id].reshape(3, 3)
        return T

    def get_body_T(self, body_id):
        T = np.eye(4)  # 4x4 단위행렬 생성
        T[:3, :3] = self.data.xpos[body_id]
        T[:3, 3] = self.data.xmat[body_id].reshape(3, 3)
        return T

    # rotation matrix getter
    def get_rotation():
        R = np.eye(3)
        return

    def reset():
        return

    def forward():
        # ik solver 호출

        return
