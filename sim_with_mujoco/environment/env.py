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

    # viewer, environment(simulator state management) 분리
    # main.py는 렌더링 루프, 시뮬레이션 루프 이중 반복문 구조
    # main.py에서 view(mujoco viewer, glfw panel), env 인스턴스 생성 -> 매 렌더링마다 panel state polling -> polled target으로 ik solver 호출 -> 목표 관절각 env.forward() ->

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

    def step(self, nstep=1):
        mujoco.mj_step(self.model, self.data, nstep)
        return

    def render():
        # 렌더링 주기 - 시뮬레이션 주기 분리
        # viewer 인스턴스의 render api 호출해서 window buffer 업데이트
        # 렌더링 로직은 viewer 인스턴스에서 전담하고 env.render()는 wrapper 용도
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

    def forward(self, qpos):
        # ik solver 호출 (보류)
        self.data.qpos = qpos
        mujoco.mj_forward(self.model, self.data)
        return
