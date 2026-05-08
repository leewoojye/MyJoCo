import numpy as np
import scipy
from sim.model.kinematics.fk import compute_fk, apply_fk
from sim.model.math3d.screw import unit_screw_axis, screw_hat, screw_vee
from sim.model.math3d.rotation import omega2rotation_matrix
from sim.model.math3d.transform import create_transform_matrix
from sim.model.robot.robot_model import RobotModel
from sim.model.robot.robot_state import RobotState
from sim.model.math3d.lie import Adjoint
from sim.model.kinematics.jacobian import (
    compute_position_jacobian,
    compute_geometric_jacobian,
)


def calculate_grasp(robot: RobotModel, state: RobotState, alpha, isThumb):
    # q = (1 - grasp) q_open + grasp q_closed
    q_open = 0

    if isThumb:  # 엄지 마디와 연결된 관절들 업데이트
        # 엄지의 초기 자세는 손바닥과 수직에 가깝고, qpos도 0이 아님
        # 엄지 자세 q를 배열로 하드코딩
        q_open_list = [0.3, -1.57, 0.35, 0.25]
        q_closed_list = [0.4, -1.57, 0.8, 0.7]
        for index, i in enumerate(range(1, 5)):  # joint 1부터 4까지 순회
            finger_node = robot.body_node_for(f"finger_r_link{i}")
            joint = finger_node.joints[0]
            qpos_index = joint["qpos_addr"]
            state.qpos[qpos_index] = (1 - alpha) * q_open_list[
                index
            ] + alpha * q_closed_list[index]
    else:  # 엄지를 제외한 관절들 업데이트
        for i in range(5, 21):
            finger_node = robot.body_node_for(f"finger_r_link{i}")
            joint = finger_node.joints[0]
            q_closed = joint["range"][1]
            qpos_index = joint["qpos_addr"]
            state.qpos[qpos_index] = (1 - alpha) * q_open + alpha * q_closed

    return state


def apply_grasp(robot: RobotModel, state: RobotState, M, alpha, isThumb=False):
    state = calculate_grasp(robot, state, alpha, isThumb)
    robot = apply_fk(robot, state, M)

    return robot
