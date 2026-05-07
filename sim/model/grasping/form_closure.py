import numpy as np
import scipy
from sim.model.kinematics.fk import compute_fk, apply_fk
from sim.model.math3d.screw import unit_screw_axis, screw_hat, screw_vee
from sim.model.math3d.rotation import omega2rotation_matrix
from sim.model.math3d.transform import create_transform_matrix
from sim.model.robot.robot_model import RobotGeometries
from sim.model.robot.state import RobotState
from sim.model.math3d.lie import Adjoint
from sim.model.kinematics.jacobian import (
    compute_position_jacobian,
    compute_geometric_jacobian,
)


def calculate_grasp(robot: RobotGeometries, state: RobotState, alpha, isThumb):
    # q = (1 - grasp) q_open + grasp q_closed
    q_open = 0

    if isThumb: # 엄지 마디와 연결된 관절 각 업데이트
        for i in range(1, 4):
            finger_node = robot.body_node_for(f"finger_r_link{i}")
            joint = finger_node.joints[0]
            q_closed = joint["range"][1]
            qpos_index = joint["qpos_addr"]
            state.qpos[qpos_index] = (1 - alpha) * q_open + alpha * q_closed
    else: # 엄지를 제외한 관절들 업데이트
        for i in range(5, 16):
            finger_node = robot.body_node_for(f"finger_r_link{i}")
            joint = finger_node.joints[0]
            q_closed = joint["range"][1]
            qpos_index = joint["qpos_addr"]
            state.qpos[qpos_index] = (1 - alpha) * q_open + alpha * q_closed

    return state


def apply_grasp(robot: RobotGeometries, state: RobotState, M, alpha, isThumb=False):
    state = calculate_grasp(robot, state, alpha, isThumb)
    robot = apply_fk(robot, state, M)

    return robot
