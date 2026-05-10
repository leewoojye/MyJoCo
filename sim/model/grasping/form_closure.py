import numpy as np
import scipy
from sim.model.grasping.contact_constraint import force_to_wrench, has_positive_k
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
            state.qpos[qpos_index] = (1 - alpha) * q_open_list[index] + alpha * q_closed_list[index]
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


# form closure에서는 모든 접촉점의 wrench로 wrench matrix G를 만들고,
# force closure는 마찰력을 고려하기에 한 접촉점에서 여러 wrench르 모아 G를 만듦
def check_form_closure(contact_points):
    # 조건0: positive span이 공간 전체를 덮어야 해 최소 7개의 접촉점을 가져야함 (공간 기준)
    if len(contact_points) < 7:
        return False
    
    wrenches = [
        force_to_wrench(contact.point, contact.normal) for contact in contact_points if contact.normal is not None
    ]

    if not wrenches:
        return False

    G = np.column_stack(wrenches)

    # 조건1. G의 rank가 공간 차원 전체
    rank = np.linalg.matrix_rank(G)

    if rank != 6:  # 평면인 경우 공간인 경우 분기처리하기
        return False

    # 조건2. Gk = 0, k > 0를 만족하는 k가 존재
    if not has_positive_k(G):
        return False

    return True
