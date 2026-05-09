from tkinter.tix import Tree

import numpy as np
import scipy
from typing import Tuple
from sim.model.collision.distance import proxy_distance
from sim.model.kinematics.fk import compute_fk, apply_fk
from sim.model.kinematics.ik import apply_ik
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


def collect_collision_records(robot: RobotModel):
    records = []
    for node in robot.root_body.iter_nodes():
        records.extend(node.collision_records)

    return records


def is_adjacent(robot: RobotModel, body_id1, body_id2):
    node1 = robot.body_nodes[body_id1]
    node2 = robot.body_nodes[body_id2]

    return node1.parent is node2 or node2.parent is node1


def collision_pairs(robot: RobotModel):
    records = collect_collision_records(robot)

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            r1 = records[i]
            r2 = records[j]

            if r1.body_id == r2.body_id:
                continue

            if is_adjacent(robot, r1.body_id, r2.body_id):  # 이웃한 링크들은 충돌 후보에서 제외
                continue

            yield r1, r2  # 제너레이터 함수


# 후보 state에서 충돌이나 접촉이 일어나는지 판단하는 함수
# return_contacts가 참이면 후보 접촉점 배열을 반환
def collision_check(
    robot: RobotModel, state: RobotState, M, return_contacts=False
) -> Tuple[bool, bool, np.ndarray]:  # (is_collision, is_contact)을 나타내는 불린 튜플과 접촉점 배열 반환
    is_contact = False  # contact 여부를 나타내는 flag 변수
    contact_candidates = []
    old_qpos = robot.state.qpos.copy()
    candidate_qpos = state.qpos.copy()
    apply_fk(robot, state, M)

    for r1, r2 in collision_pairs(robot):
        d = proxy_distance(r1, r2)
        if d <= 0.005:  # if d <= 0.0: # collision detection
            state.qpos[:] = candidate_qpos
            robot.state.qpos[:] = old_qpos
            apply_fk(robot, robot.state, M)  # 이전 상태(robot.state)로 rollback
            return True, False, None
        elif 0.005 < d <= 0.01:  # contact detection
            is_contact = True  # 충돌과 접촉이 공존할 수 있기에 충돌 감지를 먼저 모두 거치고 접촉 여부를 반환하게 함
            if return_contacts:
                contact_candidates.append((r1, r2))

    if is_contact:
        unique_candidates = set(contact_candidates)
        state.qpos[:] = candidate_qpos
        robot.state.qpos[:] = old_qpos
        apply_fk(robot, robot.state, M)

        return True, True, unique_candidates

    # 이전 상태(robot.state)로 rollback
    state.qpos[:] = candidate_qpos
    robot.state.qpos[:] = old_qpos
    apply_fk(robot, robot.state, M)

    return False, False, None


# def contact_pairs():
#     return


def contact_normal():
    return


# collision_check()로부터 후보 접촉점 배열을 받아 ContactPoint 배열을 생성
# 접촉점에서의 위치와 힘
def build_contact_candidates(robot: RobotModel, state: RobotState, M):
    is_collision, is_contact, contact_candidates = collision_check(robot, state, M, True)
    if is_collision or not is_contact:
        return None
    # 접촉점 개수가 7개 이상이 되는지 나타내는 flag 변수도 반환 (3차원 공간 기준)
    # 이는 form closure 평가에서 wrench 벡터들로 positive span을 만들 때 이용
    return
