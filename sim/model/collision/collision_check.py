import numpy as np
import scipy
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

            if is_adjacent(
                robot, r1.body_id, r2.body_id
            ):  # 이웃한 링크들은 충돌 후보에서 제외
                continue

            yield r1, r2  # 제너레이터 함수


# trajectory_goal 위치마다 충돌 여부 판단 후 충돌영역이라고 판단되면 trajectory clipping
# robot.state: 실제 현재 state
# 전달인자 state: candidate state
def collision_check(robot: RobotModel, state: RobotState, M):
    old_qpos = robot.state.qpos.copy()
    apply_fk(robot, state, M)

    for r1, r2 in collision_pairs(robot):
        d = proxy_distance(r1, r2)
        if d <= 0.005:  # if d <= 0.0:
            state.qpos[:] = old_qpos
            apply_fk(
                robot, state, M
            )  # 충돌이라고 판단되면 이전 state(robot.state)로 rollback
            return True

    return False
