import numpy as np
import open3d as o3d
import scipy
from typing import Tuple
from sim.model.collision.distance import get_proxy, proxy_distance
from sim.model.collision.proxy import BoxProxy, CapsuleProxy
from sim.model.grasping.contact import ContactPoint
from sim.model.kinematics.fk import apply_fk
from sim.model.robot.robot_model import RobotModel
from sim.model.robot.robot_state import RobotState


def collect_collision_records(robot: RobotModel):
    records = []
    for node in robot.root_body.iter_nodes():
        records.extend(node.collision_records)

    return records


def is_adjacent(robot: RobotModel, body_id1, body_id2):
    node1 = robot.body_nodes[body_id1]
    node2 = robot.body_nodes[body_id2]

    return node1.parent is node2 or node2.parent is node1


def is_right_hand_body(body_name):
    return body_name in {"arm_r_link7", "hx5_r_base"} or body_name.startswith("finger_r_link")


# collision pair 조합 범위를 제한 (ex. 손-캔, 손-테이블만)
def should_check_collision_pair(record_a, record_b):
    body_names = {record_a.body_name, record_b.body_name}
    has_right_hand = any(is_right_hand_body(body_name) for body_name in body_names)

    return has_right_hand and ("pr_cokeCan" in body_names or "base_table" in body_names)


# 가능한 접촉/충돌 후보 쌍들을 반환
def collision_pairs(robot: RobotModel):
    records = collect_collision_records(robot)

    for i in range(len(records)):
        for j in range(i + 1, len(records)):  # pair 중복 처리 효과
            r_a = records[i]
            r_b = records[j]

            if r_a.body_id == r_b.body_id:
                continue

            if is_adjacent(robot, r_a.body_id, r_b.body_id):  # 이웃한 링크들은 충돌 후보에서 제외
                continue

            yield r_a, r_b  # 제너레이터 함수


# 후보 state에서 충돌이나 접촉이 일어나는지 판단하는 함수
# return_contacts가 참이면 후보 접촉점 배열을 반환
def collision_check(
    robot: RobotModel, state: RobotState, home_pose, return_contacts=False
) -> Tuple[bool, bool, np.ndarray]:  # (is_collision, is_contact)을 나타내는 불린 튜플과 접촉점 배열 반환
    is_contact = False  # contact 여부를 나타내는 flag 변수
    contact_candidates = []
    old_qpos = robot.state.qpos.copy()
    candidate_qpos = state.qpos.copy()

    ignored_pairs = {  # 캔-테이블, 테이블-바닥, 바닥-캔 사이 접점은 충돌 감지 대상에서 배제하기 위함
        frozenset({"world", "pr_cokeCan"}),
        frozenset({"base_table", "pr_cokeCan"}),
        frozenset({"world", "base_table"}),
    }

    # 캐싱 딕셔너리: 계산한 결과를 저장했다가 재사용함
    proxy_cache = {}
    apply_fk(robot, state, home_pose)

    for r_a, r_b in collision_pairs(robot):  # collision_pairs에서 중복 처리된 pair를 가져옴
        # free joint 물체들 사이의 접촉은 충돌에서 제외 (추후 수정)
        if frozenset({r_a.body_name, r_b.body_name}) in ignored_pairs:
            continue

        # 기존 전체 collision pair 검사 대신, 현재는 오른손-캔/오른손-테이블만 검사함
        if not should_check_collision_pair(r_a, r_b):
            continue

        # 이미 거리가 너무 가까운 손가락 마디는 충돌 페어에서 제외
        # if "finger_" in r_a.body_name and "finger_" in r_b.body_name:
        #     continue

        # if (
        #     (r_a.body_name == "arm_l_link7" and "finger_l_" in r_b.body_name)
        #     or (r_b.body_name == "arm_l_link7" and "finger_l_" in r_a.body_name)
        #     or (r_a.body_name == "arm_l_link7" and "finger_r_" in r_b.body_name)
        #     or (r_b.body_name == "arm_l_link7" and "finger_r_" in r_a.body_name)
        # ):
        #     continue

        p_a, p_b, d = proxy_distance(r_a, r_b, proxy_cache)
        body_names = {r_a.body_name, r_b.body_name}

        # 손이 물체를 관통하는 상황은 충돌보다 접촉에 가까운 것으로 보고,
        if "pr_cokeCan" in body_names and d <= 0.005:
            is_contact = True

            if return_contacts:
                contact_candidates.append(
                    ContactPoint(
                        record_a=r_a,
                        record_b=r_b,
                        point=0.5 * (p_a + p_b),
                        p_a=p_a,
                        p_b=p_b,
                        normal=contact_normal(p_a, p_b),
                        distance=d,
                        depth=max(0.0, -d),  # penetration depth
                        V_a=state.body_twists.get(r_a.body_name, np.zeros(6)),
                        V_b=state.body_twists.get(r_b.body_name, np.zeros(6)),
                    )
                )

        elif d <= 0.0:  # if d <= 0.0: # collision detection: self-collision, 로봇-테이블 충돌
            state.qpos[:] = candidate_qpos
            robot.state.qpos[:] = old_qpos
            apply_fk(robot, robot.state, home_pose)  # 이전 상태(robot.state)로 rollback

            return True, False, None

        # elif 0.001 < d <= 0.005:  # contact detection
        #     is_contact = True  # 충돌과 접촉이 공존할 수 있기에 충돌 감지를 먼저 모두 거치고 접촉 여부를 반환하게 함

        #     if return_contacts:
        #         contact_candidates.append(
        #             ContactPoint(
        #                 record_a=r_a,
        #                 record_b=r_b,
        #                 point=0.5 * (p_a + p_b),
        #                 p_a=p_a,
        #                 p_b=p_b,
        #                 normal=contact_normal(p_a, p_b),
        #                 distance=d,
        #                 depth=max(0.0, -d),
        #             )
        #         )

    if is_contact:
        state.qpos[:] = candidate_qpos
        robot.state.qpos[:] = old_qpos
        apply_fk(robot, robot.state, home_pose)

        return True, True, contact_candidates

    # 이전 상태(robot.state)로 rollback
    state.qpos[:] = candidate_qpos
    robot.state.qpos[:] = old_qpos
    apply_fk(robot, robot.state, home_pose)

    return False, False, None


# a->b 방향 법선 단위 벡터
def contact_normal(point_a, point_b):
    normal = point_b - point_a
    norm = np.linalg.norm(normal)

    if norm < 1e-8:  # 크기가 0으로 수렴하는 법선벡터 처리
        return None

    return normal / norm


# collision_check()로부터 후보 접촉점 배열을 받아 ContactPoint 배열을 생성
# 접촉점에서의 위치와 힘
def build_contact_candidates(robot: RobotModel, state: RobotState, M):
    is_collision, is_contact, contact_candidates = collision_check(robot, state, M, True)
    contactpoint_list = []
    if is_collision or not is_contact:
        return None

    # 접촉점 개수가 7개 이상이 되는지 나타내는 flag 변수도 반환 (3차원 공간 기준) -> 이건 호출부에서 처리하는게 나을듯
    return contactpoint_list
