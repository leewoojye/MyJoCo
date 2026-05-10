import numpy as np
from scipy.optimize import linprog

from sim.model.math3d.rotation import create_skew


# 선형계획 문제를 푸는 linprog() 사용
def has_positive_k(G, eps=1e-6):
    m = G.shape[1]

    return linprog(
        c=np.zeros(m),
        A_eq=np.vstack([G, np.ones(m)]),
        b_eq=np.r_[np.zeros(G.shape[0]), 1],
        bounds=[(eps, None)] * m,
        method="highs",
    ).success


# edge wrench/friction로 연속적인 전체 원뿔을 근사
def cone_approximation():
    return


# coulomb 마찰 모델에 기반해 friction cone 생성
# 접촉점 하나에 대해 friction cone 하나
def friction_cone_constraint():
    # half_angle

    return


def wrench_cone_constraint():
    return


def force_to_wrench(pos, force):
    moment = create_skew(pos) @ force
    wrench = np.concatenate([moment, force])

    return wrench


# 접촉점들에 의해 가해지는 힘 벡터를 합산
# 토크/회전은 반영하지 못하고 있음 (추후 수정)
# 법선 방향: normal force
# 접선 방향: friction force
def sum_contact_force(
    contact_points,
    normal_force=1.0,  # 법선 힘의 크기를 호출부에서 받는 형태
    friction_coefficient=0.2,
):
    sum_force = np.zeros(3)

    for contact in contact_points:
        if contact.normal is None:
            continue

        v_rel = contact.v_rel  # 물체 기준에서 본 손 속도

        normal = np.asarray(contact.normal, dtype=float)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue

        normal = normal / norm
        normal_amount = np.dot(v_rel, normal)

        if normal_amount < 0:
            contact.force = np.zeros(3)
            continue

        # normal_component = normal_amount * normal  # 상대속도의 법선 성분으로, 접선 성분을 구하기 위해 계산
        # # 접선 벡터 (tangent, 마찰력 방향) 계산
        # tangent_delta = v_rel - normal_component
        # tangent_norm = np.linalg.norm(tangent_delta)

        friction_force = np.zeros(3)
        if contact.tangent is not None:
            tangent_direction = contact.tangent
            friction_amount = np.linalg.norm(friction_coefficient * normal_force)
            friction_force = friction_amount * tangent_direction

        # 이번 스텝에서 contact point에 가해진 force로 갱신
        # 접촉힘 = 법선힘 + 접선힘
        contact.force = normal_force * normal + friction_force

        sum_force += contact.force  # 문제: 힘 벡터 방향이 서로 비슷할 경우

    return sum_force


# 접촉점의 힘으로 인한 강체의 이동을 표현
def apply_body_translation(robot, body_name, displacement):
    body_node = robot.body_node_for(body_name)
    joint = body_node.joints[0]
    addr = joint["qpos_addr"]

    # qvel update는 tick/controller에서 실제 적용된 qpos 차분과 dt로 처리
    robot.state.qpos[addr : addr + 3] += displacement

    delta = np.eye(4)
    delta[:3, 3] = displacement

    for node in body_node.iter_nodes():
        for record in node.all_records():
            record.mesh.transform(delta)

        node.world_transform = delta @ node.world_transform


# 접촉점 force들을 열벡터로 갖는 matrix를 wrench matrix로 변환
# 단일 접촉점에 대해 여러 wrench 벡터가 있을 수 있고, 이들을 열벡터로 하는 행렬이 wrench matrix G
# def force_to_wrench_matrix(point: ContactPoint):
#     # for i in range(len(f)):
#     #     G[:,i]=

#     return


# 접촉점 wrench들을 열벡터로 갖는 wrench matrix G
# 한 접촉점에 대한 힘(법선, 접선, 마찰력..) 행렬?
# def contact_wrench_matrix(contact):
#     forces = [contact.force for contact in contact_points if contact.force is not None]
#     G = np.column_stack(forces)

#     return G


def composite_wrench_cone():
    return


def single_contact_constraint():
    return


def multiple_contact_constraint():
    return


# 비침투 조건
def check_impenetrability(contact):  # 단일 접촉점
    return contact.normal.T @ contact.V_rel >= 0  # 상대 트위스트의 법선 성분
