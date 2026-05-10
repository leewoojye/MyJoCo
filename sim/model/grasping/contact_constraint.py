import numpy as np
from scipy.optimize import linprog
import math

from sim.model.collision.collision_check import build_contact_candidates
from sim.model.grasping.contact import ContactPoint, ContactType
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


def compute_contact_force_sum(
    contact_points,
    displacement,
    normal_force=1.0,
    friction_coefficient=0.2,
):
    sum_force = np.zeros(3)

    for contact in contact_points:
        if contact.normal is None:
            continue

        normal = np.asarray(contact.normal, dtype=float)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue

        normal = normal / norm
        normal_amount = np.dot(displacement, normal)

        if normal_amount < 0:
            contact.force = np.zeros(3)
            continue

        normal_component = normal_amount * normal
        # 접선 방향 (tangent) = 마찰력 방향, 접촉면으로 미끄러지는 방향
        tangent_delta = displacement - normal_component
        tangent_norm = np.linalg.norm(tangent_delta)

        friction_force = np.zeros(3)
        if tangent_norm > 1e-8:
            tangent_direction = tangent_delta / tangent_norm
            friction_force = friction_coefficient * normal_force * tangent_direction

        # 이번 스텝에서 contact point에 가해진 force로 갱신
        # 접촉힘 = 법선힘 + 접선힘
        contact.force = normal_force * normal + friction_force

        sum_force += contact.force

    return sum_force


# 접촉점의 힘으로 인한 강체의 이동을 표현
def apply_body_translation(robot, body_name, displacement):
    body_node = robot.body_node_for(body_name)
    joint = body_node.joints[0]
    addr = joint["qpos_addr"]

    robot.state.qpos[addr : addr + 3] += displacement

    delta = np.eye(4)
    delta[:3, 3] = displacement

    for node in body_node.iter_nodes():
        for record in node.all_records():
            record.mesh.transform(delta)

        node.world_transform = delta @ node.world_transform


# 접촉점 force들을 열벡터로 갖는 matrix를 wrench matrix로 변환
# 단일 접촉점에 대해 여러 wrench 벡터가 있을 수 있고, 이들을 열벡터로 하는 행렬이 wrench matrix G
def force_to_wrench_matrix(point: ContactPoint):
    # for i in range(len(f)):
    #     G[:,i]=
    return


# 접촉점 wrench들을 열벡터로 갖는 wrench matrix G
# def contact_wrench_matrix():
#     return


def composite_wrench_cone():
    return


def single_contact_constraint():
    return


def multiple_contact_constraint():
    # impenetrability_check()
    return


# 비침투 조건
def impenetrability_check(wrench, V_a, V_b):
    # 접촉 법선 wrench와 두 물체의 트위스트 V_a, V_b
    V_rel = V_a - V_b  # 상대 트위스트
    return wrench.T @ (V_rel) >= 0


# form closure에서는 모든 접촉점의 wrench로 wrench matrix G를 만들고,
# force closure는 마찰력을 고려하기에 한 접촉점에서 여러 wrench르 모아 G를 만듦
def form_closure_check():
    contact_candidates = build_contact_candidates

    # 조건1. G의 rank가 공간 차원 전체
    rank = np.linalg.matrix_rank()
    if rank != 6:  # 평면인 경우 공간인 경우 분기처리하기
        return False
    # 조건2. Gk=0,k>0를 만족하는 k가 존재
    if not has_positive_k():
        return False
    return True


def force_closure_check():
    return
