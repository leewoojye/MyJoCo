from enum import Enum  # 상수를 묶는 용도
import numpy as np
from scipy.optimize import linprog
import math

from sim.model.collision.collision_check import search_contact_candidates
from sim.model.math3d.rotation import create_skew
from spots.utils.util import skew


class ContactType(Enum):
    B = "breaking_free"
    S = "sliding"
    R = "rolling"  # rolling/sticking


# 접촉점 클래스
class ContactPoint:
    # 디폴트 접촉 모드: B(breaking-free)
    # contact_type = ContactType.B

    def __init__(self, pos, force, V_a, V_b, p_a, p_b):
        self.pos = pos
        self.force = force
        self.wrench = force_to_wrench(pos, force)
        self.contact_type = contact_type(V_a, V_b, p_a, p_b)


# contact type 설정
def contact_type(V_a, V_b, p_a, p_b):  # 트위스트 V, 위치 p 모두 space frame
    w_a = V_a[:3, 0]
    w_b = V_b[:3, 0]
    v_a = V_a[3:, 0]
    v_b = V_b[3:, 0]

    # 접촉점 p에서 두 강체의 속도가 같으면 붙어있는 rolling 모드로 봄
    if v_a + skew(w_a) @ p_a == v_b + skew(w_b) @ p_b:
        return ContactType.R

    return ContactType.B


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
    contact_candidates = search_contact_candidates
    
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
