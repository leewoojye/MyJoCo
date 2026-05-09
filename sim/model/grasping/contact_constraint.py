from enum import Enum
import numpy as np
import math

from spots.utils.util import skew


class ContactType(Enum):
    B = "breaking_free"
    S = "sliding"
    R = "rolling"  # rolling/sticking


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


def single_contact_constraint():
    return


# edge wrench/friction로 연속적인 전체 원뿔을 근사
def cone_approximation():
    return


def friction_cone_constraint():

    return


# wrench들을 열벡터로 갖는 wrench matrix G
def contact_wrench_matrix():
    return


def grasp_matrix():
    return


# 비침투 조건
def impenetrability_check(wrench, V_a, V_b):
    # 접촉 법선 wrench와 두 물체의 트위스트 V_a, V_b
    V_rel = V_a - V_b  # 상대 트위스트
    return wrench.T @ (V_rel) >= 0
