from dataclasses import dataclass
from enum import Enum  # 상수를 묶는 용도
import numpy as np

from sim.model.math3d.rotation import create_skew
from sim.model.robot.geometry import GeomRecord


class ContactType(Enum):
    B = "breaking_free"
    S = "sliding"
    R = "rolling"  # rolling/sticking
    P = "penetration"


# 접촉점 클래스
@dataclass
class ContactPoint:
    record_a: GeomRecord
    record_b: GeomRecord

    # 각 프록시 위의 contact point
    p_a: np.ndarray
    p_b: np.ndarray
    distance: float  # d = 0: 접촉, d < 0: 관통
    depth: float  # penetration depth, 손과 캔이 관통 상태일 때도 contact 상태로 취급하고 관통한 깊이를 캔 이동거리 계산에 활용함
    # 법선 단위 벡터
    normal: np.ndarray

    # 트위스트
    V_a: np.ndarray
    V_b: np.ndarray
    # 가상 접촉점: 접촉에 있어 소량의 거리를 허용하므로, 실제 접촉점과 별개로 가상 접촉점이 필요
    point: np.ndarray | None = None
    # 접촉점 유형
    contact_type: ContactType = ContactType.B  # default mode: breaking-free
    # 속도
    v_a: np.ndarray | None = None
    v_b: np.ndarray | None = None
    v_rel: np.ndarray | None = None  # 상대속도
    # 힘
    force: np.ndarray | None = None

    def __post_init__(self):
        self.V_a = np.asarray(self.V_a, dtype=float).reshape(6)
        self.V_b = np.asarray(self.V_b, dtype=float).reshape(6)
        self.depth = max(0, -self.distance)
        self.point = 0.5 * (self.p_a + self.p_b)
        self.v_a = self.V_a[3:] + create_skew(self.V_a[:3]) @ self.p_a
        self.v_b = self.V_b[3:] + create_skew(self.V_b[:3]) @ self.p_b
        self.v_rel = self.v_a - self.v_b
        self.contact_type = contact_type(self)


# contact type 분류
def contact_type(contactpoint: ContactPoint):  # 트위스트 V, 위치 p 모두 space frame
    if contactpoint.V_a is None or contactpoint.V_b is None:
        if contactpoint.distance < 0:
            return ContactType.P
        return ContactType.B

    V_a = np.asarray(contactpoint.V_a, dtype=float).reshape(6)
    V_b = np.asarray(contactpoint.V_b, dtype=float).reshape(6)
    w_a = V_a[:3]
    w_b = V_b[:3]
    v_a = V_a[3:]
    v_b = V_b[3:]

    # 접촉점 p에서 두 강체의 속도가 같으면 붙어있는 rolling 모드로 봄
    if np.allclose(
        v_a + create_skew(w_a) @ contactpoint.p_a,
        v_b + create_skew(w_b) @ contactpoint.p_b,
    ):
        return ContactType.R
    elif contactpoint.distance < 0:
        return ContactType.P

    return ContactType.B
