from dataclasses import dataclass
from enum import Enum  # 상수를 묶는 용도
import numpy as np

from sim.model.robot.geometry import GeomRecord


class ContactType(Enum):
    B = "breaking_free"
    S = "sliding"
    R = "rolling"  # rolling/sticking


# 접촉점 클래스
@dataclass
class ContactPoint:
    record_a: GeomRecord
    record_b: GeomRecord

    # 가상 접촉점: 접촉에 있어 소량의 거리를 허용하므로, 실제 접촉점과 별개로 가상 접촉점이 필요
    point: np.ndarray

    # 각 프록시 위의 contact point
    point_a: np.ndarray
    point_b: np.ndarray
    distance: float

    normal: np.ndarray

    contact_type: ContactType = ContactType.B  # default mode: breaking-free

    # 선택 정보
    V_rel: np.ndarray | None = None  # 상대 트위스트
    force: np.ndarray | None = None
