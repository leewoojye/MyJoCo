from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import open3d as o3d


@dataclass
class GeomRecord:
    # open3d로 표현된 mesh (실제 렌더링에 사용)
    mesh: o3d.geometry.TriangleMesh
    # 정적인 메타데이터 필드 (MJCF metadata)
    geom_id: int
    geom_name: Optional[str]
    body_id: int
    body_name: Optional[str]
    geom_type: str
    mesh_id: Optional[int]
    mesh_name: Optional[str]
    # 동적인 메타데이터 필드ㄷ (현재 open3d mesh의 위치)
    transform: np.ndarray
    is_end_effector: bool = False


@dataclass
class EndEffector:
    name: str
    body_name: str
    body_id: int
    position: np.ndarray
    rotation: np.ndarray
    transform: np.ndarray
    geom_records: List[GeomRecord]
