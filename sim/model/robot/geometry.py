from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import open3d as o3d


def make_transform(position, rotation_matrix):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = position
    return T


@dataclass
class GeometryRecord:
    mesh: o3d.geometry.TriangleMesh
    geom_id: int
    geom_name: Optional[str]
    body_id: int
    body_name: Optional[str]
    geom_type: str
    mesh_id: Optional[int]
    mesh_name: Optional[str]
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
    geometry_records: List[GeometryRecord]
