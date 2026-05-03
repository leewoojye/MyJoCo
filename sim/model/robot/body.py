import numpy as np


class MuJoCoBodyNode:
    def __init__(self, name, body_id=None):
        self.name = name
        self.body_id = body_id

        # 1. MuJoCo XML 정보 보존 영역
        self.mass = 0.0
        self.inertia = []
        self.joints = []
        self.attributes = {}

        # 2. Open3D 시각화 객체 영역
        self.geometries = []
        self.geometry_records = []

        # 3. 계층 구조 보존 영역 (Kinematic Tree)
        self.parent = None
        self.children = []

        # 상대적 위치 변환 행렬 (body pos, quat 속성 보존)
        self.local_transform = np.eye(4)
        self.world_transform = np.eye(4)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def iter_nodes(self):
        yield self
        for child in self.children:
            yield from child.iter_nodes()

    def all_geometries(self):
        geometries = list(self.geometries)
        for child in self.children:
            geometries.extend(child.all_geometries())
        return geometries
