import numpy as np


class BodyNode:
    def __init__(self, name, body_id=None):
        self.name = name
        self.body_id = body_id

        # 기존 MJCF 정보
        self.mass = 0.0
        self.inertia = []
        self.joints = []
        self.attributes = {}

        # open3d 시각화 대상 record 배열
        # record: open3d mesh, metadata를 묶은 wrapper 객체
        self.visual_records = []
        # open3d 충돌 감지 고려 대상 record 배열
        self.collision_records = []

        # kinematic tree
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

    # visual/collision record가 합쳐진 레코드를 반환 (중복 허용)
    def all_records(self):
        return self.visual_records + self.collision_records

    def all_geometries(self):
        geometries = [record.mesh for record in self.all_records()]
        for child in self.children:
            geometries.extend(child.all_geometries())
        return geometries
