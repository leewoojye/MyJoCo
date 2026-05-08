# 로봇팔과 캔의 mesh/polygon을 근사할 sphere(몸통이나 손가락), capsule(arm) 생성
# 이는 collision check시 두 강체 사이의 거리를 계산하는데 사용
import numpy as np
import open3d as o3d
from open3d import geometry, utility, visualization

from dataclasses import replace
import copy


from sim.model.robot.body import BodyNode
from sim.model.robot.geometry import GeomRecord
from sim.model.robot.robot_model import RobotModel
from sim.model.robot.robot_state import RobotState


# open3d 기본 mesh를 cylinder mesh로 변환
# records의 transform 속성은 충돌 감지 모듈에서 수행, proxy.py는 프록시 생성만 수행
def make_cylinder_proxy(robot: RobotModel, state: RobotState, collision_records):
    # 깊은 복사 옵션1: copy.deepcopy(), 옵션2: dataclasses replace
    # proxy_records = copy.deepcopy(collision_records)
    # proxy_meshes = []
    proxy_records = []

    for record in collision_records:
        mesh = record.mesh
        bbox = mesh.get_oriented_bounding_box()
        axis_length_list = np.asarray(bbox.extent)
        axis_max_length = max(
            axis_length_list[0], max(axis_length_list[1], axis_length_list[2])
        )
        radius = (
            # 1. 길이 평균의 절반
            # 2. 더 긴 축 길이의 절반
            max(
                axis_length_list[i]
                for i in range(3)
                if axis_length_list[i] != axis_max_length
            )
            / 2
        )
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius,
            height=axis_max_length,
            resolution=32,
        )
        proxy_records.append(replace(record, mesh=cylinder))

    return proxy_records


def make_capsule_proxy(robot: RobotModel, state: RobotState):
    return
