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


# 링크의 z축이 인접한 관절 사이의 길이, 즉 링크의 길이라고 가정하고 링크의 길이 계산
def get_link_length(robot: RobotModel, body_name, record):
    body_node = robot.body_node_for(body_name)
    z = body_node.world_transform[:3, 2].copy()  # 링크의 z축
    z = z / np.linalg.norm(z)
    v = np.asarray(record.mesh.vertices)  # 링크 mesh를 구성하는 모든 정점
    length = (v @ z).max() - (v @ z).min()

    return length


# open3d 기본 mesh를 cylinder mesh로 변환
# records의 transform 속성은 충돌 감지 모듈에서 수행, proxy.py는 프록시 생성만 수행
def make_cylinder_proxy(robot: RobotModel, state: RobotState, collision_records):
    # 깊은 복사 옵션1: copy.deepcopy(), 옵션2: dataclasses replace
    # proxy_records = copy.deepcopy(collision_records)
    # proxy_meshes = []
    proxy_records = []

    for record in collision_records:
        mesh = record.mesh
        bbox = (
            mesh.get_oriented_bounding_box()
        )  # open3d 내장함수로 mesh의 바운딩박스를 얻음
        axis_length_list = np.asarray(bbox.extent)

        # 실린더 높이 옵션1: 링크의 길이를 실린더의 높이로 설정
        # height = get_link_length(robot, record.body_name, record)

        # 실린더 높이 옵션2: bbox의 가장 긴 축을 링크의 길이로 가정
        axis_max_length = max(
            axis_length_list[0], max(axis_length_list[1], axis_length_list[2])
        )
        radius = (
            # 실린더 반지름 옵션1. 두 길이 평균의 절반
            # 실린더 반지름 옵션2. 더 긴 한 축의 절반
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
        )  # open3d create 함수는 기본적으로 원점 기준 객체를 반환
        proxy_records.append(replace(record, mesh=cylinder))

        T = np.eye(4)

    return proxy_records


def make_capsule_proxy(robot: RobotModel, state: RobotState):
    return
