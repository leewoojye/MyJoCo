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
        bbox = mesh.get_oriented_bounding_box()  # open3d 내장함수로 mesh의 바운딩박스를 얻음
        axis_length_list = np.asarray(bbox.extent, dtype=float)

        # 실린더 높이 옵션1: 링크의 길이를 실린더의 높이로 설정
        # height = get_link_length(robot, record.body_name, record)

        # 실린더 높이 옵션2: bbox의 가장 긴 축을 링크의 길이로 가정
        long_axis = int(np.argmax(axis_length_list))
        short_axes = [i for i in range(3) if i != long_axis]
        axis_max_length = float(axis_length_list[long_axis])

        radius = max(axis_length_list[short_axes[0]], axis_length_list[short_axes[1]]) / 2
        if axis_max_length <= 0.0 or radius <= 0.0:
            proxy_records.append(replace(record, mesh=mesh))
            continue

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius,
            height=axis_max_length,
            resolution=32,
        )  # open3d create 함수는 기본적으로 원점 기준 객체를 반환

        # cylinder 방향 설정
        z_axis = bbox.R[:, long_axis]
        x_axis = bbox.R[:, short_axes[0]]
        x_axis = x_axis - z_axis * np.dot(z_axis, x_axis)

        if np.linalg.norm(x_axis) < 1e-8:
            x_axis = bbox.R[:, short_axes[1]]
            x_axis = x_axis - z_axis * np.dot(z_axis, x_axis)

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])
        cylinder.rotate(R, center=(0, 0, 0))
        cylinder.translate(bbox.center)
        cylinder.compute_vertex_normals()

        proxy_records.append(replace(record, mesh=cylinder))

    return proxy_records


def make_capsule_proxy(robot: RobotModel, state: RobotState):
    return
