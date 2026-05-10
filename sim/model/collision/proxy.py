# 로봇팔과 캔의 mesh/polygon을 근사할 sphere(몸통이나 손가락), capsule(arm) 생성
# 이는 collision check시 두 강체 사이의 거리를 계산하는데 사용
from dataclasses import dataclass
import numpy as np
import open3d as o3d


from sim.model.robot.geometry import GeomRecord
from sim.model.robot.robot_model import RobotModel


# primitive cylinder 클래스
@dataclass
class CylinderProxy:
    center: np.ndarray  # 실린더 global center
    axis: np.ndarray
    radius: float
    half_height: float  # 실린더 중심 정보와 함께 사용하기 좋음
    record: GeomRecord


@dataclass
class BoxProxy:
    center: np.ndarray
    axes: np.ndarray  # 각 열이 box local axis
    half_extents: np.ndarray


@dataclass
class CapsuleProxy:
    p0: np.ndarray
    p1: np.ndarray
    radius: float
    record: GeomRecord


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
def make_cylinder_proxy(record: GeomRecord):
    # 깊은 복사 옵션1: copy.deepcopy(), 옵션2: dataclasses replace
    # proxy_records = copy.deepcopy(collision_records)

    bbox = record.mesh.get_oriented_bounding_box()  # open3d 내장함수로 mesh의 바운딩박스를 얻음
    extent_list = np.asarray(bbox.extent, dtype=float)

    # 실린더 높이 옵션1: 링크의 길이를 실린더의 높이로 설정
    # height = get_link_length(robot, record.body_name, record)

    # 실린더 높이 옵션2: bbox의 가장 긴 축을 링크의 길이로 가정
    long_axis = int(np.argmax(extent_list))
    short_axes = [i for i in range(3) if i != long_axis]

    cylinder_axis = np.asarray(bbox.R[:, long_axis], dtype=float)  # bbox의 가장 긴 축을 실린더의 축으로 설정
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)

    # axis_max_length = float(extent_list[long_axis])
    # radius = max(extent_list[short_axes[0]], extent_list[short_axes[1]]) / 2
    # if axis_max_length <= 0.0 or radius <= 0.0:
    #     proxy_records.append(replace(record, mesh=mesh))
    #     continue

    # cylinder = o3d.geometry.TriangleMesh.create_cylinder(  # 문제: open3d의 create_cylinder()는 여전히 mesh 기반임 하... (cylinder primitive 객체와는 다름)
    #     radius=radius,
    #     height=axis_max_length,
    #     resolution=32,
    # )  # open3d create 함수는 기본적으로 원점 기준 객체를 반환

    return CylinderProxy(
        record=record,
        center=np.asarray(bbox.center, dtype=float),
        axis=cylinder_axis,
        radius=0.5 * max(extent_list[short_axes[0]], extent_list[short_axes[1]]),
        half_height=0.5 * extent_list[long_axis],
    )


def make_box_proxy(record: GeomRecord):
    bbox = record.mesh.get_oriented_bounding_box()

    return BoxProxy(
        center=np.asarray(bbox.center, dtype=float),
        axes=np.asarray(bbox.R, dtype=float),
        half_extents=0.5 * np.asarray(bbox.extent, dtype=float),
    )


def make_capsule_proxy(record: GeomRecord):
    bbox = record.mesh.get_oriented_bounding_box()
    extent_list = np.asarray(bbox.extent, dtype=float)
    long_axis = int(np.argmax(extent_list))
    short_axes = [i for i in range(3) if i != long_axis]

    axis = np.asarray(bbox.R[:, long_axis], dtype=float)
    axis = axis / np.linalg.norm(axis)
    radius = 0.5 * max(extent_list[short_axes[0]], extent_list[short_axes[1]])
    half_height = max(0.0, 0.5 * extent_list[long_axis] - radius)

    return CapsuleProxy(
        record=record,
        p0=np.asarray(bbox.center, dtype=float) - axis * half_height,
        p1=np.asarray(bbox.center, dtype=float) + axis * half_height,
        radius=radius,
    )
