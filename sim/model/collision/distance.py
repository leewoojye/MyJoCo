# sphere - sphere
# sphere - capsule
# capsule - capsule
# box - capsule


import numpy as np
import open3d as o3d
import fcl
from scipy.spatial.transform import Rotation
from sim.model.collision.proxy import (
    BoxProxy,
    make_box_proxy,
    CapsuleProxy,
    make_capsule_proxy,
)
from sim.model.robot.geometry import GeomRecord


# capsule proxy를 사용하지 않는 개체는 mesh 그대로 반환하도록 함
def use_capsule_proxy(record: GeomRecord):
    return record.body_name not in {"world", "base_table"}


def get_proxy(record: GeomRecord, proxy_cache=None):
    if proxy_cache is not None and record.geom_id in proxy_cache:
        return proxy_cache[record.geom_id]

    if record.body_name == "base_table":
        proxy = make_box_proxy(record)
    elif use_capsule_proxy(record):
        proxy = make_capsule_proxy(record)
    else:
        proxy = record.mesh

    if proxy_cache is not None:
        proxy_cache[record.geom_id] = proxy

    return proxy


# fcl 인스턴스의 z축을 proxy 축으로 변환하는 회전행렬을 생성
def make_align_rotation_matrix(axis):
    z = np.array([0.0, 0.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    return Rotation.align_vectors([axis], [z])[0].as_matrix()


def capsule_to_fcl(capsule: CapsuleProxy):
    axis = capsule.p1 - capsule.p0
    length = np.linalg.norm(axis)
    center = 0.5 * (capsule.p0 + capsule.p1)

    if length < 1e-8:  # fallback
        geom = fcl.Sphere(capsule.radius)
        tf = fcl.Transform(np.eye(3), center)
        return fcl.CollisionObject(geom, tf)

    R = make_align_rotation_matrix(axis)
    geom = fcl.Capsule(capsule.radius, length)
    tf = fcl.Transform(R, center)

    return fcl.CollisionObject(geom, tf)


def box_to_fcl(box: BoxProxy):
    size = 2.0 * box.half_extents

    geom = fcl.Box(size[0], size[1], size[2])
    tf = fcl.Transform(box.axes, box.center)

    return fcl.CollisionObject(geom, tf)


def distance_capsule_capsule(capsule_a, capsule_b):
    obj_a = capsule_to_fcl(capsule_a)
    obj_b = capsule_to_fcl(capsule_b)
    request = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True)
    result = fcl.DistanceResult()
    distance = fcl.distance(
        obj_a,
        obj_b,
        request,
        result,
    )
    p_a = result.nearest_points[0]
    p_b = result.nearest_points[1]
    normal = None

    if distance > 0:
        normal = p_b - p_a
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-8:
            normal = normal / normal_norm
    else:  # 접촉 상태가 관통으로 판단되면 fcl.CollisionResult()로 normal vector를 받아옴
        collision_request = fcl.CollisionRequest(enable_contact=True, num_max_contacts=1)
        collision_result = fcl.CollisionResult()
        fcl.collide(obj_a, obj_b, collision_request, collision_result)

        if collision_result.contacts:
            contact = collision_result.contacts[0]
            normal = np.asarray(contact.normal, dtype=float)
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-8:
                normal = normal / normal_norm
                depth = float(contact.penetration_depth)
                point = np.asarray(contact.pos, dtype=float)
                p_a = point - 0.5 * depth * normal
                p_b = point + 0.5 * depth * normal
                distance = -depth

    # if normal is None or np.linalg.norm(normal) <= 1e-8:
    #     normal = 0.5 * (capsule_b.p0 + capsule_b.p1) - 0.5 * (capsule_a.p0 + capsule_a.p1)
    #     normal_norm = np.linalg.norm(normal)
    #     normal = normal / normal_norm if normal_norm > 1e-8 else None

    return p_a, p_b, float(distance), normal


def distance_capsule_box(capsule, box):
    request = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True)
    result = fcl.DistanceResult()
    distance = fcl.distance(
        capsule_to_fcl(capsule),
        box_to_fcl(box),
        request,
        result,
    )

    return result.nearest_points[0], result.nearest_points[1], float(distance)


# 두 프록시 사이 최소 거리와 최소 거리를 만드는 두 위치 반환
# 부호 없는 거리 측도도 괜찮은가?->관통 깊이 계산을 위해 상대 거리를 반환해야 함
def proxy_distance(record1: GeomRecord, record2: GeomRecord, proxy_cache=None):
    proxy1 = get_proxy(record1, proxy_cache)
    proxy2 = get_proxy(record2, proxy_cache)

    if isinstance(proxy1, CapsuleProxy) and isinstance(proxy2, CapsuleProxy):
        return distance_capsule_capsule(proxy1, proxy2)

    if isinstance(proxy1, CapsuleProxy) and isinstance(proxy2, BoxProxy):
        return distance_capsule_box(proxy1, proxy2)

    if isinstance(proxy1, BoxProxy) and isinstance(proxy2, CapsuleProxy):
        point_b, point_a, distance = distance_capsule_box(proxy2, proxy1)
        return point_a, point_b, distance

    return None
