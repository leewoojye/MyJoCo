# sphere - sphere
# sphere - capsule
# capsule - capsule
# box - cylinder


import numpy as np
import open3d as o3d
from sim.model.collision.proxy import make_cylinder_proxy
from sim.model.robot.geometry import GeomRecord


def use_cylinder_proxy(record: GeomRecord):
    return record.body_name not in {"world", "base_table"}


def cylinder_proxy_mesh(record: GeomRecord):
    bbox = record.mesh.get_oriented_bounding_box()
    extent = np.asarray(bbox.extent, dtype=float)
    long_axis = int(np.argmax(extent))
    short_axes = [i for i in range(3) if i != long_axis]

    height = float(extent[long_axis])
    radius = 0.5 * float(max(extent[short_axes[0]], extent[short_axes[1]]))
    if height <= 0.0 or radius <= 0.0:
        return record.mesh

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=height,
        resolution=16,
    )

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

    return cylinder


def distance_mesh(record: GeomRecord, proxy_cache=None):
    if not use_cylinder_proxy(record):
        return record.mesh

    if proxy_cache is None:
        return make_cylinder_proxy(None, None, [record])[0].mesh

    if record.geom_id not in proxy_cache:
        proxy_cache[record.geom_id] = make_cylinder_proxy(None, None, [record])[0].mesh

    return proxy_cache[record.geom_id]


# 두 프록시 사이 최소 거리와 최소 거리를 만드는 두 위치 반환
def proxy_distance(r1: GeomRecord, r2: GeomRecord, proxy_cache=None):
    mesh1 = distance_mesh(r1, proxy_cache)
    mesh2 = distance_mesh(r2, proxy_cache)

    # open3d pointcloud 클래스를 이용해 프록시 위 점들을 샘플링하고, 거리를 계산
    pointcloud1 = mesh1.sample_points_uniformly(number_of_points=50)  # pointcloud 객체
    pointcloud2 = mesh2.sample_points_uniformly(number_of_points=50)

    points1 = np.asarray(pointcloud1.points)  # pointcloud를 이루는 점들의 위치 배열
    points2 = np.asarray(pointcloud2.points)

    distance_vectors = points1[:, None, :] - points2[None, :, :]
    squared_distances = np.sum(distance_vectors * distance_vectors, axis=2)
    i, j = np.unravel_index(np.argmin(squared_distances), squared_distances.shape)

    return points1[i], points2[j], float(np.sqrt(squared_distances[i, j]))

    # distance_12 = pointcloud1.compute_point_cloud_distance(pointcloud2)
    # distance_21 = pointcloud2.compute_point_cloud_distance(pointcloud1)

    # return min(min(distance_12), min(distance_21))
