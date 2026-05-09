# sphere - sphere
# sphere - capsule
# capsule - capsule
# box - cylinder


import numpy as np
from sim.model.robot.geometry import GeomRecord


# 두 프록시 사이 최소 거리와 최소 거리를 만드는 두 위치 반환
def proxy_distance(r1: GeomRecord, r2: GeomRecord):
    # open3d pointcloud 클래스를 이용해 프록시 위 점들을 샘플링하고, 거리를 계산
    pointcloud1 = r1.mesh.sample_points_uniformly(number_of_points=1000)  # pointcloud 객체
    pointcloud2 = r2.mesh.sample_points_uniformly(number_of_points=1000)

    points1 = np.asarray(pointcloud1.points)  # pointcloud를 이루는 점들의 위치 배열
    points2 = np.asarray(pointcloud2.points)

    min_distance = float("inf") # 무한대로 초기화
    closest_p1 = None  # 최단 거리를 만드는 두 점의 위치
    closest_p2 = None

    # pointcloud 점들을 순회하며 최소 거리 탐색
    for p1 in points1:
        for p2 in points2:
            d = np.linalg.norm(p2 - p1)

            if d < min_distance:
                min_distance = d
                closest_p1 = p1
                closest_p2 = p2

    return closest_p1, closest_p2, min_distance

    # distance_12 = pointcloud1.compute_point_cloud_distance(pointcloud2)
    # distance_21 = pointcloud2.compute_point_cloud_distance(pointcloud1)

    # return min(min(distance_12), min(distance_21))
