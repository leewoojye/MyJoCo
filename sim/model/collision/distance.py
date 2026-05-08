# sphere - sphere
# sphere - capsule
# capsule - capsule
# box - cylinder


from sim.model.robot.geometry import GeomRecord


def proxy_distance(r1: GeomRecord, r2: GeomRecord):
    pcd1 = r1.mesh.sample_points_uniformly(number_of_points=1000)
    pcd2 = r2.mesh.sample_points_uniformly(number_of_points=1000)

    d12 = pcd1.compute_point_cloud_distance(pcd2)
    d21 = pcd2.compute_point_cloud_distance(pcd1)

    return min(min(d12), min(d21))
