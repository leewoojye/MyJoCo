# 로봇팔과 캔의 mesh/polygon을 근사할 sphere(몸통이나 손가락), capsule(arm) 생성
# 이는 collision check시 두 강체 사이의 거리를 계산하는데 사용
import numpy as np
import open3d as o3d
from open3d import geometry, utility, visualization


from sim.model.robot.body import BodyNode
from sim.model.robot.robot_model import RobotModel
from sim.model.robot.robot_state import RobotState


# open3d 기본 mesh를 cylinder mesh로 변환
# 로봇팔, 손가락 proxy
def make_cylinder_proxy(robot: RobotModel, state: RobotState, node: BodyNode):
    mesh = node.collision_records
    bbox = mesh.get_oriented_bounding_box()
    axis_length_list = np.asarray(bbox.extent)
    axis_max_length = max(
        axis_length_list[0], max(axis_length_list[1], axis_length_list[2])
    )
    radius = (
        sum(
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
    # cylinder_position=state.
    return


def make_capsule_proxy(robot: RobotModel, state: RobotState):
    return
