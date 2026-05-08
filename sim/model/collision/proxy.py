# 로봇팔과 캔의 mesh/polygon을 근사할 sphere(몸통이나 손가락), capsule(arm) 생성
# 이는 collision check시 두 강체 사이의 거리를 계산하는데 사용
import numpy as np
import open3d as o3d
from open3d import geometry, utility, visualization


from sim.model.robot.robot_model import RobotGeometries
from sim.model.robot.state import RobotState


# open3d 기본 mesh를 cylinder mesh로 변환
def make_cylinder_proxy(robot: RobotGeometries, state: RobotState, mesh):
    bbox = mesh.get_oriented_bounding_box()
    axises = np.asarray(bbox.extent)
    return


def make_capsule_proxy(robot: RobotGeometries, state: RobotState):
    return
