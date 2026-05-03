import numpy as np
from sim.model.math3d.rotation import create_skew
from sim.model.math3d.screw import unit_screw_axis, screw_hat
from scipy.linalg import expm
from sim.model.robot.body import MuJoCoBodyNode
from sim.model.robot.joint import JointType
from sim.model.robot.robot_model import RobotGeometries
from sim.model.robot.state import RobotState


def append_joints(all_joints, node: MuJoCoBodyNode):
    # 부모 노드는 1개로 가정
    if node.parent is not None:
        append_joints(all_joints, node.parent)

    # 여러 관절을 갖는 링크를 고려해 매스텝마다 joint 정렬해줌
    for joint in sorted(node.joints, key=lambda joint: joint["qpos_addr"]):
        all_joints.append(joint)

    return


# omy의 e.e를 position jacobian으로 다루기
# IK 과정에서 중간 계산 결과를 활용할 수 있도록 link_poses 인자를 받게 함
def compute_position_jacobian(robot: RobotGeometries, link_poses=None):
    # end-effector 하드코딩, e.e 시작으로 역으로 관절 순회
    target = robot.body_node_for("link6")
    target_transform = (
        link_poses[target.name] if link_poses is not None else target.world_transform
    )
    pos_ee = target_transform[:3, 3]

    all_joints = []
    append_joints(all_joints, target)

    # 자코비안 행렬은 e.e에서 root까지 단일 경로만 고려
    all_S = []
    for joint in all_joints:  # 회전관절 가정 (추후 수정)
        q = joint["world_pos"]
        w = joint["world_axis"]
        S = unit_screw_axis(q, w, 0, joint["type"])
        v = S[3:, 0]
        J_i = np.cross(w, pos_ee) + v  # np.cross(omega, p_ee - q)
        all_S.append(J_i.reshape(3, 1))  # 회전 성분을 제외한 열벡터 append

    # 자코비안 열 인덱스와 joint id(qpos_addr)이 일치하지 않을 수 있어 관절 객체 리스트로 같이 반환
    return np.concatenate(all_S, axis=1), all_joints


# end-effector(ex. omy의 link6)에 대한 자코비안 행렬 생성
def compute_geometric_jacobian(robot: RobotGeometries):
    # end-effector 하드코딩, e.e 시작으로 역으로 관절 순회
    target = robot.body_node_for("link6")
    # parent = target.parent.copy()
    # J = np.zeros(6, target.joints["joint_id"])  # 6xn (n: e.e까지 관절개수) 행렬 초기화

    all_joints = []
    append_joints(all_joints, target)
    # all_joints.sort(
    #     key=lambda joint: joint["qpos_addr"]
    # )  # joint id 기준으로 오름차순 정렬 (추후 수정)

    # 자코비안 행렬은 e.e에서 root까지 단일 경로만 고려
    all_S = []
    for joint in all_joints:  # 회전관절 가정 (추후 수정)
        J_i = unit_screw_axis(joint["world_pos"], joint["world_axis"], 0, joint["type"])
        all_S.append(J_i)  # 열벡터 append

    # 자코비안 열 인덱스와 joint id(qpos_addr)이 일치하지 않을 수 있어 관절 객체 리스트로 같이 반환
    return np.concatenate(all_S, axis=1), all_joints


def finite_difference_position_jacobian():
    return


# 특이성 평가
def manipulability():
    return


def condition_number():
    return
