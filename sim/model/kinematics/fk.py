import numpy as np
from sim.model.math3d.rotation import create_skew
from sim.model.math3d.screw import screw_hat, unit_screw_axis
from scipy.linalg import expm
from sim.model.robot.body import MuJoCoBodyNode
from sim.model.robot.joint import JointType
from sim.model.robot.robot_model import RobotGeometries
from sim.model.robot.state import RobotState

# PoE: product of exponential
# 관절각 집합: state.qpos
# def compute_fk_all_links(robot: RobotGeometries, state: RobotState, M):
#   T = np.eye(4)

#   # end-effector까지의 관절만 FK계산에 이용함
#   # 말단에서 body 방향으로 관절을 순회하고 역방향 리스트를 만듦
#   target = robot.body_node_for("link6")

#   chain_nodes = []
#   node = target

#   while node is not None:
#       chain_nodes.append(node)
#       node = node.parent

#   chain_nodes.reverse()

#   for node in robot.root_body.iter_nodes():
#     all_nodes[node.name] = node.world_transform.copy()

#   chain_joints = []
#   for node in chain_nodes:
#       for joint in node.joints:
#           chain_joints.append((node, joint))

#   chain_joints.sort(key=lambda item: item[1]["qpos_addr"])

#   link_poses = {}

#   for node, joint in chain_joints:
#     current_link = node.name
#     q = joint["pos"]
#     w = joint["axis"]
#     # joint["type"] 기반 동작하도록 수정
#     S = unit_screw_axis(q,w,0,joint["type"])
#     S_exp = screw_hat(S) * state.get(joint["name"])
#     # S_exp = screw_hat(S) * state.qpos[joint["qpos_addr"]]
#     T = T @ expm(S_exp)
#     link_poses[current_link] = T @ M[current_link]

#   # end-effector의 home figuration(관절각이 모두 0인 상태)을 마지막에 곱함
#   # T = T @ M
#   return link_poses


# root body에서 시작해 재귀적으로 transform matrix 계산
def compute_fk_all_links_recursive(
    node: MuJoCoBodyNode,
    state: RobotState,
    M,
    cum_T,
    all_link_poses,
    include_node_joints=True,
):
    if include_node_joints:
        for joint in node.joints:
            joint_transform = M[node.name]
            q = joint_transform[:3, :3] @ joint["pos"] + joint_transform[:3, 3]
            w = joint_transform[:3, :3] @ joint["axis"]
            S = unit_screw_axis(q, w, 0, joint["type"])
            theta = state.get(joint["name"])
            # PoE에서 세타는 기준 자세로부터 joint displacement (각변위)
            # home configuration M 성분이 0이 아니면 목표 관절값에서 M만큼 빼줌
            if "_qpos" in M:  # _qpos는 home configuration의 qpos
                theta -= M["_qpos"][joint["qpos_addr"]]
            S_exp = screw_hat(S) * theta
            cum_T = cum_T @ expm(S_exp)

    all_link_poses[node.name] = cum_T @ M[node.name]

    for child in node.children:
        compute_fk_all_links_recursive(child, state, M, cum_T, all_link_poses)
    return


# 주어진 state를 입력으로 end-effector의 위치를 계산만하여 반환
def compute_fk(robot: RobotGeometries, state: RobotState, M):
    # all_link_poses를 전역변수로 선언하면 compute_fk를 동시에 호출할 때 문제가 됨
    all_link_poses = {}
    T = np.eye(4)
    root_node = robot.body_node_for("lift_link") or robot.root_body
    compute_fk_all_links_recursive(
        root_node, state, M, T, all_link_poses, include_node_joints=False
    )
    return all_link_poses


# FK를 재귀적으로 계산한 후, 노드 및 mesh의 transform matrix가 업데이트된 RobotGeometries 인스턴스 반환
def apply_fk(robot: RobotGeometries, state: RobotState, M):
    all_link_poses = {}
    T = np.eye(4)
    root_node = robot.body_node_for("lift_link") or robot.root_body
    compute_fk_all_links_recursive(
        root_node, state, M, T, all_link_poses, include_node_joints=False
    )

    for node in root_node.iter_nodes():
        old_T = node.world_transform
        new_T = all_link_poses[node.name]

        # (참고) 역행렬: body/frame 좌표계를 바꾸는 효과
        # (참고) cancellation
        # new_T = delta @ old_T, delta는 space frame 기준 보정 행렬
        delta = new_T @ np.linalg.inv(old_T)

        for mesh in node.geometries:
            mesh.transform(delta)

        node.world_transform = new_T

    return robot
