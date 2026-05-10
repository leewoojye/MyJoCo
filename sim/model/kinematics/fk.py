import numpy as np
from sim.model.math3d.screw import screw_hat, unit_screw_axis
from scipy.linalg import expm
from sim.model.robot.body import BodyNode
from sim.model.robot.joint import JointType
from sim.model.robot.robot_model import RobotModel
from sim.model.robot.robot_state import RobotState


# PoE: product of exponential
# root body에서 시작해 재귀적으로 transform matrix 계산
def compute_fk_all_links_recursive(
    node: BodyNode,
    state: RobotState,
    home_pose,
    cum_T,
    all_link_poses,
    include_node_joints=True,
):
    if include_node_joints:  # 현재 노드에 달린 joint 변환을 FK에 반영할지 정하는 flag
        for joint in node.joints:
            joint_transform = home_pose[node.name]
            q = joint_transform[:3, :3] @ joint["pos"] + joint_transform[:3, 3]
            w = joint_transform[:3, :3] @ joint["axis"]
            S = unit_screw_axis(q, w, 0, joint["type"])
            theta = state.get(joint["name"])  # PoE에서 세타는 기준 자세로부터 joint displacement (각변위)

            # M(home configuration) 성분이 0이 아니면 목표 관절값에서 M만큼 빼줌
            if "_qpos" in home_pose:  # _qpos는 home configuration의 qpos
                theta -= home_pose["_qpos"][joint["qpos_addr"]]
                
            S_exp = screw_hat(S) * theta
            cum_T = cum_T @ expm(S_exp)

    all_link_poses[node.name] = cum_T @ home_pose[node.name]

    for child in node.children:
        compute_fk_all_links_recursive(child, state, home_pose, cum_T, all_link_poses)

    return


# 주어진 state를 입력으로 end-effector의 위치를 계산만하여 반환
def compute_fk(robot: RobotModel, state: RobotState, home_pose):
    # all_link_poses를 전역변수로 선언하면 compute_fk를 동시에 호출할 때 문제가 됨
    all_link_poses = {}
    T = np.eye(4)

    root_node = robot.body_node_for("lift_link") or robot.root_body
    compute_fk_all_links_recursive(root_node, state, home_pose, T, all_link_poses, include_node_joints=False)

    return all_link_poses


# FK를 재귀적으로 계산한 후, 노드 및 mesh의 transform matrix가 업데이트된 RobotGeometries 인스턴스 반환
def apply_fk(robot: RobotModel, state: RobotState, home_pose):
    all_link_poses = {}
    T = np.eye(4)

    root_node = robot.body_node_for("lift_link") or robot.root_body
    compute_fk_all_links_recursive(root_node, state, home_pose, T, all_link_poses, include_node_joints=False)

    for node in root_node.iter_nodes():
        old_T = node.world_transform
        new_T = all_link_poses[node.name]

        # (참고) 역행렬: body/frame 좌표계를 바꾸는 효과
        # (참고) cancellation
        # new_T = delta @ old_T, delta는 space frame 기준 보정 행렬
        delta = new_T @ np.linalg.inv(old_T)

        for record in node.all_records():
            record.mesh.transform(delta)

        node.world_transform = new_T

    return robot
