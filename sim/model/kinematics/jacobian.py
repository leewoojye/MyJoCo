import numpy as np
from sim.model.math3d.rotation import create_skew
from sim.model.math3d.screw import unit_screw_axis, screw_hat
from scipy.linalg import expm
from sim.model.robot.body import BodyNode
from sim.model.robot.joint import JointType
from sim.model.robot.robot_model import RobotModel
from sim.model.robot.robot_state import RobotState


def append_joints(
    all_joints,
    node: BodyNode,
    include_nodes=False,
    root_body_name="lift_link",
):
    if node.name == root_body_name:
        return

    # 부모 노드는 1개로 가정
    if node.parent is not None:
        append_joints(
            all_joints,
            node.parent,
            include_nodes=include_nodes,
            root_body_name=root_body_name,
        )

    # 여러 관절을 갖는 링크를 고려해 매스텝마다 joint 정렬해줌
    for joint in sorted(node.joints, key=lambda joint: joint["qpos_addr"]):
        all_joints.append((node, joint) if include_nodes else joint)

    return


# omy의 e.e를 position jacobian으로 다루기
# IK 과정에서 중간 계산 결과를 활용할 수 있도록 link_poses 인자를 받게 함
def compute_position_jacobian(
    robot: RobotModel, link_poses=None, target_body="arm_r_link7"
):
    # end-effector 하드코딩, e.e 시작으로 역으로 관절 순회
    target = robot.body_node_for(target_body)
    target_transform = (
        link_poses[target.name] if link_poses is not None else target.world_transform
    )
    pos_ee = target_transform[:3, 3]

    all_joints = []
    # (node, joint) 중 하나만을 배열 원소로 삼을 것인지 결정하는 flag 변수 include_nodes
    append_joints(all_joints, target, include_nodes=True)

    # 자코비안 행렬은 e.e에서 root까지 단일 경로만 고려
    all_S = []
    joints = []
    for node, joint in all_joints:  # 회전관절 가정 (추후 수정)
        joint_transform = (
            # 외부에서 link_poses(space transform)를 넘겨받음, 이는 M을 요구하는 FK와 달리 jacobian은 '현재' 상태를 기준으로 열을 구성하기 때문임
            # 현재 자세의 link_poses가 필요한 이유는 로딩 시 저장된 joint["world_axis"], joint["world_pos"]는 home pose 기준이기 때문임, 로봇이 움직이면 joint 축 방향도 world 기준으로 바뀌는데, 그걸 반영하려면 현재 body transform이 필요함
            link_poses[node.name] if link_poses is not None else node.world_transform
        )

        # q_sb = R_sb @ q_b + p_sb  # 점 변환
        # w_sb = R_sb @ w_b  # 방향 변환
        q = (
            joint_transform[:3, :3] @ joint["pos"] + joint_transform[:3, 3]
        )  # joint_transform[:3, :3] : space-frame transformation(T_sb)
        w = joint_transform[:3, :3] @ joint["axis"]
        if joint["type"] == JointType.SLIDE:
            J_i = w
        else:
            S = unit_screw_axis(q, w, 0, joint["type"])
            v = S[3:, 0]
            J_i = np.cross(w, pos_ee) + v  # np.cross(omega, p_ee - q)
        all_S.append(J_i.reshape(3, 1))  # 회전 성분을 제외한 열벡터 append
        joints.append(joint)

    # 자코비안 열 인덱스와 joint id(qpos_addr)이 일치하지 않을 수 있어 관절 객체 리스트로 같이 반환
    # space frame 기준 position jacobian
    return np.concatenate(all_S, axis=1), joints


# end-effector(ex. omy의 link6)에 대한 자코비안 행렬 생성
def compute_geometric_jacobian(
    robot: RobotModel, link_poses=None, target_body="arm_r_link7"
):
    # end-effector 하드코딩, e.e 시작으로 역으로 관절 순회
    target = robot.body_node_for(target_body)
    # parent = target.parent.copy()
    # J = np.zeros(6, target.joints["joint_id"])  # 6xn (n: e.e까지 관절개수) 행렬 초기화

    all_joints = []
    append_joints(all_joints, target, include_nodes=True)
    # all_joints.sort(
    #     key=lambda joint: joint["qpos_addr"]
    # )  # joint id 기준으로 오름차순 정렬 (추후 수정)

    # 자코비안 행렬은 e.e에서 root까지 단일 경로만 고려
    all_S = []
    joints = []
    for node, joint in all_joints:  # 회전관절 가정 (추후 수정)
        joint_transform = (
            link_poses[node.name] if link_poses is not None else node.world_transform
        )

        q = joint_transform[:3, :3] @ joint["pos"] + joint_transform[:3, 3]
        w = joint_transform[:3, :3] @ joint["axis"]
        J_i = unit_screw_axis(q, w, 0, joint["type"])
        all_S.append(J_i)  # 열벡터 append
        joints.append(joint)

    # 자코비안 열 인덱스와 joint id(qpos_addr)이 일치하지 않을 수 있어 관절 객체 리스트로 같이 반환
    return np.concatenate(all_S, axis=1), joints


def finite_difference_position_jacobian():
    return


# 특이성 평가
def manipulability():
    return


def condition_number():
    return
