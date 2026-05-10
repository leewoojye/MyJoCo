import stat

import numpy as np
import scipy
from sim.model.kinematics.fk import compute_fk
from sim.model.robot.robot_model import RobotModel
from sim.model.robot.robot_state import RobotState
from sim.model.math3d.lie import Adjoint
from sim.model.kinematics.jacobian import (
    compute_position_jacobian,
    compute_geometric_jacobian,
)


# 감쇠 역행렬
# 아무리 특이점에 가까워지더라도 람다 값 때문에 분모가 0이 되지 않아 관절 속도가 안전하게 제한됨
def damped_pseudoinverse(J, lambda_=1e-3):  # 예약어 충돌방지를 위한 언더바
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + lambda_**2 * np.eye(m))


# 관절각 제약을 고려해 클리핑
def clamp_joint_ranges(qpos, joints):
    clipped_qpos = qpos.copy()

    for joint in joints:
        if not joint.get("limited", False):
            continue

        qpos_addr = joint["qpos_addr"]
        lower, upper = joint["range"]
        clipped_qpos[qpos_addr] = np.clip(clipped_qpos[qpos_addr], lower, upper)

    return clipped_qpos


# 현재 e.e와 목표 간 twist error 계산
def calculate_twist_error(T_sb, T_sd):
    # SE(3) 변환 행렬의 기하학적 특성을 활용한 역행렬 계산
    R = T_sb[0:3, 0:3]
    p = T_sb[0:3, 3]

    T_sb_inv = np.eye(4)
    T_sb_inv[0:3, 0:3] = R.T
    T_sb_inv[0:3, 3] = -np.dot(R.T, p)

    T_error = T_sb_inv @ T_sd
    # T_error = np.dot(T_sb_inv, T_sd)

    V_b_mat = scipy.linalg.logm(T_error)
    # V_b_mat = np.log(T_error) # np.log는 matrix logarithm이 아니라 element-wise log

    # 허수부 오차 제거
    V_b_mat = np.real(V_b_mat)

    w_x = V_b_mat[2, 1]
    w_y = V_b_mat[0, 2]
    w_z = V_b_mat[1, 0]
    v_x = V_b_mat[0, 3]
    v_y = V_b_mat[1, 3]
    v_z = V_b_mat[2, 3]

    V_b = np.array([w_x, w_y, w_z, v_x, v_y, v_z])

    # V_b hat 연산 결과(행렬)와 벡터 모두 반환
    return V_b_mat, V_b


def solve_newton_raphson_coordinate(robot: RobotModel, state: RobotState, target_pos, M, target_body="arm_r_link7"):
    # M: home configuration, 관절각이 0일 때 모든 링크의 T 집합
    # home_qpos = np.zeros(
    #     len(state.qpos),
    # )
    # theta = home_qpos

    theta_prev = state.qpos.copy()
    theta = state.qpos.copy()  # 직전 IK의 해로 세타를 초기화
    iter = 0

    while True:
        state.qpos = theta.copy()
        link_poses = compute_fk(robot, state, M)
        e = target_pos - link_poses[target_body][:3, 3]  # (추후 수정)
        J, joints = compute_position_jacobian(robot, link_poses, target_body)

        rows, cols = J.shape
        if rows == cols:  # J is Full rank and square
            dq = np.dot(np.linalg.pinv(J), e)
        elif (
            rows > cols
        ):  # closest in the 2-norm sense, 원하는 task velocity를 모두 만들 수 없으니 가장 가까운 해를 찾음
            dq = np.dot(np.linalg.pinv(J.T @ J) @ J.T, e)
        else:  # smallest 2-norm among all solutions, 목표 velocity가 여러 개 있을 수 있으니 이 중 가장 작은 해를 고름
            dq = np.dot(J.T @ np.linalg.pinv(J @ J.T), e)

        # newton-raphson iteration 내에서 각변위 제한
        # max_dq = 0.03  # 단위: rad
        # dq = np.clip(dq, -max_dq, max_dq)

        # 계산된 관절 부위만 갱신함(ex. e.e와 root 사이의 관절각만 업데이트)
        for joint, dq_i in zip(joints, dq):
            theta[joint["qpos_addr"]] += dq_i

        theta = clamp_joint_ranges(theta, joints)

        # e가 벡터일 경우 불린 배열을 반환하여 조건문이 애매해짐
        # if np.abs(e) <= 1e-4:
        if np.linalg.norm(e) <= 1e-4:
            break
        if iter > 20:
            break

        iter += 1

    # 각변위 제한: 사용자 지정 제약 + xml 명세 기반 제약
    # max_theta_step = 0.1
    # delta_theta = theta - theta_prev
    # delta_theta = np.clip(delta_theta, -max_theta_step, max_theta_step)
    # state.qpos = clamp_joint_ranges(theta_prev + delta_theta, joints)

    # 각변위 제한 없는 버전
    state.qpos = clamp_joint_ranges(theta, joints)

    return state


def solve_newton_raphson_geometric(
    robot: RobotModel, state: RobotState, target_pos, rot, home_pose, target_body="arm_r_link7"
):  # target_body->end-effector 인스턴스 수정예정
    T_sb = robot.body_node_for(target_body).world_transform
    T_sd = T_sb.copy()
    T_sd[:3, 3] = target_pos
    # T_sd[:3, :3] = T_sd[:3, :3] @ rot  # 현재 회전에 RPY를 계속 곱하면 목표가 매 tick 도망감
    T_sd[:3, :3] = rot  # 기준 회전에 RPY offset을 적용한 절대 목표 회전

    theta_prev = state.qpos.copy()
    theta = state.qpos.copy()
    iter = 0

    while True:
        state.qpos = theta.copy()
        link_poses = compute_fk(robot, state, home_pose)
        T_sb = link_poses[target_body]
        twist_error_mat, twist_error = calculate_twist_error(T_sb, T_sd)

        # space frame 기준 자코비안 행렬 J
        J, joints = compute_geometric_jacobian(robot, link_poses, target_body)

        J_b = Adjoint(np.linalg.inv(T_sb)) @ J  # body frame로의 좌표계 변환을 위해 big adjoint 연산 수행
        rows, cols = J_b.shape

        if rows == cols:
            dq = np.linalg.pinv(J_b) @ twist_error
        elif rows > cols:
            dq = np.linalg.pinv(J_b.T @ J_b) @ J_b.T @ twist_error
        else:
            dq = J_b.T @ np.linalg.pinv(J_b @ J_b.T) @ twist_error

        for joint, dq_i in zip(joints, dq):
            theta[joint["qpos_addr"]] += dq_i

        theta = clamp_joint_ranges(theta, joints)

        if np.linalg.norm(twist_error) <= 1e-4:
            break
        if iter > 20:
            break

        iter += 1

    # 각변위 제한: 사용자 지정 제약 + xml 명세 기반 제약
    # max_theta_step = 0.1
    # delta_theta = theta - theta_prev
    # delta_theta = np.clip(delta_theta, -max_theta_step, max_theta_step)
    # state.qpos = clamp_joint_ranges(theta_prev + delta_theta, joints)

    # 각변위 제한 없는 버전
    state.qpos = clamp_joint_ranges(theta, joints)

    return state


def solve_position_ik(robot: RobotModel, state: RobotState, target_pos, home_pose, target_body="arm_r_link7"):
    new_state = solve_newton_raphson_coordinate(robot, state, target_pos, home_pose, target_body)

    return new_state


# position + pose (6D twist) 기반 자코비안 행렬 활용
# def solve_pose_ik_recursive():
#     return


# position + pose (6D twist) 기반 자코비안 행렬 활용
# target_pos(3,): input position
def solve_pose_ik(robot: RobotModel, state: RobotState, target_pos, rot, home_pose, target_body="arm_r_link7"):
    new_state = solve_newton_raphson_geometric(robot, state, target_pos, rot, home_pose, target_body)

    return new_state


# IK로 state의 qpos만 갱신함. FK/mesh 갱신은 호출부에서 확정된 state에 대해 수행.
def apply_ik(
    robot: RobotModel, state: RobotState, target_pos, home_pose, mode="position", target_body="arm_r_link7", rot=None
):
    if mode == "position":
        new_state = solve_position_ik(robot, state, target_pos, home_pose, target_body)
    elif mode == "pose":
        new_state = solve_pose_ik(robot, state, target_pos, rot, home_pose, target_body)

    return new_state
