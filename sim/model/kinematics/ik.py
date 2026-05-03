import stat

import numpy as np
from sim.model.kinematics.fk import compute_fk, apply_fk
from sim.model.math3d.screw import unit_screw_axis, screw_hat, screw_vee
from sim.model.math3d.rotation import omega2rotation_matrix
from sim.model.math3d.transform import create_transform_matrix
from sim.model.robot.robot_model import RobotGeometries
from sim.model.robot.state import RobotState
from sim.model.kinematics.jacobian import compute_position_jacobian


# 감쇠 역행렬
# 아무리 특이점에 가까워지더라도 람다 값 때문에 분모가 0이 되지 않아 관절 속도가 안전하게 제한됨
def damped_pseudoinverse(J, lambda_=1e-3):  # 예약어 충돌방지를 위한 언더바
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + lambda_**2 * np.eye(m))


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

    V_b_mat = np.log(T_error)

    # 허수부 오차 제거
    V_b_mat = np.real(V_b_mat)

    w_x = V_b_mat[2, 1]
    w_y = V_b_mat[0, 2]
    w_z = V_b_mat[1, 0]
    v_x = V_b_mat[0, 3]
    v_y = V_b_mat[1, 3]
    v_z = V_b_mat[2, 3]

    V_b_vec = np.array([w_x, w_y, w_z, v_x, v_y, v_z])

    return V_b_mat, V_b_vec


def solve_newton_raphson_coordinate(
    robot: RobotGeometries, state: RobotState, target, M
):
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
        e = target - link_poses["link6"][:3, 3]  # (추후 수정)
        J, joints = compute_position_jacobian(robot, link_poses)

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

        # e가 벡터일 경우 불린 배열을 반환하여 조건문이 애매해짐
        # if np.abs(e) <= 1e-4:
        if np.linalg.norm(e) <= 1e-4:
            break
        if iter > 10:
            break

        iter += 1

    # ik target panel 변화 단위로 각변위 제한
    max_theta_step = 0.1
    delta_theta = theta - theta_prev
    delta_theta = np.clip(delta_theta, -max_theta_step, max_theta_step)
    state.qpos = theta_prev + delta_theta

    return state


def solve_newton_raphson_geometric(
    robot: RobotGeometries, state: RobotState, target, M
):
    # theta = M
    # e = target - compute_fk(robot, state, target, theta)
    # T_sd =
    # skew_Vb = np.log(np.linalg.inv())
    # delta = theta + np.linalg.pinv(compute_position_jacobian(robot)) @
    return


def solve_position_ik(robot: RobotGeometries, state: RobotState, target, M):
    new_state = solve_newton_raphson_coordinate(robot, state, target, M)
    return new_state


# position + pose (6D twist) 기반 자코비안 행렬 활용
def solve_pose_ik_recursive():
    return


# position + pose (6D twist) 기반 자코비안 행렬 활용
# 관절각 계산만
def solve_pose_ik(robot: RobotGeometries, state: RobotState, target, M):
    return


# 계산된 관절각 적용
def apply_ik(robot: RobotGeometries, state: RobotState, target, M):
    new_state = solve_position_ik(robot, state, target, M)
    robot = apply_fk(robot, new_state, M)

    return robot
