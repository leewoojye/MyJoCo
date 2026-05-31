import numpy as np
from regex import R
from scipy.spatial.transform import Rotation, Slerp

from sim.model.math3d.transform import create_transform_matrix


# time-scaling s(t): [0,T]->[0,1]
# T: total duration, t: time variable
# s(0) = 0, s(Tf) = 1, s_dot(0) = 0, s_dot(T) = 0 가정
def cubic_time_scaling(T, t):
    scaled_t = (
        t / T
    )  # 서로 다른 입력 T에 대해 동일한 time-scaling 계수를 사용하기 위해 정규화된 시간을 사용함. 다시 말해, t로 표현된 s(t)가 0과 1사이 값을 반환하게 하기 위해 t를 정규화함
    s_t = 3 * scaled_t**2 - 2 * scaled_t**3
    s_dot_t = (6 * scaled_t - 6 * scaled_t**2) / T
    s_ddot_t = (6 - 12 * scaled_t) / T**2

    return s_t, s_dot_t, s_ddot_t


# s(0) = 0, s(Tf) = 1, s_dot(0) = 0, s_dot(T) = 0, s_ddot(0) = 0, s_ddot(T) = 0
def quintic_time_scaling(T, t):
    scaled_t = t / T
    # a0 = 0
    # a1 = 0
    # a2 = 0
    # a3 = 10 / T**3
    # a4 = -15 / T**4
    # a5 = 6 / T**5

    s_t = 10 * scaled_t**3 - 15 * scaled_t**4 + 6 * scaled_t**5
    s_dot_t = (30 * scaled_t**2 - 60 * scaled_t**3 + 30 * scaled_t**4) / T
    s_ddot_t = (60 * scaled_t - 180 * scaled_t**2 + 120 * scaled_t**3) / T**2

    return s_t, s_dot_t, s_ddot_t


def interpolate_position(p_start, p_end, T, t, return_acc=False):
    s_t, _, s_ddot_t = cubic_time_scaling(T, t)
    # s_t, s_dot_t, s_ddot_t = quintic_time_scaling(T, t)
    p_t = p_start + s_t * (p_end - p_start)
    if return_acc:
        return p_t, s_ddot_t
    return p_t


def interpolate_position_quintic(p_start, p_end, T, t):
    s_t, s_dot_t, s_ddot_t = quintic_time_scaling(T, t)
    p_t = p_start + s_t * (p_end - p_start)
    p_dot_t = s_dot_t * (p_end - p_start)
    p_dotdot_t = s_ddot_t * (p_end - p_start)

    return p_t, p_dot_t, p_dotdot_t


# 단순 선형 위치 보간
def interpolate_position_simple(p_start, p_end, T, t):
    scaled_t = t / T
    p_t = p_start + scaled_t * (p_end - p_start)

    return p_t


# 회전 보간
# R(t) = R0 @ exp( s(t) * log(R0.T @ R1) )
def interpolate_rotation(R_start, R_end, T, t):
    s_t, s_dot_t, s_dotdot_t = cubic_time_scaling(T, t)
    R_rel = R_start.T @ R_end

    rotvec_rel = Rotation.from_matrix(R_rel).as_rotvec()
    R_inc = Rotation.from_rotvec(s_t * rotvec_rel).as_matrix()

    w_t = R(t) @ (s_dot_t * rotvec_rel)
    w_dot_t = R(t) @ (s_dotdot_t * rotvec_rel)

    return R_start @ R_inc, w_t, w_dot_t


def interpolate_rotation_slerp(R_start, R_end, T, t):
    s_t, s_dot_t, s_dotdot_t = cubic_time_scaling(T, t)
    s_t = np.clip(s_t, 0.0, 1.0)

    key_rots = Rotation.from_matrix([R_start, R_end])
    slerp = Slerp([0.0, 1.0], key_rots)
    R_t = slerp([s_t]).as_matrix()[0]

    rotvec_rel = Rotation.from_matrix(R_start.T @ R_end).as_rotvec()
    w_t = R_t @ (s_dot_t * rotvec_rel)
    w_dot_t = R_t @ (s_dotdot_t * rotvec_rel)

    return R_t, w_t, w_dot_t


# 자세 보간
def interpolate_pose(T_start, T_end, T, t):
    p_start = T_start[:3, 3]
    p_end = T_end[:3, 3]
    R_start = T_start[:3, :3]
    R_end = T_end[:3, :3]
    p_t, p_dot_t, p_dotdot_t = interpolate_position_quintic(p_start, p_end, T, t)
    R_t, w_t, w_dot_t = interpolate_rotation_slerp(R_start, R_end, T, t)
    T_t = create_transform_matrix(R_t, p_t)

    twist_des = np.r_[w_t, p_dot_t]
    twistdot_des = np.r_[w_dot_t, p_dotdot_t]

    return T_t, twist_des, twistdot_des


# def trajectory_generator(p_start, p_end, T, dt):
#     num_point = np.floor(T / dt) + 1
#     for i in range(num_point):
#         s_t, s_dot_t, s_ddot_t = quintic_time_scaling(T, dt*i)

#     return
