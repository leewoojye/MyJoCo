import numpy as np
from scipy.spatial.transform import Rotation, Slerp


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


def interpolate_position(p_start, p_end, T, t):
    s_t, _, _ = cubic_time_scaling(T, t)
    # s_t, s_dot_t, s_ddot_t = quintic_time_scaling(T, t)
    p_current = p_start + s_t * (p_end - p_start)

    return p_current


# 단순 선형 위치 보간
def interpolate_position_simple(p_start, p_end, T, t):
    scaled_t = t / T
    p_current = p_start + scaled_t * (p_end - p_start)

    return p_current


# 회전 보간
def interpolate_rotation(r_start, r_end, T, t):
    s_t, _, _ = cubic_time_scaling(T, t)
    s_t = np.clip(s_t, 0.0, 1.0)

    key_rots = Rotation.from_matrix([r_start, r_end])
    slerp = Slerp([0.0, 1.0], key_rots)

    return slerp([s_t]).as_matrix()[0]


# def trajectory_generator(p_start, p_end, T, dt):
#     num_point = np.floor(T / dt) + 1
#     for i in range(num_point):
#         s_t, s_dot_t, s_ddot_t = quintic_time_scaling(T, dt*i)

#     return
