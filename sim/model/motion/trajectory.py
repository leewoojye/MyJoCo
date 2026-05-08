import numpy as np
import scipy
from sim.model.kinematics.fk import compute_fk, apply_fk
from sim.model.math3d.screw import unit_screw_axis, screw_hat, screw_vee
from sim.model.math3d.rotation import omega2rotation_matrix
from sim.model.math3d.transform import create_transform_matrix
from sim.model.robot.robot_model import RobotModel
from sim.model.robot.robot_state import RobotState
from sim.model.math3d.lie import Adjoint
from sim.model.kinematics.jacobian import (
    compute_position_jacobian,
    compute_geometric_jacobian,
)


# time-scaling s(t): [0,T]->[0,1]
# T: total duration, t: time variable
# s(0) = 0, s(Tf) = 1, s_dot(0) = 0, s_dot(T) = 0 가정
def cubic_time_scaling(T, t):
    scaled_t = (
        t / T
    )  # 서로 다른 입력 T에 대해 동일한 time-scaling 계수를 사용하기 위해 정규화된 시간을 사용함. 다시 말해, t로 표현된 s(t)가 0과 1사이 값을 반환하게 하기 위해 t를 정규화함
    a0 = 0
    a1 = 1
    a2 = 3 / T**2
    a3 = -2 / T**3
    s_t = a2 * scaled_t**2 + a3 * scaled_t**3
    s_dot_t = 2 * a2 * scaled_t + 3 * a3 * scaled_t**2
    s_ddot_t = 2 * a2 + 6 * a3 * scaled_t
    return s_t, s_dot_t, s_ddot_t


# s(0) = 0, s(Tf) = 1, s_dot(0) = 0, s_dot(T) = 0, s_ddot(0) = 0, s_ddot(T) = 0
def quintic_time_scaling(T, t):
    scaled_t = t / T
    a0 = 0
    a1 = 0
    a2 = 0
    a3 = 10 / T**3
    a4 = -15 / T**4
    a5 = 6 / T**5
    s_t = 10 * scaled_t**3 - 15 * scaled_t**4 + 6 * scaled_t**5
    s_dot_t = (30 * scaled_t**2 - 60 * scaled_t**3 + 30 * scaled_t**4) / T
    s_ddot_t = (60 * scaled_t - 180 * scaled_t**2 + 120 * scaled_t**3) / T**2
    return s_t, s_dot_t, s_ddot_t


def interpolate_position(p_start, p_end, T, t):
    s_t, s_dot_t, s_ddot_t = quintic_time_scaling(T, t)
    p_current = p_start + s_t * (p_end - p_start)
    return p_current


def interpolate_pose(p_start, p_end, T, t):
    return


# def trajectory_generator(p_start, p_end, T, dt):
#     num_point = np.floor(T / dt) + 1
#     for i in range(num_point):
#         s_t, s_dot_t, s_ddot_t = quintic_time_scaling(T, dt*i)

#     return
