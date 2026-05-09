import numpy as np
from scipy.linalg import expm


def create_skew(vector):
    x1 = vector[0]
    x2 = vector[1]
    x3 = vector[2]
    # x1, x2, x3 = vector

    return np.array([[0, -x3, x2], [x3, 0, -x1], [-x2, x1, 0]])


def rpy2rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


# def rotation_matrix2rpy(R):
#     # 3x3 회전행렬 → roll, pitch, yaw
#     pass


# 단위회전축 ω와 각변위 θ로부터 회전행렬 생성
def omega2rotation_matrix(w, theta):
    skew_omega = create_skew(w)

    return expm(skew_omega * theta)  # 행렬(ω), 스칼라 곱
    # return np.eye(3) + np.sin(theta) * skew_omega + (1 - np.cos(theta)) * (skew_omega @ skew_omega) # Rodrigues' formula version
