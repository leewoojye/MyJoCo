import numpy as np
from sim.model.kinematics.fk import compute_fk
from sim.model.math3d.screw import unit_screw_axis, screw_hat


# 감쇠 역행렬
# 아무리 특이점에 가까워지더라도 람다 값 때문에 분모가 0이 되지 않아 관절 속도가 안전하게 제한됨
def damped_pseudoinverse(J, lambda_=1e-3): # 예약어 충돌방지를 위한 언더바
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + lambda_**2 * np.eye(m))


def solve_newton_raphson_coordinate():
    return


def solve_newton_raphson_geometric():
    return


def solve_position_ik():
    return


def solve_pose_ik_recursive():
    return


# 관절각 계산만
def solve_pose_ik():
    return


# 계산된 관절각 적용
def apply_pose_ik():
    return
