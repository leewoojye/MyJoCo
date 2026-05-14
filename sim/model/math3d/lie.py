import numpy as np
from sim.model.math3d.rotation import create_skew
from sim.model.math3d.transform import inverse_T


# big adjont matrix 생성
def Adjoint(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]

    ad = np.zeros((6, 6))
    ad[:3, :3] = R
    ad[3:, :3] = create_skew(p) @ R
    ad[3:, 3:] = R
    return ad


# small adjoint matrix
def adjoint(T: np.ndarray) -> np.ndarray:
    return


# 회전축을 갖는 특수한 벡터(자코비안,트위스트,wrench)는 변환 행렬 T를 바로 곱해주는 게 아닌 Adj(T)를 곱해야 변환 효과가 나타남
def transform_frame_jacobian(T_ab, J_b):
    return Adjoint(T_ab) @ J_b  # J_a


# (참고) twsit, wrench는 dual 관계
# 이는 두 벡터 곱의 결과가 스칼라라, 한 벡터의 좌표계가 변하면 다른 벡터가 이를 상쇄하기 위한 좌표계를 가져야 함을 암시
def transform_frame_wrench(T_ab, F_b):
    T_ba = inverse_T(T_ab)
    return Adjoint(T_ba).T @ F_b
