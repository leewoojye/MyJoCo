import numpy as np


# 객체로 표현된 rotation, transform matrix도 고려

# 4x4 skew-symmetric matrix 생성
def create_skew(vector):
    x1 = vector[0]
    x2 = vector[1]
    x3 = vector[2]
    return np.array([[0, -x3, x2], [x3, 0, -x1], [-x2, x1, 0]])


# 회전행렬과 위치벡터로 구성된 동차변환행렬(transform matrix) 생성
def create_transform_matrix(R, p):
    T = np.eye(4)  # 4x4 단위행렬 생성
    T[:3, :3] = R
    T[:3, 3] = p
    return T


# 동차변환행렬의 역행렬을 반환
def inverse_T(T):
    T = T.copy()  # 값복사
    R = T[:3, :3]
    p = T[:3, 3]
    T[:3, :3] = np.transpose(R)
    T[:3, 3] = (-1) * np.transpose(R) @ p
    return T
