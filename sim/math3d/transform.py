import numpy as np

# 4x4 skew-symmetric matrix 생성(반대칭 행렬)
def create_skew(vector):
  x1 = vector[0]
  x2 = vector[1]
  x3 = vector[2]
  return np.array([
    [0, -x3, x2],
    [x3, 0, -x1],
    [-x2, x1, 0]
  ])

# 회전행렬과 위치벡터로 구성된 동차변환행렬(transform matrix) 생성
def create_transform_matrix(R, p):
  T = np.eye(4) # 4x4 단위행렬 생성
  T[:3, :3] = R
  T[:3, 3] = p
  return T

# 단위회전축 ω와 각변위 θ로부터 회전행렬 생성
def omega2rotation_matrix(w, theta):
  skew_omega = create_skew(w)
  skew_omega @ 
  return np.eye(3) + np.sin(theta)