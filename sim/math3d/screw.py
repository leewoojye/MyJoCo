import numpy as np
from sim.math3d.rotation import skew

# q: position, s: screw axis, h: pitch
def unit_screw_axis(q, s, h, joint_type):
  q = np.asarray(q, dtype=float)
  s = np.asarray(s, dtype=float)

  # 단위 회전축 벡터 s 크기 정규화
  norm = np.linalg.norm(s)
  if norm == 0:
    raise ValueError("s must be nonzero")
  s = s / norm

  # 회전관절인 경우 회전축은 0, 속도 v에는 선속도 성분만 남음
  if joint_type == "prismatic":
    w = np.zeros(3)
    v = s
  else:
    w = s
    v = -np.cross(s, q) + h * s

  S = np.concatenate((w, v)).reshape(6, 1)
  return S

def screw_hat(S):
  S = np.asarray(S, dtype=float).reshape(6,)
  w = S[:3]
  v = S[3:]
  
  S_hat = np.zeros((4, 4))
  S_hat[:3, :3] = skew(w)
  S_hat[:3, 3] = v

  return S_hat