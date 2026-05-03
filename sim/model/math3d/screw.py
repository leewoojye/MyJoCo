import numpy as np
from sim.model.math3d.rotation import create_skew


# q: position, s: screw axis, h: pitch
def unit_screw_axis(q, s, h, joint_type):
    q = np.asarray(q, dtype=float)
    s = np.asarray(s, dtype=float)

    # 단위 회전축 벡터 s 크기 정규화
    norm = np.linalg.norm(s)
    if norm == 0:
        raise ValueError("s must be nonzero")
    s = s / norm

    joint_type_value = getattr(joint_type, "value", joint_type)

    # 직선관절인 경우 회전축은 0, 속도 v에는 선속도 성분만 남음
    if joint_type_value == "slide":
        w = np.zeros(3)
        v = s
    elif joint_type_value == "hinge":  # 회전관절
        w = s
        v = -np.cross(s, q)
        # v = -np.cross(s, q) + h * s
    else:
        w = s
        v = -np.cross(s, q) + h * s

    S = np.concatenate((w, v)).reshape(6, 1)  # 열벡터 형식으로 반환
    return S


# 햇 연산
def screw_hat(S):
    S = np.asarray(S, dtype=float).reshape(
        6,
    )
    w = S[:3]
    v = S[3:]

    S_hat = np.zeros((4, 4))
    S_hat[:3, :3] = create_skew(w)
    S_hat[:3, 3] = v

    return S_hat


# 햇 반대 연산(vee 연산)
def screw_vee(S_hat):
    w = np.array(
        [
            S_hat[2, 1],
            S_hat[0, 2],
            S_hat[1, 0],
        ]
    )

    v = S_hat[:3, 3]

    return np.concatenate([w, v])  # axis=0으로 concat, 열벡터 반환
