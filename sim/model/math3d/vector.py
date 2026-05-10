import numpy as np


# a->b 방향 법선 단위 벡터
def contact_normal(point_a, point_b):
    normal = point_b - point_a
    norm = np.linalg.norm(normal)

    if norm < 1e-8:  # 크기가 0으로 수렴하는 법선벡터 처리
        return None

    return normal / norm


# 접선 단위 벡터
def contact_tangent(v_rel, normal):
    v_n = np.dot(v_rel, normal) * normal
    tangent = v_rel - v_n

    norm = np.linalg.norm(tangent)
    if norm > 1e-8:
        tangent = tangent / norm
    else:
        tangent = None

    return tangent
