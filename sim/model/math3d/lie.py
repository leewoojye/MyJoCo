import numpy as np
from sim.model.math3d.rotation import create_skew


# big adjont matrix 생성
def Adjoint(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    if T.shape != (4, 4):
        raise ValueError("shape error")
    ad = np.zeros((6, 6))
    ad[:3, :3] = R
    ad[3:, :3] = create_skew(p) @ R
    ad[3:, 3:] = R
    return ad


# small adjoint matrix
def adjoint(T: np.ndarray) -> np.ndarray:
    return
