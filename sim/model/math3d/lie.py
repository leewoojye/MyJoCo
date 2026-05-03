import numpy as np
from sim.model.math3d.rotation import skew

def adjoint(T: np.ndarray) -> np.ndarray:
  R = T[:3, :3]
  p = T[:3, 3]
  if T.shape != (4, 4):
    raise ValueError(f"T must have shape (4, 4), got {T.shape}")
  ad = np.zeros((6,6))
  ad[:3, :3] = R
  ad[3:, :3] = skew(p) @ R
  ad[3:, 3:] = R
  return ad