import mujoco
import numpy as np


def solve_inverse_dynamics(model, data, qacc):
    data.qacc = qacc.copy()
    mujoco.mj_inverse(model, data)
    qfrc_inverse = data.qfrc_inverse  # inverse dynamics 결과
    return qfrc_inverse.copy()
