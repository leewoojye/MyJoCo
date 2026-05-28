import mujoco
import numpy as np


# <position> actuator:
# ctrl = 목표 관절각, MuJoCo가 force 계산

# <motor> actuator:
# ctrl = 직접 힘/토크 명령

# <velocity> actuator:
# ctrl = 목표 속도, MuJoCo가 속도 오차 기반 force 계산


def solve_inverse_dynamics(model, data, qacc):
    data.qacc = qacc.copy()
    mujoco.mj_inverse(model, data)
    qfrc_inverse = data.qfrc_inverse  # inverse dynamics 결과
    return qfrc_inverse.copy()
