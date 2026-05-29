import mujoco
import numpy as np

# MjData의 contact 객체
# c.geom1: 접촉한 첫 번째 geom id
# c.geom2: 접촉한 두 번째 geom id
# c.dist: 두 geom 사이 거리
# c.friction: friction parameters


def is_hand_finger_contact(body1, body2):
    pair = {body1, body2}

    return ("hx5_r_base" in pair and any(name.startswith("finger_r_link") for name in pair)) or (
        "hx5_l_base" in pair and any(name.startswith("finger_l_link") for name in pair)
    )


def is_robot_table_contact(body1, body2):
    pair = {body1, body2}

    return (
        ("base_table" in pair and any(name.startswith("finger_r_link") for name in pair))
        or ("base_table" in pair and any(name.startswith("finger_l_link") for name in pair))
        or ("base_table" in pair and any(name.startswith("arm_r_link") for name in pair))
        or ("base_table" in pair and any(name.startswith("arm_l_link") for name in pair))
        or ("base_table" in pair and "hx5_r_base" in pair)
        or ("base_table" in pair and "hx5_l_base" in pair)
    )


# robot-table 등 hard collision 검사
def is_collision(model, data):
    ignored_pairs = {  # 캔-테이블, 테이블-바닥, 바닥-캔 쌍은 충돌 판정에서 제외
        frozenset({"world", "pr_cokeCan"}),
        frozenset({"base_table", "pr_cokeCan"}),
        frozenset({"world", "base_table"}),
    }

    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1])
        body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2])

        # if frozenset({body1, body2}) in ignored_pairs:
        #     continue
        # if body1 == body2:
        #     continue
        # if is_hand_finger_contact(body1, body2):  # hand-finger geom 간 dist가 가깝게 측정되는 문제
        #     continue
        # if contact.dist < -1e-4:  # penetration, mj_step() 내부 설정과 동일
        #     return True

        if is_robot_table_contact(body1, body2):
            if contact.dist < -1e-4:
                return True

    return False
