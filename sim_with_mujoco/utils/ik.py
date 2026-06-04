import mujoco
import numpy as np

from sim.model.kinematics.ik import calculate_twist_error
from sim.model.math3d.lie import Adjoint
from sim_with_mujoco.utils.collision import is_collision
from sim_with_mujoco.utils.math3d import get_body_T


# 아무리 특이점에 가까워지더라도 람다 값 때문에 분모가 0이 되지 않아 관절 속도가 안전하게 제한됨
def damped_pseudoinverse(J, damping=1e-3):
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(m))


# body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm")
# joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow")
# site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site") # body(link) 위에 붙여둔 특정 위치/방향 표식
def solve_ik(
    model, data, body_id, target_T=None, is_pose=[True, False], joint_names=None, check_collision=False
):  # 비고: site_id
    # 궤적 보간 및 rpy는 외부에서 적용하고, 즉 정확한 target pose는 외부에서 설정
    if target_T is None:
        targets = list(body_id)
    else:
        targets = [(body_id, target_T)]

    if isinstance(is_pose, (bool, np.bool_)):  # 일반화된 multi-task IK
        is_pose_list = [bool(is_pose)] * len(targets)
    else:
        is_pose_list = list(is_pose)

    # 복원용 MjData 저장
    prev_data = mujoco.MjData(model)
    prev_data.qpos[:] = data.qpos  # 값복사
    prev_data.qvel[:] = data.qvel
    mujoco.mj_forward(model, prev_data)  # 완전한 상태의 prev_data 만듦

    # IK 결과 반환용 MjData, 기존 data 변화 없이 결과 qpos만 반환하기 위함
    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = data.qpos
    ik_data.qvel[:] = data.qvel
    mujoco.mj_forward(model, ik_data)

    max_dq = 0.03
    max_iter = 50
    iter = 0
    if joint_names is None:
        joint_ids = []
        body_id = targets[0][0]
        # e.e->base 방향으로 순회하며 joint id 배열을 구함
        while body_id > 0:
            for jnt_offset in range(model.body_jntnum[body_id]):
                joint_id = model.body_jntadr[body_id] + jnt_offset
                # e.e->base를 잇는 kinematic tree에 회전/직선 관절만 있다고 가정
                if model.jnt_type[joint_id] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                    joint_ids.append(joint_id)
            body_id = model.body_parentid[body_id]

        # joint id vs. dof id
        # 단일 관절은 여러 dof를 가질 수 있음(예. 회전축이 여러개). 다만 회전/직선 관절은 joint당 하나의 dof를 갖음
        joint_ids = joint_ids[::-1]  # e.e에서 거슬러 올라가 만든 배열을 뒤집음
    else:
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in joint_names]
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=int)

    def get_stacked_ik():  # multi-target에 대해 자코비안 행렬 및 에러의 집합을 반환
        J_list = []
        err_list = []

        # 자코비안(body origin 기준), 트위스트 에러 계산
        for (target_body_id, T_sd), is_pose_i in zip(targets, is_pose_list):
            T_sb = get_body_T(ik_data, target_body_id)
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, ik_data, jacp, jacr, target_body_id)  # mj_jacBody vs. mj_jacSite

            if is_pose_i:
                _, err_i = calculate_twist_error(T_sb, T_sd)  # 트위스트 에러
                J = np.vstack([jacr, jacp])  # shape: (6, model.nv), (w, v)열의 집합
                J_b = Adjoint(np.linalg.inv(T_sb)) @ J  # 전체 jacobian DOF에 대한 body frame jacobian
                J_i = J_b[:, dof_ids]  # IK에 사용할 jacobian 열만 필터링
            else:
                err_i = T_sd[:3, 3] - T_sb[:3, 3]  # 위치 오차 벡터
                J_i = jacp[:, dof_ids]

            J_list.append(J_i)
            err_list.append(err_i)

        return np.vstack(J_list), np.hstack(err_list)

    while True:
        J_active, err = get_stacked_ik()
        rows, cols = J_active.shape
        if rows == cols:
            # dq = np.linalg.pinv(J_active) @ twist_error # Moore–Penrose pseudoinverse
            dq = damped_pseudoinverse(J_active) @ err
        elif rows > cols:
            dq = damped_pseudoinverse(J_active.T @ J_active) @ J_active.T @ err
        else:
            dq = J_active.T @ damped_pseudoinverse(J_active @ J_active.T) @ err

        # dq = np.clip(dq, -max_dq, max_dq) # qpos 클리핑시 infeasible한 해가 나올 가능성
        for joint_id, dq_i in zip(joint_ids, dq):
            qadr = model.jnt_qposadr[joint_id]

            # dq = np.clip(dq, -0.05, 0.05)
            ik_data.qpos[qadr] += dq_i

            # 관절각 제약 기반 클리핑
            if model.jnt_limited[joint_id]:
                q_lower, q_upper = model.jnt_range[joint_id]
                ik_data.qpos[qadr] = np.clip(ik_data.qpos[qadr], q_lower, q_upper)
            mujoco.mj_forward(model, ik_data)

        # mujoco.mj_forward(model, ik_data)
        # kinematic simulator를 위한 충돌 판별
        if check_collision and is_collision(model, ik_data):  # forward 결과가 penetration이면 prev_data로 rollback
            mujoco.mj_copyData(ik_data, model, prev_data)
        else:  # 충돌이 아니라면 prev_data 갱신
            mujoco.mj_copyData(prev_data, model, ik_data)

        # forward 이후 갱신된 트위스트(위치) 오차로 종료 조건 검사 (반복문 앞에 배치해도 무관)
        _, err = get_stacked_ik()
        if np.linalg.norm(err) <= 1e-4:
            break
        if iter > max_iter:
            break

        iter += 1

    # 각변위 제약 반영: 사용자 지정 제약 + xml 명세 기반 제약
    # max_theta_step = 0.1
    # delta_theta = theta - theta_prev
    # delta_theta = np.clip(delta_theta, -max_theta_step, max_theta_step)

    # 반환: ik target qpos
    return ik_data.qpos.copy(), joint_ids
