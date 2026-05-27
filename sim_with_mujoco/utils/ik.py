import mujoco
import numpy as np

from sim.model.kinematics.ik import calculate_twist_error
from sim.model.math3d.lie import Adjoint
from sim_with_mujoco.utils.math3d import get_body_T


# 아무리 특이점에 가까워지더라도 람다 값 때문에 분모가 0이 되지 않아 관절 속도가 안전하게 제한됨
def damped_pseudoinverse(J, damping=1e-3):
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(m))


# body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm")
# joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow")
# site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site") # body(link) 위에 붙여둔 특정 위치/방향 표식
def solve_pose_ik(model, data, body_id, target_T, is_pose=True):  # 비고: site_id
    # 궤적 보간 및 rpy는 외부에서 적용하고, 즉 정확한 target pose는 외부에서 설정

    # theta_prev = data.qpos.copy()  # forward 적용 이전의 qpos
    # theta = data.qpos.copy()
    iter = 0
    joint_ids = []
    bid = body_id
    # e.e->base 방향으로 순회하며 joint id 배열을 구함
    while bid > 0:
        for jnt_offset in range(model.body_jntnum[bid]):
            jid = model.body_jntadr[bid] + jnt_offset
            # e.e->base를 잇는 kinematic tree에 회전/직선 관절만 있다고 가정
            if model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                joint_ids.append(jid)
        bid = model.body_parentid[bid]

    # joint id vs. dof id
    # 단일 관절은 여러 dof를 가질 수 있음(예. 회전축이 여러개). 다만 회전/직선 관절은 joint당 하나의 dof를 갖음
    joint_ids = joint_ids[::-1]  # e.e에서 거슬러 올라가 만든 배열을 뒤집음
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=int)

    while True:
        # 자코비안(body origin 기준), 트위스트 에러 계산
        T_sb = get_body_T(data, body_id)
        T_sd = target_T
        if is_pose:
            _, err = calculate_twist_error(T_sb, T_sd)  # 트위스트 에러
        else:
            err = T_sd[:3, 3] - T_sb[:3, 3]  # 위치 오차 벡터

        # body origin 기준으로 자코비안 계산
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)  # mj_jacBody vs. mj_jacSite

        J = np.vstack([jacr, jacp])  # shape: (6, model.nv), (w, v)열의 집합
        J_b = Adjoint(np.linalg.inv(T_sb)) @ J  # 전체 jacobian DOF에 대한 body frame jacobian
        J_active = J_b[:, dof_ids]  # IK에 사용할 jacobian 열만 필터링
        rows, cols = J_active.shape

        if is_pose:  # pose IK
            if rows == cols:
                # dq = np.linalg.pinv(J_active) @ twist_error # Moore–Penrose pseudoinverse
                dq = damped_pseudoinverse(J_active) @ err
            elif rows > cols:
                dq = damped_pseudoinverse(J_active.T @ J_active) @ J_active.T @ err
            else:
                dq = J_active.T @ damped_pseudoinverse(J_active @ J_active.T) @ err
        else:  # position IK
            if rows == cols:
                dq = np.dot(damped_pseudoinverse(J_active), err)
            elif (
                rows > cols
            ):  # closest in the 2-norm sense, 원하는 task velocity를 모두 만들 수 없으니 가장 가까운 해를 찾음
                dq = np.dot(damped_pseudoinverse(J_active.T @ J_active) @ J_active.T, err)
            else:  # smallest 2-norm among all solutions, 목표 velocity가 여러 개 있을 수 있으니 이 중 가장 작은 해를 고름
                dq = np.dot(J_active.T @ damped_pseudoinverse(J_active @ J_active.T), err)

        for jid, dq_i in zip(joint_ids, dq):
            # jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qadr = model.jnt_qposadr[jid]

            data.qpos[qadr] += dq_i

            # 관절각 제약 기반 클리핑
            if model.jnt_limited[jid]:
                q_lower, q_upper = model.jnt_range[jid]
                data.qpos[qadr] = np.clip(data.qpos[qadr], q_lower, q_upper)

        # 각변위만큼 이동
        mujoco.mj_forward(model, data)
        T_sb = get_body_T(data, body_id)

        # forward 이후 갱신된 트위스트(위치) 오차로 종료 조건 검사 (반복문 앞에 배치해도 무관)
        if is_pose:
            _, err = calculate_twist_error(T_sb, T_sd)  # 트위스트 에러
        else:
            err = T_sd[:3, 3] - T_sb[:3, 3]  # 위치 오차 벡터

        if np.linalg.norm(err) <= 1e-4:
            break
        if iter > 20:
            break

        iter += 1

    # 각변위 제약 반영: 사용자 지정 제약 + xml 명세 기반 제약
    # max_theta_step = 0.1
    # delta_theta = theta - theta_prev
    # delta_theta = np.clip(delta_theta, -max_theta_step, max_theta_step)

    # 반환: ik error
    return
