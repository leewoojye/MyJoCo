import mujoco
import numpy as np

from sim.model.kinematics.ik import calculate_twist_error
from sim.model.motion.trajectory import interpolate_joint_ros, interpolate_pose
from sim_with_mujoco.utils.ik import solve_ik
from sim_with_mujoco.utils.math3d import get_body_T, get_singular_values

# singularity check
# stacked IK weighting
# trajectory blending
# ROS JointTrajectory


# cartesian waypoint별 IK로 joint trajectory 생성 및 planning 결과 반환
def plan_cartesian_trajectory(
    model,
    data,
    targets,
    is_pose,
    joint_names,
    duration,
    step_distance=0.01,  # task-space 상 이동거리단위
    check_collision=False,
    damping=1e-2,
    pose_threshold=1e-3,
    joint_step_limit=0.35,
    singularity_threshold=None,
    max_attempt=3,
):
    # 현재 pose와 목표 pose/position 사이의 오차 norm 계산
    def pose_error_norm(check_data, target_body_id, T_des, is_pose):
        T_curr = get_body_T(check_data, target_body_id)

        if is_pose:
            _, err = calculate_twist_error(T_curr, T_des)
            return np.linalg.norm(err)

        return np.linalg.norm(T_des[:3, 3] - T_curr[:3, 3])

    body_id, target_T = targets[0]
    start_T = get_body_T(data, body_id)
    distance = np.linalg.norm(target_T[:3, 3] - start_T[:3, 3])
    num_segments = max(1, int(np.ceil(distance / step_distance)))  # 구간 개수 설정

    ref_data = mujoco.MjData(model)
    check_data = mujoco.MjData(model)
    q_prev = data.qpos.copy()

    joint_ids = None
    q_waypoints = [q_prev.copy()]
    time_from_start = [0.0]
    success = True

    s_prev = 0.0
    ds = 1.0 / num_segments
    ds_default = ds
    retry_count = 0

    while s_prev < 1.0:
        s = min(1.0, s_prev + ds)
        t = duration * s
        T_i, _, _ = interpolate_pose(start_T, target_T, duration, t)

        mujoco.mj_copyData(ref_data, model, data)
        ref_data.qpos[:] = q_prev
        ref_data.qvel[:] = 0.0
        mujoco.mj_forward(model, ref_data)

        waypoint_targets = [(body_id, T_i), *targets[1:]]

        q_i, joint_ids = solve_ik(
            model,
            ref_data,
            waypoint_targets,
            is_pose=is_pose,
            joint_names=joint_names,
            check_collision=check_collision,
            damping=damping,
        )

        check_data.qpos[:] = q_i
        check_data.qvel[:] = 0.0
        mujoco.mj_forward(model, check_data)

        target_errors = [
            pose_error_norm(check_data, target_body_id, target_T_i, target_is_pose)
            for (target_body_id, target_T_i), target_is_pose in zip(waypoint_targets, is_pose)
        ]
        max_pose_error = max(target_errors)

        qpos_ids = model.jnt_qposadr[joint_ids]
        dq = q_i[qpos_ids] - q_prev[qpos_ids]

        retry_waypoint = max_pose_error > pose_threshold or np.linalg.norm(dq) > joint_step_limit

        if singularity_threshold is not None:
            sigular_min, _ = get_singular_values(model, check_data, body_id, joint_ids)
            if sigular_min < singularity_threshold:
                retry_waypoint = True

        if retry_waypoint:
            if retry_count < max_attempt:
                ds *= 0.5
                retry_count += 1
                continue

            success = False
            break

        q_waypoints.append(q_i.copy())
        time_from_start.append(t)
        q_prev = q_i.copy()
        s_prev = s
        ds = min(ds_default, ds * 2.0)
        retry_count = 0

    return {
        "success": success and s_prev >= 1.0,
        "q_waypoints": np.asarray(q_waypoints),
        "time_from_start": np.asarray(time_from_start),
        "joint_ids": np.asarray(joint_ids, dtype=int) if joint_ids is not None else np.array([], dtype=int),
        "qpos_ids": model.jnt_qposadr[joint_ids] if joint_ids is not None else np.array([], dtype=int),
    }


# 시간 기준 joint trajectory 샘플링 및 q, qdot, qddot 반환
def sample_trajectory(
    q_waypoints,
    time_from_start,
    t,
    qpos_ids=None,
    q_dot_waypoints=None,
    q_dotdot_waypoints=None,
):
    q_waypoints = np.asarray(q_waypoints)
    time_from_start = np.asarray(time_from_start)

    # if qpos_ids is None:
    #     waypoints = q_waypoints
    #     qpos_ids = None
    # else:
    #     qpos_ids = np.asarray(qpos_ids, dtype=int)
    #     waypoints = q_waypoints[:, qpos_ids]

    qpos_ids = np.asarray(qpos_ids, dtype=int)
    waypoints = q_waypoints[:, qpos_ids]

    # waypoint 차분 기반 속도 짐작, 인접한 구간을 매끄럽게 잇는 효과
    def qvel_from_waypoint():
        q_dot = np.zeros_like(waypoints)

        if len(waypoints) <= 2:
            return q_dot

        for i in range(1, len(waypoints) - 1):
            dt = time_from_start[i + 1] - time_from_start[i - 1]
            q_dot[i] = (waypoints[i + 1] - waypoints[i - 1]) / dt

        return q_dot

    if q_dot_waypoints is None:
        q_dot_waypoints = qvel_from_waypoint()
    else:
        q_dot_waypoints = np.asarray(q_dot_waypoints)

    if q_dotdot_waypoints is None:
        q_dotdot_waypoints = np.zeros_like(waypoints)
    else:
        q_dotdot_waypoints = np.asarray(q_dotdot_waypoints)

    if t <= time_from_start[0]:
        q = waypoints[0].copy()
        q_dot = q_dot_waypoints[0].copy()
        q_dotdot = q_dotdot_waypoints[0].copy()
    elif t >= time_from_start[-1]:
        q = waypoints[-1].copy()
        q_dot = q_dot_waypoints[-1].copy()
        q_dotdot = q_dotdot_waypoints[-1].copy()
    else:
        seg_idx = np.searchsorted(time_from_start, t, side="right") - 1
        t0 = time_from_start[seg_idx]
        t1 = time_from_start[seg_idx + 1]
        dt = t1 - t0
        tau = t - t0

        q0 = waypoints[seg_idx]
        q1 = waypoints[seg_idx + 1]
        v0 = q_dot_waypoints[seg_idx]
        v1 = q_dot_waypoints[seg_idx + 1]
        a0 = q_dotdot_waypoints[seg_idx]
        a1 = q_dotdot_waypoints[seg_idx + 1]

        q, q_dot, q_dotdot = interpolate_joint_ros(
            q0,
            q1,
            v0,
            a0,
            dt,
            tau,
            q_dot_end=v1,
            q_dotdot_end=a1,
        )

    # if qpos_ids is None:
    #     return q, q_dot, q_dotdot

    q_full = q_waypoints[0].copy()
    q_full[qpos_ids] = q

    return q_full, q_dot, q_dotdot
