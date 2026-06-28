from pathlib import Path

import glfw
import mujoco
import numpy as np

from sim.model.math3d.rotation import rpy2rotation_matrix
from sim_with_mujoco.utils.math3d import get_site_T
from sim_with_mujoco.viewer.viewer import Viewer


ROOT_DIR = Path(__file__).resolve().parents[1]
XML_PATH = ROOT_DIR / "assets" / "robots" / "apptronik_apollo" / "scene.xml"

PALM_SITE_NAME = "r_palm_site"
STAND_KEY_NAME = "stand"
OBJECT_JOINT_NAME = "hot_dog_free"
OBJECT_QPOS = np.array([0.0, -0.32, 0.965, 0.70710678, 0.0, 0.0, 0.70710678])
RIGHT_ARM_JOINT_NAMES = [
    "r_shoulder_aa",
    "r_shoulder_ie",
    "r_shoulder_fe",
    "r_elbow_fe",
    "r_wrist_roll",
    "r_wrist_yaw",
    "r_wrist_pitch",
]


def mj_id(model, objtype, name):
    obj_id = mujoco.mj_name2id(model, objtype, name)
    if obj_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return obj_id


def joint_ids_from_names(model, joint_names):
    return [mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]


def actuated_dof_ids(model):
    dof_ids = []
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue

        joint_id = model.actuator_trnid[actuator_id, 0]
        if model.jnt_type[joint_id] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            dof_ids.append(model.jnt_dofadr[joint_id])

    return np.array(sorted(set(dof_ids)), dtype=int)


def actuator_id_from_joint(model, joint_id):
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        if model.actuator_trnid[actuator_id, 0] == joint_id:
            return actuator_id
    raise ValueError(f"Actuator not found for joint id: {joint_id}")


def rotation_vector(R):
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    return (
        theta
        / (2.0 * np.sin(theta))
        * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    )


def damped_pseudoinverse(J, damping=1e-2):
    rows, cols = J.shape
    if rows <= cols:
        return J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(rows))
    return np.linalg.inv(J.T @ J + damping**2 * np.eye(cols)) @ J.T


def clamp_ctrl(model, actuator_id, ctrl):
    if model.actuator_ctrllimited[actuator_id]:
        lo, hi = model.actuator_ctrlrange[actuator_id]
        return np.clip(ctrl, lo, hi)
    return ctrl


def make_target_T(initial_T, panel_target):
    target_T = initial_T.copy()
    target_T[:3, 3] = panel_target[:3]
    target_T[:3, :3] = initial_T[:3, :3] @ rpy2rotation_matrix(
        panel_target[3],
        panel_target[4],
        panel_target[5],
    )
    return target_T


def solve_site_ik(model, data, site_id, target_T, joint_ids, damping=5e-2):
    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = data.qpos
    ik_data.qvel[:] = 0.0
    mujoco.mj_forward(model, ik_data)

    dof_ids = np.array([model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=int)
    max_dq_norm = 0.04

    for _ in range(60):
        current_T = get_site_T(ik_data, site_id)
        pos_err = target_T[:3, 3] - current_T[:3, 3]
        rot_err = rotation_vector(target_T[:3, :3] @ current_T[:3, :3].T)
        err = np.hstack([pos_err, rot_err])

        if np.linalg.norm(err) < 1e-4:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        J = np.vstack([jacp, jacr])[:, dof_ids]

        dq = damped_pseudoinverse(J, damping=damping) @ err
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_dq_norm:
            dq = dq / dq_norm * max_dq_norm

        for joint_id, dq_i in zip(joint_ids, dq):
            qadr = model.jnt_qposadr[joint_id]
            ik_data.qpos[qadr] += dq_i
            if model.jnt_limited[joint_id]:
                lo, hi = model.jnt_range[joint_id]
                ik_data.qpos[qadr] = np.clip(ik_data.qpos[qadr], lo, hi)

        mujoco.mj_forward(model, ik_data)

    return ik_data.qpos.copy()


def apply_position_ctrl(model, data, q_des, joint_ids):
    for joint_id in joint_ids:
        actuator_id = actuator_id_from_joint(model, joint_id)
        qadr = model.jnt_qposadr[joint_id]
        data.ctrl[actuator_id] = clamp_ctrl(model, actuator_id, q_des[qadr])


def set_object_pose(model, data):
    joint_id = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_JOINT_NAME)
    qadr = model.jnt_qposadr[joint_id]
    dadr = model.jnt_dofadr[joint_id]
    data.qpos[qadr : qadr + 7] = OBJECT_QPOS
    data.qvel[dadr : dadr + 6] = 0.0


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    key_id = mj_id(model, mujoco.mjtObj.mjOBJ_KEY, STAND_KEY_NAME)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    set_object_pose(model, data)
    mujoco.mj_forward(model, data)

    site_id = mj_id(model, mujoco.mjtObj.mjOBJ_SITE, PALM_SITE_NAME)
    joint_ids = joint_ids_from_names(model, RIGHT_ARM_JOINT_NAMES)
    gravity_comp_dof_ids = actuated_dof_ids(model)
    hold_ctrl = data.ctrl.copy()

    initial_T = get_site_T(data, site_id)
    target_T = initial_T.copy()
    q_des = data.qpos.copy()

    viewer = Viewer(model, data)
    viewer.init_viewer(initial_T[:3, 3], slider_range=(-0.35, 0.35), rotation_slider_range=(-0.5, 0.5))
    viewer.cam.lookat[:] = [0.25, -0.55, 0.9]
    viewer.cam.distance = 1.8
    viewer.cam.azimuth = 145
    viewer.cam.elevation = -18
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

    sim_steps_per_frame = 8

    try:
        while not glfw.window_should_close(viewer.window):
            glfw.poll_events()
            polled_target, _ = viewer.poll_target()

            if polled_target is not None:
                target_T = make_target_T(initial_T, polled_target)
                q_des = solve_site_ik(model, data, site_id, target_T, joint_ids)

            for _ in range(sim_steps_per_frame):
                data.ctrl[:] = hold_ctrl
                apply_position_ctrl(model, data, q_des, joint_ids)
                mujoco.mj_forward(model, data)
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[gravity_comp_dof_ids] = data.qfrc_bias[gravity_comp_dof_ids]
                mujoco.mj_step(model, data)

            viewer.render()
    finally:
        viewer.terminate_viewer()


if __name__ == "__main__":
    main()
