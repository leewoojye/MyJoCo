import os
from pathlib import Path

import glfw
import mujoco
import numpy as np

from sim.model.math3d.rotation import rpy2rotation_matrix
from sim_with_mujoco.utils.math3d import get_site_T
from sim_with_mujoco.viewer.viewer import Viewer


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_XML_PATH = (
    ROOT_DIR
    / "temp"
    / "dexjoco_src"
    / "dexjoco"
    / "dexjoco"
    / "sim"
    / "envs"
    / "xmls"
    / "arena_arm_hand_bucket_pick.xml"
)
XML_PATH = Path(os.environ.get("DEXJOCO_XML_PATH", DEFAULT_XML_PATH))

PANDA_HOME = np.array([0.0, -0.785, 0.0, -2.35, 0.0, 1.57, np.pi / 4])
PANDA_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
PANDA_ACTUATOR_NAMES = [f"actuator{i}" for i in range(1, 8)]

ALLEGRO_JOINT_NAMES = [
    "ffj0",
    "ffj1",
    "ffj2",
    "ffj3",
    "mfj0",
    "mfj1",
    "mfj2",
    "mfj3",
    "rfj0",
    "rfj1",
    "rfj2",
    "rfj3",
    "thj0",
    "thj1",
    "thj2",
    "thj3",
]
ALLEGRO_ACTUATOR_NAMES = [
    "ffa0",
    "ffa1",
    "ffa2",
    "ffa3",
    "mfa0",
    "mfa1",
    "mfa2",
    "mfa3",
    "rfa0",
    "rfa1",
    "rfa2",
    "rfa3",
    "tha0",
    "tha1",
    "tha2",
    "tha3",
]
ALLEGRO_OPEN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.263, 0.0, 0.0, 0.0])
ALLEGRO_CLOSED = np.array([0.0, 1.2, 1.0, 0.7, 0.0, 1.2, 1.0, 0.7, 0.0, 1.2, 1.0, 0.7, 1.0, 0.2, 0.8, 0.6])
ALLEGRO_BODY_PREFIXES = ("ff_", "mf_", "rf_", "th_")
OBJECT_BODY_PREFIXES = ("boxed_food", "bucket")


def mj_id(model, objtype, name):
    obj_id = mujoco.mj_name2id(model, objtype, name)
    if obj_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return obj_id


def joint_qpos_addr(model, joint_name):
    joint_id = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_qposadr[joint_id]


def joint_dof_addr(model, joint_name):
    joint_id = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_dofadr[joint_id]


def actuator_id(model, actuator_name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)


def rotation_vector(R):
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    return theta / (2.0 * np.sin(theta)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def clamp_ctrl(model, actuator_id_i, ctrl):
    if model.actuator_ctrllimited[actuator_id_i]:
        lo, hi = model.actuator_ctrlrange[actuator_id_i]
        return np.clip(ctrl, lo, hi)
    return ctrl


def tune_grasp_contact(model):
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) or ""
        is_hand_geom = body_name == "allegro_palm" or body_name.startswith(ALLEGRO_BODY_PREFIXES)
        is_object_geom = body_name.startswith(OBJECT_BODY_PREFIXES)

        if is_hand_geom:
            model.geom_friction[geom_id] = [2.0, 0.02, 0.001]
            model.geom_condim[geom_id] = 4
        elif is_object_geom:
            model.geom_friction[geom_id] = [1.5, 0.01, 0.001]
            model.geom_condim[geom_id] = 4


def initialize_state(model, data):
    for name, value in zip(PANDA_JOINT_NAMES, PANDA_HOME):
        data.qpos[joint_qpos_addr(model, name)] = value

    for name, value in zip(ALLEGRO_JOINT_NAMES, ALLEGRO_OPEN):
        data.qpos[joint_qpos_addr(model, name)] = value

    for name, value in zip(ALLEGRO_ACTUATOR_NAMES, ALLEGRO_OPEN):
        data.ctrl[actuator_id(model, name)] = value

    mujoco.mj_forward(model, data)


def make_target_T(initial_T, panel_target):
    target_T = initial_T.copy()
    target_T[:3, 3] = panel_target[:3]
    target_T[:3, :3] = initial_T[:3, :3] @ rpy2rotation_matrix(panel_target[3], panel_target[4], panel_target[5])
    return target_T


def apply_allegro_ctrl(model, data, thumb_alpha, finger_alpha):
    q_des = ALLEGRO_OPEN.copy()
    q_des[:12] = (1.0 - finger_alpha) * ALLEGRO_OPEN[:12] + finger_alpha * ALLEGRO_CLOSED[:12]
    q_des[12:] = (1.0 - thumb_alpha) * ALLEGRO_OPEN[12:] + thumb_alpha * ALLEGRO_CLOSED[12:]

    for name, value in zip(ALLEGRO_ACTUATOR_NAMES, q_des):
        aid = actuator_id(model, name)
        data.ctrl[aid] = clamp_ctrl(model, aid, value)


def site_opspace_torque(model, data, site_id, target_T):
    dof_ids = np.array([joint_dof_addr(model, name) for name in PANDA_JOINT_NAMES], dtype=int)
    qpos_ids = np.array([joint_qpos_addr(model, name) for name in PANDA_JOINT_NAMES], dtype=int)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    Jp = jacp[:, dof_ids]
    Jr = jacr[:, dof_ids]
    J = np.vstack([Jp, Jr])

    current_T = get_site_T(data, site_id)
    q = data.qpos[qpos_ids]
    dq = data.qvel[dof_ids]

    pos_err = target_T[:3, 3] - current_T[:3, 3]
    rot_err = rotation_vector(target_T[:3, :3] @ current_T[:3, :3].T)

    xdot = Jp @ dq
    w = Jr @ dq

    kp_pos = np.array([350.0, 350.0, 350.0])
    kd_pos = 2.0 * np.sqrt(kp_pos) * 2.5
    kp_rot = np.array([120.0, 120.0, 120.0])
    kd_rot = 2.0 * np.sqrt(kp_rot) * 2.0

    ddx = kp_pos * pos_err - kd_pos * xdot
    dw = kp_rot * rot_err - kd_rot * w
    task_acc = np.hstack([ddx, dw])

    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    M = M[dof_ids, :][:, dof_ids]
    M_inv = np.linalg.inv(M)
    Mx_inv = J @ M_inv @ J.T
    Mx = np.linalg.pinv(Mx_inv, rcond=1e-3)

    tau = J.T @ Mx @ task_acc

    kp_null = np.array([8.0, 8.0, 8.0, 6.0, 3.0, 3.0, 2.0])
    kd_null = 2.0 * np.sqrt(kp_null)
    ddq_null = kp_null * (PANDA_HOME - q) - kd_null * dq
    Jbar = M_inv @ J.T @ Mx
    tau += (np.eye(len(dof_ids)) - J.T @ Jbar.T) @ ddq_null
    tau += data.qfrc_bias[dof_ids]

    return tau


def apply_panda_ctrl(model, data, tau):
    for actuator_name, tau_i in zip(PANDA_ACTUATOR_NAMES, tau):
        aid = actuator_id(model, actuator_name)
        data.ctrl[aid] = clamp_ctrl(model, aid, tau_i)


def load_model():
    if not XML_PATH.exists():
        raise FileNotFoundError(
            f"DexJoCo XML not found: {XML_PATH}\n"
            "Expected clone: git clone --depth 1 https://github.com/brave-eai/dexjoco.git temp/dexjoco_src"
        )

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    tune_grasp_contact(model)
    data = mujoco.MjData(model)
    initialize_state(model, data)
    return model, data


def main():
    model, data = load_model()
    site_id = mj_id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    initial_T = get_site_T(data, site_id)
    target_T = initial_T.copy()

    viewer = Viewer(model, data)
    viewer.init_viewer(initial_T[:3, 3], slider_range=(-0.6, 0.6), rotation_slider_range=(-0.5, 0.5))
    viewer.cam.lookat[:] = [0.0, 0.0, 1.05]
    viewer.cam.distance = 2.4
    viewer.cam.azimuth = 150
    viewer.cam.elevation = -25

    sim_steps_per_frame = 8
    thumb_alpha = 0.0
    finger_alpha = 0.0

    try:
        while not glfw.window_should_close(viewer.window):
            glfw.poll_events()
            polled_target, _ = viewer.poll_target()

            if polled_target is not None:
                target_T = make_target_T(initial_T, polled_target)
                thumb_alpha = float(polled_target[6])
                finger_alpha = float(polled_target[7])

            for _ in range(sim_steps_per_frame):
                mujoco.mj_forward(model, data)
                tau = site_opspace_torque(model, data, site_id, target_T)
                apply_panda_ctrl(model, data, tau)
                apply_allegro_ctrl(model, data, thumb_alpha, finger_alpha)
                mujoco.mj_step(model, data)

            viewer.render()
    finally:
        viewer.terminate_viewer()


if __name__ == "__main__":
    main()
