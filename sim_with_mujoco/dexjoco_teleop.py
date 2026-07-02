import os
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np

from sim.model.math3d.rotation import rpy2rotation_matrix
from sim_with_mujoco.utils.dynamics import computed_torque_control
from sim_with_mujoco.utils.ik_qp import solve_differential_ik
from sim_with_mujoco.utils.math3d import get_body_T
from sim_with_mujoco.viewer.viewer import Viewer


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_XML_PATH = (
    ROOT_DIR
    / "assets"
    / "robots"
    / "dexjoco"
    / "xmls"
    / "arena_arm_hand_bucket_pick.xml"
)
XML_PATH = Path(os.environ.get("DEXJOCO_XML_PATH", DEFAULT_XML_PATH))

EE_BODY_NAME = "allegro_palm"

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
ALLEGRO_OPEN = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.263, 0.0, 0.0, 0.0]
)
ALLEGRO_CLOSED = np.array(
    [0.0, 1.2, 1.0, 0.7, 0.0, 1.2, 1.0, 0.7, 0.0, 1.2, 1.0, 0.7, 1.0, 0.2, 0.8, 0.6]
)

def mj_id(model, objtype, name):
    obj_id = mujoco.mj_name2id(model, objtype, name)
    if obj_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return obj_id


def joint_qpos_addr(model, joint_name):
    joint_id = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_qposadr[joint_id]


def actuator_id(model, actuator_name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)


def clamp_ctrl(model, actuator_id_i, ctrl):
    if model.actuator_ctrllimited[actuator_id_i]:
        lo, hi = model.actuator_ctrlrange[actuator_id_i]
        return np.clip(ctrl, lo, hi)
    return ctrl


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
    target_T[:3, :3] = initial_T[:3, :3] @ rpy2rotation_matrix(
        panel_target[3],
        panel_target[4],
        panel_target[5],
    )
    return target_T


def apply_allegro_ctrl(model, data, thumb_alpha, finger_alpha):
    q_des = ALLEGRO_OPEN.copy()
    q_des[:12] = (1.0 - finger_alpha) * ALLEGRO_OPEN[:12] + finger_alpha * ALLEGRO_CLOSED[:12]
    q_des[12:] = (1.0 - thumb_alpha) * ALLEGRO_OPEN[12:] + thumb_alpha * ALLEGRO_CLOSED[12:]

    for name, value in zip(ALLEGRO_ACTUATOR_NAMES, q_des):
        aid = actuator_id(model, name)
        data.ctrl[aid] = clamp_ctrl(model, aid, value)


def apply_panda_motor_ctrl(model, data, tau):
    for actuator_name, tau_i in zip(PANDA_ACTUATOR_NAMES, tau):
        aid = actuator_id(model, actuator_name)
        gear = model.actuator_gear[aid, 0]
        ctrl = tau_i / gear if gear != 0.0 else tau_i
        data.ctrl[aid] = clamp_ctrl(model, aid, ctrl)


def load_model():
    if not XML_PATH.exists():
        raise FileNotFoundError(
            f"DexJoCo XML not found: {XML_PATH}\n"
            "Expected vendored asset path: assets/robots/dexjoco/xmls/arena_arm_hand_bucket_pick.xml"
        )

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    initialize_state(model, data)
    return model, data


def main():
    model, data = load_model()
    viewer = Viewer(model, data)

    ee_body_id = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
    initial_pose = get_body_T(data, ee_body_id)
    target_T = initial_pose.copy()

    viewer.init_viewer(initial_pose[:3, 3], slider_range=(-0.6, 0.6), rotation_slider_range=(-0.5, 0.5))
    viewer.cam.lookat[:] = [0.0, 0.0, 1.05]
    viewer.cam.distance = 2.4
    viewer.cam.azimuth = 150
    viewer.cam.elevation = -25

    joint_ids = np.array(
        [mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in PANDA_JOINT_NAMES],
        dtype=int,
    )
    q_des = data.qpos.copy()
    q_dot_des = np.zeros(len(joint_ids))
    q_dotdot_des = np.zeros(len(joint_ids))
    ref_data = mujoco.MjData(model)

    sim_steps_per_frame = 1
    steps_per_sim = 8
    poll_interval = 1.0 / 60.0
    last_poll_time = 0.0

    thumb_alpha = 0.0
    finger_alpha = 0.0

    try:
        while not glfw.window_should_close(viewer.window):
            for _ in range(sim_steps_per_frame):
                glfw.poll_events()

                now = time.time()
                if now - last_poll_time >= poll_interval:
                    last_poll_time = now
                    polled_target, _ = viewer.poll_target()

                    if polled_target is not None:
                        target_T = make_target_T(initial_pose, polled_target)
                        thumb_alpha = float(polled_target[6])
                        finger_alpha = float(polled_target[7])

                for _ in range(steps_per_sim):
                    mujoco.mj_copyData(ref_data, model, data)
                    ref_data.qpos[:] = q_des
                    ref_data.qvel[:] = 0.0
                    mujoco.mj_forward(model, ref_data)

                    q_des, q_dot_des, _ = solve_differential_ik(
                        model,
                        ref_data,
                        [(ee_body_id, target_T, True, 1.0)],
                        joint_ids,
                        model.opt.timestep,
                    )
                    q_dotdot_des[:] = 0.0

                    tau_des = computed_torque_control(
                        model,
                        data,
                        q_des,
                        q_dot_des,
                        q_dotdot_des,
                        joint_ids,
                        kp=35,
                        kd=12,
                    )

                    apply_panda_motor_ctrl(model, data, tau_des)
                    apply_allegro_ctrl(model, data, thumb_alpha, finger_alpha)
                    mujoco.mj_step(model, data)

            viewer.render()
    finally:
        viewer.terminate_viewer()


if __name__ == "__main__":
    main()
