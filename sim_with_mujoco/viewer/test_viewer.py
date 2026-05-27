import time

import glfw
import mujoco
import numpy as np

from sim_with_mujoco.viewer.glfw_panel import GlfwTargetPanel
# from sim_with_mujoco.mjcf import parser

XML_PATH = "/Users/woojyelee/workspace/my_robotics/assets/robots/robotis_ffw/scene_ffw_sh5.xml"

# # 1. GLFW window / OpenGL context 생성
# window = glfw.create_window(1200, 900, "Demo", None, None)
# glfw.make_context_current(window)

# # 2. MuJoCo visualization objects
# cam = mujoco.MjvCamera()
# opt = mujoco.MjvOption()
# scene = mujoco.MjvScene(model, maxgeom=10000)
# context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# # 3. 매 프레임
# mujoco.mj_step(model, data)

# viewport = mujoco.MjrRect(0, 0, width, height)

# mujoco.mjv_updateScene(
#     model,
#     data,
#     opt,
#     None,
#     cam,
#     mujoco.mjtCatBit.mjCAT_ALL,
#     scene,
# )

# mujoco.mjr_render(viewport, scene, context)

# glfw.swap_buffers(window)
# glfw.poll_events()


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)

    slider_range = (-0.2, 0.2)  # end-effector 조작 범위
    rotation_slider_range = (-0.15, 0.15)
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "arm_r_link7",
    )
    initial_target = data.xpos[body_id].copy()

    hand_pose_panel = GlfwTargetPanel(
        initial_target,
        slider_range=slider_range,
        rotation_slider_range=rotation_slider_range,
    )

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    window = glfw.create_window(1200, 900, "MuJoCo GLFW Demo", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(
        model,
        mujoco.mjtFontScale.mjFONTSCALE_150,
    )

    # mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)

    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 1.0]
    cam.distance = 3.0
    cam.azimuth = 90
    cam.elevation = -20

    try:
        while not glfw.window_should_close(window): # 렌더링 루프
            glfw.poll_events()
            target = hand_pose_panel.poll_target(window)
            if target is not None:
                print(target)

            step_start = time.time()

            while time.time() - step_start < 1.0 / 60.0: # 시뮬레이션 루프
                mujoco.mj_step(model, data)

            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)

            mujoco.mjv_updateScene(
                model,
                data,
                opt,
                None,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )

            mujoco.mjr_render(viewport, scene, context)
            hand_pose_panel.render(window, context)

            glfw.swap_buffers(window)

    finally:
        glfw.terminate()


if __name__ == "__main__":
    main()
