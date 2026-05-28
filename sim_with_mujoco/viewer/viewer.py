import glfw
import mujoco

from sim_with_mujoco.viewer.glfw_panel import GlfwTargetPanel


# mujoco viewer + target panel 통합
class Viewer:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def init_viewer(self, initial_target_pos, slider_range=(-0.2, 0.2), rotation_slider_range=(-0.15, 0.15)):
        self.hand_pose_panel = GlfwTargetPanel(
            initial_target_pos,
            slider_range=slider_range,
            rotation_slider_range=rotation_slider_range,
        )

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        self.window = glfw.create_window(1200, 900, "MyJoCo", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # 창 갱신을 모니터 refresh 주기에 맞춤

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(
            self.model,
            mujoco.mjtFontScale.mjFONTSCALE_150,
        )

        mujoco.mjv_defaultOption(self.opt)
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0, 0, 1.0]
        self.cam.distance = 3.0
        self.cam.azimuth = 90
        self.cam.elevation = -20

    # poll_target wrapper
    def poll_target(self):
        return self.hand_pose_panel.poll_target(self.window)

    def render(self):
        self.width, self.height = glfw.get_framebuffer_size(self.window)
        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

        # 다음 화면을 그려 버퍼에 저장
        mujoco.mjr_render(self.viewport, self.scene, self.context)
        self.hand_pose_panel.render(self.window, self.context)

        # GLFW/OpenGL - double buffering
        # render()가 그린 화면을 비로소 창에 띄움
        glfw.swap_buffers(self.window)

    def terminate_viewer(self):
        glfw.terminate()
