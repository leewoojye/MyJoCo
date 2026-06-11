import glfw
import mujoco
import numpy as np


class GUIPanel:
    AXES = ("azimuth", "elevation", "distance", "lookat_z")
    LABEL_WIDTH = 105
    LABEL_HEIGHT = 28
    TEXT_CHAR_WIDTH = 10
    TEXT_BASELINE_OFFSET = 2

    def __init__(
        self,
        initial_camera,  # initial_camera = [azimuth, elevation, distance, lookat_z]
        azimuth_range=(90.0, 270.0),
        elevation_range=(-90.0, 20.00),
        distance_range=(1.0, 5.0),
        lookat_range=(0.0, 2.0),
    ):
        self.target = np.asarray(initial_camera, dtype=float).reshape(len(self.AXES))
        self.ranges = [azimuth_range, elevation_range, distance_range, lookat_range]
        self.drag_index = None
        self.changed = False

    def poll_camera(self, window):
        mouse_x, mouse_y = glfw.get_cursor_pos(window)
        win_w, win_h = glfw.get_window_size(window)
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        mouse_x = mouse_x * fb_w / win_w
        mouse_y = (win_h - mouse_y) * fb_h / win_h

        pressed = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS

        if pressed:
            if self.drag_index is None:
                self.drag_index = self._hit_slider(mouse_x, mouse_y)
            if self.drag_index is not None:
                self._set_value(self.drag_index, mouse_x)
        else:
            self.drag_index = None

        if not self.changed:
            return None

        self.changed = False
        return self.target.copy()

    # def poll_target(self, window):
    #     return self.poll_camera(window)

    def render(self, window, context):
        width, height = glfw.get_framebuffer_size(window)

        for i, axis in enumerate(self.AXES):
            y = height - 50 - i * 35
            x = max(20, width - 610)

            mujoco.mjr_rectangle(
                mujoco.MjrRect(x, y, self.LABEL_WIDTH, self.LABEL_HEIGHT),
                0.1,
                0.1,
                0.1,
                0.8,
            )
            text_width = len(axis) * self.TEXT_CHAR_WIDTH
            text_x = x + (self.LABEL_WIDTH - text_width) / 2
            text_y = y + self.TEXT_BASELINE_OFFSET
            mujoco.mjr_text(
                mujoco.mjtFont.mjFONT_NORMAL,
                axis,
                context,
                text_x / max(width, 1),
                text_y / max(height, 1),
                1.0,
                1.0,
                1.0,
            )

            mujoco.mjr_rectangle(
                mujoco.MjrRect(x + 110, y + 10, 160, 5),
                0.4,
                0.4,
                0.4,
                1.0,
            )

            lo, hi = self.ranges[i]
            t = (self.target[i] - lo) / (hi - lo)
            knob_x = int(x + 110 + t * 160)

            mujoco.mjr_rectangle(
                mujoco.MjrRect(knob_x - 4, y + 2, 8, 20),
                1.0,
                0.6,
                0.2,
                1.0,
            )

    def _hit_slider(self, mouse_x, mouse_y):
        width, height = glfw.get_framebuffer_size(glfw.get_current_context())

        for i in range(len(self.AXES)):
            y = height - 50 - i * 35
            x = max(20, width - 610)
            if x + 110 <= mouse_x <= x + 270 and y <= mouse_y <= y + 28:
                return i

        return None

    def _set_value(self, index, mouse_x):
        width, _ = glfw.get_framebuffer_size(glfw.get_current_context())
        x = max(20, width - 610)

        lo, hi = self.ranges[index]
        t = np.clip((mouse_x - (x + 110)) / 160, 0.0, 1.0)
        value = lo + t * (hi - lo)

        self.target[index] = value
        self.changed = True
