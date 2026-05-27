import numpy as np
import open3d.visualization.gui as gui


class TargetPanel:
    def __init__(
        self,
        initial_target,  # 3차원 초기 위치
        slider_range=(-1.0, 1.0),
        rotation_slider_range=(-0.3, 0.3),
    ):
        self.base_target = np.r_[
            np.asarray(initial_target, dtype=float).reshape(3), np.zeros(3)
        ]  # dx, dy, dz, roll, pitch, yaw
        # self.base_target[3:] = np.zeros(3)
        self.offset = np.zeros(6)
        self.target = self.base_target.copy()
        self.widget = gui.Vert(6, gui.Margins(8, 8, 8, 8))
        self._sliders = []
        self._value_labels = []
        self._poll_pending = False

        self.widget.add_child(gui.Label("Right hand pose target"))
        for index, axis_name in enumerate(("dX", "dY", "dZ", "Roll", "Pitch", "Yaw")):
            axis_slider_range = slider_range if index < 3 else rotation_slider_range
            self._add_axis_slider(index, axis_name, axis_slider_range)

        # for index, axis_name in enumerate(("Roll", "Pitch", "Yaw")):
        #     self._add_axis_slider(index, axis_name, slider_range)

    def _add_axis_slider(self, index, axis_name, slider_range):
        row = gui.Horiz(6)
        label = gui.Label(axis_name)
        value_label = gui.Label("0.000")
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(float(slider_range[0]), float(slider_range[1]))
        slider.double_value = 0.0

        row.add_child(label)
        row.add_child(slider)
        row.add_child(value_label)
        self.widget.add_child(row)
        self._sliders.append(slider)
        self._value_labels.append(value_label)

    def poll_target(self):
        offset = np.array([slider.double_value for slider in self._sliders], dtype=float)
        changed = self._poll_pending or not np.allclose(offset, self.offset)
        if not changed:
            return None

        self._poll_pending = False
        self.offset[:] = offset
        self.target = self.base_target + self.offset  # ik solver가 받을 하위목표지점 계산
        for index, value in enumerate(self.offset):
            self._value_labels[index].text = f"{value:.3f}"

        return self.target.copy()

    # 프로그램 내부에서 패널UI의 타겟을 강제로 설정할 때 사용
    # self-collision solver의 부품으로 사용될 수 있음
    def set_target(self, target, notify=False, reset_base=True):
        target = np.asarray(target, dtype=float).reshape(6)
        if reset_base:
            self.base_target[:] = target
            self.offset[:] = 0.0
        else:
            self.offset[:] = target - self.base_target

        self.target = self.base_target + self.offset
        for index, slider in enumerate(self._sliders):
            slider.double_value = self.offset[index]
            self._value_labels[index].text = f"{self.offset[index]:.3f}"

        if notify:
            self._poll_pending = True
