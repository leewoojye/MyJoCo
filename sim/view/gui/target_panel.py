from typing import Callable

import numpy as np
import open3d.visualization.gui as gui


class TargetPanel:
    def __init__(
        self,
        initial_target,  # 3차원 초기 위치
        on_target_changed: Callable[[np.ndarray], None] | None = None,
        slider_range=(-1.0, 1.0),
        rotation_slider_range=(-0.3, 0.3),
    ):
        self.base_target = np.r_[
            np.asarray(initial_target, dtype=float).reshape(3), np.zeros(3)
        ]  # dx, dy, dz, roll, pitch, yaw
        # self.base_target[3:] = np.zeros(3)
        self.offset = np.zeros(6)
        self.target = self.base_target.copy()
        self.on_target_changed = on_target_changed
        self.widget = gui.Vert(6, gui.Margins(8, 8, 8, 8))
        self._sliders = []
        self._value_labels = []
        self._is_setting_target = False

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
        slider.set_on_value_changed(lambda value, axis_index=index: self._set_axis(axis_index, value))

        row.add_child(label)
        row.add_child(slider)
        row.add_child(value_label)
        self.widget.add_child(row)
        self._sliders.append(slider)
        self._value_labels.append(value_label)

    def _set_axis(self, index, value):
        self.offset[index] = float(value)
        self.target = self.base_target + self.offset  # ik solver가 받을 하위목표지점 계산
        self._value_labels[index].text = f"{value:.3f}"
        if self.on_target_changed is not None and not self._is_setting_target:  # 움직이면 있으면 callback 함수를 호출
            self.on_target_changed(self.target.copy())

    # 프로그램 내부에서 패널UI의 타겟을 강제로 설정할 때 사용
    # self-collision solver의 부품으로 사용될 수 있음
    def set_target(self, target, notify=False, reset_base=True):
        self._is_setting_target = True
        try:
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
        finally:
            self._is_setting_target = False

        if notify and self.on_target_changed is not None:
            self.on_target_changed(self.target.copy())
