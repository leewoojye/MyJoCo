from typing import Callable

import numpy as np
import open3d.visualization.gui as gui


class GraspPanel:
    def __init__(
        self,
        # 초기 alpha와 엄지 여부를 나타내는 변수
        initial_grasp,
        isThumb=False,
        # alpha(np.ndarray), isThumb(bool)을 입력으로 받음
        on_grasp_changed: Callable[[np.ndarray, bool], None] | None = None,
        slider_range=(0.0, 1.0),
    ):
        self.base_alpha = np.asarray(initial_grasp, dtype=float).reshape(2).copy()
        self.offset = np.zeros(2)
        self.alpha = self.base_alpha.copy()
        self.isThumb = isThumb
        self.on_grasp_changed = on_grasp_changed
        self.widget = gui.Vert(6, gui.Margins(8, 8, 8, 8))
        self._sliders = []
        self._value_labels = []

        self.widget.add_child(gui.Label("Right hand pose target"))
        for index, axis_name in enumerate(("thumb", "finger")):
            self._add_axis_slider(index, axis_name, slider_range)

    def _add_axis_slider(self, index, axis_name, slider_range):
        row = gui.Horiz(6)
        label = gui.Label(axis_name)
        value_label = gui.Label("0.000")
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(float(slider_range[0]), float(slider_range[1]))
        slider.double_value = 0.0
        slider.set_on_value_changed(
            lambda value, axis_index=index: self._set_axis(axis_index, value)
        )

        row.add_child(label)
        row.add_child(slider)
        row.add_child(value_label)
        self.widget.add_child(row)
        self._sliders.append(slider)
        self._value_labels.append(value_label)

    def _set_axis(self, index, value):
        self.offset[index] = float(value)
        self.alpha = self.base_alpha + self.offset
        self.isThumb = index == 0 # 패널에서 첫 인덱스는 thumb을 조종함
        self._value_labels[index].text = f"{value:.3f}"
        if self.on_grasp_changed is not None:  # 움직이면 있으면 callback 함수를 호출
            self.on_grasp_changed(self.alpha.copy(), self.isThumb)

    def set_target(self, target_alpha, isThumb, notify=False):
        self.base_alpha[:] = np.asarray(target_alpha, dtype=float).reshape(2)
        self.offset[:] = 0.0
        self.alpha = self.base_alpha.copy()
        self.isThumb = isThumb
        for index, slider in enumerate(self._sliders):
            slider.double_value = 0.0
            self._value_labels[index].text = "0.000"

        if notify and self.on_grasp_changed is not None:
            self.on_grasp_changed(self.alpha.copy(), self.isThumb)

    # def __init__(self, on_grasp_changed: Callable | None = None):
    #     pass

    # def build_widget(self):
    #     pass

    # def set_grasp_command(self, command, notify=False):
    #     pass

    # def get_grasp_command(self):
    #     pass

    # def _set_thumb(self, value):
    #     pass

    # def _set_grasp(self, value):
    #     pass
