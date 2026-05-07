from typing import Callable

import numpy as np
import open3d.visualization.gui as gui


class JointPanel:
    def __init__(
        self,
        initial_target,
        on_target_changed: Callable[[np.ndarray], None] | None = None,
        slider_range=(0.0, 1.0),
    ):
        self.base_target = np.asarray(initial_target, dtype=float).reshape(3).copy()
        self.offset = np.zeros(3)
        self.target = self.base_target.copy()
        self.on_target_changed = on_target_changed
        self.widget = gui.Vert(6, gui.Margins(8, 8, 8, 8))
        self._sliders = []
        self._value_labels = []

        self.widget.add_child(gui.Label("Hand grasp target"))
        for index, axis_name in enumerate(("Right thumb", "Right grasp")):
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
        self.target = self.base_target + self.offset
        self._value_labels[index].text = f"{value:.3f}"
        if self.on_target_changed is not None:
            self.on_target_changed(self.target.copy())

    def set_target(self, target, notify=False):
        self.base_target[:] = np.asarray(target, dtype=float).reshape(3)
        self.offset[:] = 0.0
        self.target = self.base_target.copy()
        for index, slider in enumerate(self._sliders):
            slider.double_value = 0.0
            self._value_labels[index].text = "0.000"

        if notify and self.on_target_changed is not None:
            self.on_target_changed(self.target.copy())
