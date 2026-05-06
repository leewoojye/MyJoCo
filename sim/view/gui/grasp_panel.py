from typing import Callable


class GraspPanel:
    def __init__(self, on_grasp_changed: Callable | None = None):
        pass

    def build_widget(self):
        pass

    def set_grasp_command(self, command, notify=False):
        pass

    def get_grasp_command(self):
        pass

    def _set_thumb(self, value):
        pass

    def _set_grasp(self, value):
        pass
