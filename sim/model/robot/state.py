import mujoco
import numpy as np

from sim.model.robot.joint import get_mujoco_name, qpos_width_for_joint_type


def initial_qpos_from_keyframe(model, keyframe_name="home"):
    if model.nkey == 0:
        return model.qpos0.copy()

    key_id = 0
    for candidate_id in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, candidate_id)
        if name == keyframe_name:
            key_id = candidate_id
            break

    return model.key_qpos[key_id].copy()


class RobotState:
    """Current robot joint state, separated from the body tree structure."""

    def __init__(self, joint_names, qpos_addrs, qpos_widths, initial_qpos):
        self.joint_names = list(joint_names)
        self.joint_index = {name: i for i, name in enumerate(self.joint_names)}
        self.qpos_addrs = dict(qpos_addrs)
        self.qpos_widths = dict(qpos_widths)
        self.qpos = np.asarray(initial_qpos, dtype=float).copy()

    @classmethod
    def from_model(cls, model, keyframe_name="home"):
        joint_names = []
        qpos_addrs = {}
        qpos_widths = {}

        for joint_id in range(model.njnt):
            joint_name = get_mujoco_name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
                f"joint_{joint_id}",
            )
            joint_names.append(joint_name)
            qpos_addrs[joint_name] = int(model.jnt_qposadr[joint_id])
            qpos_widths[joint_name] = qpos_width_for_joint_type(
                model.jnt_type[joint_id]
            )

        return cls(
            joint_names=joint_names,
            qpos_addrs=qpos_addrs,
            qpos_widths=qpos_widths,
            initial_qpos=initial_qpos_from_keyframe(model, keyframe_name),
        )

    def get(self, joint_name):
        qpos = self.get_qpos(joint_name)
        if qpos.size == 1:
            return float(qpos[0])
        return qpos

    def set(self, joint_name, value):
        addr = self.qpos_addrs[joint_name]
        width = self.qpos_widths[joint_name]
        values = np.asarray(value, dtype=float).reshape(-1)

        if values.size != width:
            raise ValueError(
                f"{joint_name} expects {width} qpos value(s), got {values.size}"
            )

        self.qpos[addr : addr + width] = values

    def get_qpos(self, joint_name):
        addr = self.qpos_addrs[joint_name]
        width = self.qpos_widths[joint_name]
        return self.qpos[addr : addr + width].copy()

    def as_dict(self):
        return {joint_name: self.get(joint_name) for joint_name in self.joint_names}
