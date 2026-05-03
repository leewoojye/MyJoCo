from enum import Enum

import mujoco


class JointType(Enum):
    FREE = "free"
    BALL = "ball"
    SLIDE = "slide"
    HINGE = "hinge"
    UNKNOWN = "unknown"


def get_mujoco_name(model, obj_type, obj_id, fallback):
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else fallback


def joint_type_from_mujoco(joint_type):
    joint_type = int(joint_type)
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return JointType.FREE
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return JointType.BALL
    if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
        return JointType.SLIDE
    if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
        return JointType.HINGE
    return JointType.UNKNOWN


def qpos_width_for_joint_type(joint_type):
    if not isinstance(joint_type, JointType):
        joint_type = joint_type_from_mujoco(joint_type)

    if joint_type == JointType.FREE:
        return 7
    if joint_type == JointType.BALL:
        return 4
    return 1


def create_joint_record(model, data, joint_id):
    dof_id = int(model.jnt_dofadr[joint_id])
    joint_type = joint_type_from_mujoco(model.jnt_type[joint_id])
    joint = {
        "joint_id": joint_id,
        "name": get_mujoco_name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
            f"joint_{joint_id}",
        ),
        "type": joint_type,
        "axis": model.jnt_axis[joint_id].copy(),
        "pos": model.jnt_pos[joint_id].copy(),
        "world_axis": data.xaxis[joint_id].copy(),
        "world_pos": data.xanchor[joint_id].copy(),
        "range": model.jnt_range[joint_id].copy(),
        "limited": bool(model.jnt_limited[joint_id]),
        "qpos_addr": int(model.jnt_qposadr[joint_id]),
        "dof_addr": dof_id,
    }

    if dof_id >= 0:
        joint["dof"] = {
            "armature": float(model.dof_armature[dof_id]),
            "damping": float(model.dof_damping[dof_id]),
            "frictionloss": float(model.dof_frictionloss[dof_id]),
        }

    return joint
