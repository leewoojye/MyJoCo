import mujoco


def actuator_ids_from_joints(model, joint_ids):
    joint_id_set = set(joint_ids)
    actuator_ids = []

    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue

        joint_id = model.actuator_trnid[actuator_id, 0]
        if joint_id in joint_id_set:
            actuator_ids.append(actuator_id)

    return actuator_ids


def joint_ids_from_actuators(model, actuator_ids):
    joint_ids = []

    for actuator_id in actuator_ids:
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue

        joint_ids.append(model.actuator_trnid[actuator_id, 0])

    return joint_ids
