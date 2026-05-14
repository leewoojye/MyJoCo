import numpy as np

from sim.model.solver.force_closure import evaluate_grasp_state


# 물체-로봇손 접촉에 관여하는 접촉점들만 필터링
def object_body_contacts(contacts, object_body_name):
    object_contacts = []

    for contact in contacts or []:
        if contact.normal is None:
            continue

        if contact.record_a.body_name == object_body_name:
            contact.normal = -contact.normal # 힘을 받는 주체를 물체로 통일하기 위함
            object_contacts.append(contact)
        elif contact.record_b.body_name == object_body_name:
            object_contacts.append(contact)

    return object_contacts


# evaluate_grasp_state() wrapper
def evaluate_grasp_hold(robot, object_body_name, object_contacts, object_mass, friction_coefficient):
    object_center = robot.body_node_for(object_body_name).world_transform[:3, 3]
    gravity_force = np.array([0.0, 0.0, -object_mass * 9.8])
    gravity_wrench = np.r_[np.cross(object_center, gravity_force), gravity_force]

    return evaluate_grasp_state(
        contact_points=object_contacts,
        external_wrench=gravity_wrench,
        friction_coefficient=friction_coefficient,
    )


def update_grasp_state(
    robot,
    object_name,
    body_name,
    object_contacts,
    grasp_changed,
    is_grasped,
    offset,
    object_mass,
    friction_coefficient,
):
    if object_contacts and grasp_changed and not is_grasped:
        next_can_grasp, _ = evaluate_grasp_hold(
            robot,
            object_name,
            object_contacts,
            object_mass,
            friction_coefficient,
        )

        if next_can_grasp:
            object_center = robot.body_node_for(object_name).world_transform[:3, 3]
            body_pos = robot.body_node_for(body_name).world_transform[:3, 3]
            return True, object_center - body_pos

        return False, None

    if grasp_changed and not is_grasped: # 이미 잡은 상태이고, 움직이면 안되는 상황
        return False, None

    if not is_grasped:
        return False, None

    return is_grasped, offset
