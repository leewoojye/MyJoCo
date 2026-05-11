import numpy as np


def apply_translation(robot, body_name, displacement):
    body_node = robot.body_node_for(body_name)
    joint = body_node.joints[0]
    addr = joint["qpos_addr"]

    # qvel update는 tick/controller에서 실제 적용된 qpos 차분과 dt로 처리
    robot.state.qpos[addr : addr + 3] += displacement

    delta = np.eye(4)
    delta[:3, 3] = displacement

    for node in body_node.iter_nodes():
        for record in node.all_records():
            record.mesh.transform(delta)

        node.world_transform = delta @ node.world_transform


# update_body_qvel_from_displacement
def update_qvel(robot, body_name, displacement, dt):
    if dt <= 0:
        return

    body_node = robot.body_node_for(body_name)
    joint = body_node.joints[0]
    addr = joint["qpos_addr"]

    robot.state.qvel[addr : addr + 3] = displacement / dt


# 시간에 따라 힘->가속도->속도를 누적해(적분) 상태를 업데이트
def integrate_force(robot, body_name, force, mass, dt):
    force = np.asarray(force, dtype=float).reshape(3).copy()
    force_norm = np.linalg.norm(force)

    if mass <= 0 or dt <= 0 or force_norm <= 0:
        return np.zeros(3)

    object_acc = force / mass
    object_displacement = 0.5 * object_acc * dt**2
    apply_translation(robot, body_name, object_displacement) # qpos 및 mesh transform 갱신
    update_qvel(robot, body_name, object_displacement, dt)

    return object_displacement
