import numpy as np
import open3d as o3d
import time

from sim.model.collision.collision_check import collision_check
from sim.model.grasping.contact_constraint import (
    apply_body_translation,
    compute_contact_force_sum,
)
from sim.model.kinematics.fk import apply_fk, compute_fk
from sim.model.kinematics.ik import apply_ik
from sim.model.grasping.form_closure import calculate_grasp
from sim.model.motion.trajectory import interpolate_position
from sim.model.robot.loader_with_mujoco import build_robot_geometries
from sim.model.robot.robot_state import RobotState
from sim.view.renderer import run_target_window


def create_grid(size=4.0, z=0.002, spacing=0.25, color=(0.55, 0.57, 0.60)):
    half = size / 2
    coords = np.arange(-half, half + spacing * 0.5, spacing)

    points = []
    lines = []
    for coord in coords:
        start = len(points)
        points.extend([[-half, coord, z], [half, coord, z]])
        lines.append([start, start + 1])

        start = len(points)
        points.extend([[coord, -half, z], [coord, half, z]])
        lines.append([start, start + 1])

    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    grid.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return grid


def create_background_geometries(model, data):
    return [
        create_grid(),
    ]


def draw_scene(robot_geometries):
    scene_geometries = [
        *create_background_geometries(robot_geometries.model, robot_geometries.data),
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25),
        *robot_geometries.open3d_geometries(),
    ]

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="MyJoCo")
    for geometry in scene_geometries:
        vis.add_geometry(geometry)

    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0.78, 0.82, 0.88])
    render_option.mesh_show_back_face = True
    render_option.line_width = 1.0

    vis.run()
    vis.destroy_window()


def main():
    robot = build_robot_geometries()

    # home configuration 저장
    home_poses = {}
    for node in robot.root_body.iter_nodes():
        home_poses[node.name] = node.world_transform.copy()

    # robot 클래스 state와 독립적인 state 인스턴스 생성
    # state = RobotState.from_model(robot.model)

    # 새로운 state 인스턴스 생성 예시 (omy.xml)
    state = robot.state  # robot state 필드 참조, set()으로 수정 가능
    # state_omy = RobotState.from_model(robot.model)
    # state_omy.set("Joint1", 0.5)
    # state_omy.set("Joint2", -0.4)
    # state_omy.set("Joint3", 0.7)
    # state_omy.set("Joint4", -0.6)
    # state_omy.set("Joint5", 0.3)
    # state_omy.set("Joint6", 0.2)
    # state_omy.set("rh_r1", 0.3)
    # state_omy.set("rh_r2", -0.3)
    # state_omy.set("rh_l1", -0.3)
    # state_omy.set("rh_l2", 0.3)

    # 관절에 대한 자세만 지정함, 관절의 자세가 하위 링크/바디의 자세 결정
    state.set("lift_joint", -0.15)

    state.set("head_joint1", 0.0)
    state.set("head_joint2", 0.0)

    state.set("arm_l_joint1", 0.0)
    state.set("arm_l_joint2", 0.0)
    state.set("arm_l_joint3", 0.0)
    state.set("arm_l_joint4", -1.57)
    state.set("arm_l_joint5", 0.0)
    state.set("arm_l_joint6", 0.0)
    state.set("arm_l_joint7", 0.0)

    state.set("arm_r_joint1", 0.0)
    state.set("arm_r_joint2", 0.0)
    state.set("arm_r_joint3", 0.0)
    state.set("arm_r_joint4", -1.57)
    state.set("arm_r_joint5", 0.0)
    state.set("arm_r_joint6", 0.0)
    state.set("arm_r_joint7", 0.0)

    state.set("finger_l_joint1", 0.0)
    state.set("finger_l_joint2", 2.09)
    state.set("finger_l_joint3", 0.0)
    state.set("finger_l_joint4", 0.0)

    state.set("finger_r_joint1", 0.0)
    state.set("finger_r_joint2", -2.09)
    state.set("finger_r_joint3", 0.0)
    state.set("finger_r_joint4", 0.0)

    robot = apply_fk(robot, robot.state, home_poses)
    home_poses = {node.name: node.world_transform.copy() for node in robot.root_body.iter_nodes()}
    home_poses["_qpos"] = robot.state.qpos.copy()
    right_target_body = "arm_r_link7"
    left_target_body = "arm_l_link7"
    left_hand_pos = (
        robot.body_node_for(left_target_body).world_transform[:3, 3].copy()
    )  # position IK로 위치를 고정할 왼손 위치 추출

    target_goal = None  # target panel에서 주어진 목표

    # 로봇팔 궤적 관련 변수
    trajectory_start = None
    trajectory_goal = None  # 형성된 단일 궤적의 목표
    # trajectory_start_time = None
    last_tick_time = None
    trajectory_elapsed = 0.0
    trajectory_duration = 1.0  # 단일 궤적 시간 고정
    object_body_name = "pr_cokeCan"

    # hand grasp 관련 변수
    grasp_alpha = np.zeros(2)
    isThumb = False

    # callback 함수
    # 패널입력, 시간을 받아 궤적 생성 -> ...
    def handle_target_changed(target_pos):
        # nonlocal trajectory_start, trajectory_goal, trajectory_start_time
        nonlocal target_goal
        # trajectory_start = robot.body_node_for("hx5_r_base").world_transform[:3, 3].copy()
        # trajectory_start = robot.body_node_for(right_target_body).world_transform[:3, 3].copy()

        # 타겟패널입력은 새로운 궤적 생성 타이밍에 관여하며, 입력을 감지하면 trajectory 관련 변수를 갱신함
        # trajectory_goal = target_pos.copy()
        # trajectory_start_time = time.perf_counter()
        target_goal = target_pos.copy()

    def handle_grasp_changed(alpha, isthumb):
        nonlocal isThumb
        next_grasp_alpha = np.asarray(alpha, dtype=float).reshape(2)
        current_alpha = float(grasp_alpha[0] if isthumb else grasp_alpha[1])
        next_alpha = float(next_grasp_alpha[0] if isthumb else next_grasp_alpha[1])

        candidate_state = RobotState(
            robot.state.joint_names,
            robot.state.qpos_addrs,
            robot.state.qpos_widths,
            robot.state.qpos.copy(),
        )
        calculate_grasp(robot, candidate_state, next_alpha, isthumb)

        is_collision, is_contact, _ = collision_check(
            robot,
            candidate_state,
            home_poses,
            return_contacts=False,
        )

        if next_alpha > current_alpha and is_collision and not is_contact:
            return

        grasp_alpha[:] = next_grasp_alpha
        isThumb = isthumb
        robot.state.qpos[:] = candidate_state.qpos
        apply_fk(robot, robot.state, home_poses)

    # 패널 감지 콜백함수가 궤적 생성 시점에 관여한다면, tick(유사 clock) 콜백함수가 로봇의 실제 ik 적용과 렌더링을 맡음
    def handle_tick():
        # nonlocal trajectory_start, trajectory_goal, trajectory_start_time, isThumb
        nonlocal trajectory_start, trajectory_goal, target_goal, last_tick_time, trajectory_elapsed, isThumb

        if target_goal is None:
            return False

        current_hand_pos = robot.body_node_for(right_target_body).world_transform[:3, 3].copy()
        now = time.perf_counter()

        if last_tick_time is None:
            last_tick_time = now

        dt = now - last_tick_time
        last_tick_time = now

        if trajectory_goal is None:
            trajectory_start = current_hand_pos.copy()
            trajectory_goal = target_goal.copy()
            trajectory_elapsed = 0.0
        elif not np.allclose(trajectory_goal, target_goal):
            trajectory_start = current_hand_pos.copy()
            trajectory_goal = target_goal.copy()
            trajectory_elapsed = 0.0

        trajectory_elapsed += dt

        # elapsed = time.perf_counter() - trajectory_start_time
        # t = min(elapsed, trajectory_duration)
        t = min(trajectory_elapsed, trajectory_duration)
        target_pos = interpolate_position(trajectory_start, trajectory_goal, trajectory_duration, t)

        # 오른손에 pose IK 적용
        # apply_ik(robot, robot.state, target_pos, home_poses, "pose")
        # # 왼손은 position IK 적용
        # # apply_ik(robot, robot.state, target_pos, home_poses, "position")
        # # 오른손 손가락
        # apply_grasp(robot, state, home_poses, selected_grasp_alpha(), isThumb)

        candidate_state = RobotState(
            robot.state.joint_names,
            robot.state.qpos_addrs,
            robot.state.qpos_widths,
            robot.state.qpos.copy(),
        )

        # current_hand_pos = robot.body_node_for("hx5_r_base").world_transform[:3, 3].copy()
        # current_hand_pos = robot.body_node_for(right_target_body).world_transform[:3, 3].copy()

        apply_ik(
            robot,
            candidate_state,
            target_pos,
            home_poses,
            "pose",
            right_target_body,
        )
        apply_ik(
            robot,
            candidate_state,
            left_hand_pos,
            home_poses,
            "position",
            left_target_body,
        )

        # e.e 위치 변위
        candidate_poses = compute_fk(robot, candidate_state, home_poses)
        candidate_hand_pos = candidate_poses[right_target_body][:3, 3].copy()
        hand_displacement = candidate_hand_pos - current_hand_pos

        is_collision, is_contact, contacts = collision_check(
            robot,
            candidate_state,
            home_poses,
            return_contacts=True,
        )

        if is_collision and not is_contact:
            trajectory_goal = None
            trajectory_start = None
            # trajectory_start_time = None
            target_goal = None
            last_tick_time = None
            trajectory_elapsed = 0.0
            return False

        robot.state.qpos[:] = candidate_state.qpos

        # 오른손에 FK 적용
        apply_fk(robot, robot.state, home_poses)

        # 왼손은 position IK -> FK (closed-chain...)

        # 물체와의 접촉 감지 및 물체 이동 계산
        object_contacts = []
        for contact in contacts or []:
            if contact.normal is None:
                continue

            if contact.record_a.body_name == object_body_name:
                contact.normal = -contact.normal
                object_contacts.append(contact)
            elif contact.record_b.body_name == object_body_name:
                object_contacts.append(contact)

        if object_contacts:
            object_force = compute_contact_force_sum(
                object_contacts,
                hand_displacement,
            )
            force_norm = np.linalg.norm(object_force)

            if force_norm > 0:
                # e.e 위치 변위를 물체 이동에 반영 (추후 수정)
                object_displacement = np.linalg.norm(hand_displacement) * object_force / force_norm
                apply_body_translation(robot, object_body_name, object_displacement)

        # if elapsed >= trajectory_duration:
        if trajectory_elapsed >= trajectory_duration:
            trajectory_start = None
            trajectory_goal = None
            # trajectory_start_time = None
            target_goal = None
            last_tick_time = None
            trajectory_elapsed = 0.0

        return True

    # body frame 기준 joint/link 렌더링
    # e.e pose, grasp, tick 콜백 함수 모두 전달
    run_target_window(
        robot,
        on_target_changed=handle_target_changed,
        on_grasp_changed=handle_grasp_changed,
        on_tick=handle_tick,
    )  # callback으로 연결


if __name__ == "__main__":
    main()
