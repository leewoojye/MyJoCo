import numpy as np
import open3d as o3d
import time

from sim.model.collision.collision_check import check_collision
from sim.model.grasping.contact_constraint import (
    apply_body_translation,
    sum_contact_force,
)
from sim.model.kinematics.fk import apply_fk
from sim.model.kinematics.ik import apply_ik
from sim.model.grasping.force_closure import evaluate_grasp_state
from sim.model.grasping.form_closure import compute_grasp
from sim.model.kinematics.jacobian import compute_body_twist, compute_geometric_jacobian
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

    # 새로운 state 인스턴스 생성 예시 (omy.xml)
    state = robot.state  # robot state 필드 참조, set()으로 수정 가능
    # state_omy = RobotState.from_model(robot.model) # robot 클래스 state와 독립적인 state 인스턴스 생성
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

    state.set("finger_l_joint1", 0.3)
    state.set("finger_l_joint2", 1.57)
    state.set("finger_l_joint3", -0.35)
    state.set("finger_l_joint4", -0.25)

    state.set("finger_r_joint1", 0.3)
    state.set("finger_r_joint2", -1.57)
    state.set("finger_r_joint3", 0.35)
    state.set("finger_r_joint4", 0.25)

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
    trajectory_duration = 0.1  # 단일 궤적 시간 고정
    object_body_name = "pr_cokeCan"  # 상호작용할 물체 지정 (추후 리펙토링)
    friction_coefficient = (
        0.2  # force closure solver 입장에서는 마찰계수가 커질수록 가능한 접촉 힘 cone이 넓어져서 grasp 성공에 용이
    )

    # hand grasp 관련 변수
    grasp_alpha = np.zeros(2)
    grasp_goal = None  # grasp panel에서 들어온 목표 alpha
    grasp_goal_is_thumb = False  # 엄지인지 나머지 네마디인지
    can_grasp = False  # grasp 여부를 나타내는 flag 변수
    offset = None  # 캔을 잡은 상태에서 캔 중심과 손 기준점 사이의 거리, 손과 캔의 변위를 같게 유지하기 위함

    # dynamics 상태 관리
    dynamics_dict = {
        "pr_cokeCan": {
            "mass": 0.35,
            "velocity": np.zeros(3),
            "acceleration": np.zeros(3),
            "force": np.zeros(3),
            "is_grasped": False,
        }
    }

    # callback 함수
    # 패널입력, 시간을 받아 궤적 생성 -> ...
    def handle_target_changed(target_pos):
        nonlocal target_goal
        # trajectory_start = robot.body_node_for("hx5_r_base").world_transform[:3, 3].copy()
        # trajectory_start = robot.body_node_for(right_target_body).world_transform[:3, 3].copy()

        # 타겟패널입력은 새로운 궤적 생성 타이밍에 관여하며, 입력을 감지하면 trajectory 관련 변수를 갱신함
        # trajectory_goal = target_pos.copy()
        # trajectory_start_time = time.perf_counter()
        target_goal = target_pos.copy()

    def handle_grasp_changed(alpha, is_thumb):
        nonlocal grasp_goal, grasp_goal_is_thumb
        grasp_goal = np.asarray(alpha, dtype=float).reshape(2)
        grasp_goal_is_thumb = is_thumb

    # 패널 감지 콜백함수가 궤적 생성 시점에 관여한다면, tick(유사 clock) 콜백함수가 로봇의 실제 ik 적용과 렌더링을 맡음
    def handle_tick():
        nonlocal \
            trajectory_start, \
            trajectory_goal, \
            target_goal, \
            last_tick_time, \
            trajectory_elapsed, \
            grasp_goal, \
            grasp_goal_is_thumb, \
            can_grasp, \
            offset, \
            dynamics_dict

        if target_goal is None and grasp_goal is None:
            return False

        grasp_changed = grasp_goal is not None
        prev_qpos = robot.state.qpos.copy()  # 초기 qpos: rollback이나 qvel을 구할 때 필요함
        current_hand_pos = robot.body_node_for(right_target_body).world_transform[:3, 3].copy()
        current_tick_time = time.perf_counter()

        if last_tick_time is None:
            last_tick_time = current_tick_time

        raw_dt = current_tick_time - last_tick_time
        last_tick_time = current_tick_time
        physics_dt = min(raw_dt, 1.0 / 30.0)

        if target_goal is not None:
            if trajectory_goal is None:
                trajectory_start = current_hand_pos.copy()
                trajectory_goal = target_goal.copy()
                trajectory_elapsed = 0.0
            elif not np.allclose(trajectory_goal, target_goal):
                trajectory_start = current_hand_pos.copy()
                trajectory_goal = target_goal.copy()
                trajectory_elapsed = 0.0

            trajectory_elapsed += raw_dt

            # elapsed = time.perf_counter() - trajectory_start_time
            # t = min(elapsed, trajectory_duration)
            t = min(trajectory_elapsed, trajectory_duration)
            target_pos = interpolate_position(trajectory_start, trajectory_goal, trajectory_duration, t)

        candidate_state = RobotState(
            robot.state.joint_names,
            robot.state.qpos_addrs,
            robot.state.qpos_widths,
            robot.state.qpos.copy(),
        )

        if target_goal is not None:
            # 오른손에 pose IK 계산, 캔 잡는 모션을 더 용이하게
            apply_ik(
                robot,
                candidate_state,
                target_pos,
                home_poses,
                "pose",  # "pose"
                right_target_body,
            )

            # 왼손은 position IK 계산
            apply_ik(
                robot,
                candidate_state,
                left_hand_pos,
                home_poses,
                "position",
                left_target_body,
            )

        if grasp_goal is not None and not can_grasp:
            next_alpha = float(grasp_goal[0] if grasp_goal_is_thumb else grasp_goal[1])
            compute_grasp(robot, candidate_state, next_alpha, grasp_goal_is_thumb)  # candidate_state

        # 접촉점 상대속도 기반 (강체 자체 상대속도 아님)
        # 흐름도: 상대속도로 각 접촉점에 가해지는 힘 계산(힘 크기는 고정시킨 상태)->계산된 힘 벡터 합산->물체의 질량으로 가속도 및 속도와 변위까지 계산
        # v_contact = v_body + w × (p - center) # 회전 고려
        # v_contact = v_body # 회전 무시

        is_collision, is_contact, contacts = check_collision(
            robot,
            candidate_state,
            home_poses,
            return_contacts=False,
        )

        if is_collision and not is_contact:
            trajectory_goal = None
            trajectory_start = None
            # trajectory_start_time = None
            target_goal = None
            last_tick_time = None
            trajectory_elapsed = 0.0
            grasp_goal = None
            return current_hand_pos.copy()

        robot.state.qpos[:] = candidate_state.qpos

        if grasp_goal is not None:
            if not can_grasp:
                grasp_alpha[:] = grasp_goal
            grasp_goal = None

        # 충돌 판단 직후 관절 속도 업데이트
        if raw_dt > 0:
            for joint_name, addr in robot.state.qpos_addrs.items():
                width = robot.state.qpos_widths[joint_name]
                robot.state.qvel[addr : addr + width] = (
                    robot.state.qpos[addr : addr + width] - prev_qpos[addr : addr + width]
                ) / raw_dt

        # FK 적용: 왼손, 오른손, 손가락 동시에
        apply_fk(robot, robot.state, home_poses)

        # 트위스트 업데이트 시점: 트위스트는 현재 qpos에 대한 자코비안 행렬과 관절 속도의 곱으로 표현되므로 qpos를 적용한 이후 업데이트함
        robot.state.body_twists.clear()
        for node in robot.root_body.iter_nodes():
            body_name = node.name

            if body_name == object_body_name:
                joint = node.joints[0]
                addr = joint["qpos_addr"]
                robot.state.body_twists[body_name] = np.r_[np.zeros(3), robot.state.qvel[addr : addr + 3]]
            elif body_name in {"world", "base_table"}:
                robot.state.body_twists[body_name] = np.zeros(6)
            elif body_name in {"arm_r_link7", "hx5_r_base"} or body_name.startswith("finger_r_link"):
                robot.state.body_twists[body_name] = compute_body_twist(
                    robot,
                    robot.state,
                    target_body=body_name,
                )

        # 접촉 bodynode에 기반해 캔의 변위 설정
        if can_grasp:
            contact_body_pos = robot.body_node_for(right_target_body).world_transform[:3, 3].copy()
            body_node = robot.body_node_for(object_body_name)
            object_pos = body_node.world_transform[:3, 3].copy()

            if offset is None:
                offset = object_pos - contact_body_pos

            object_displacement = contact_body_pos + offset - object_pos
            apply_body_translation(robot, object_body_name, object_displacement)

            if raw_dt > 0:
                joint = body_node.joints[0]
                addr = joint["qpos_addr"]
                robot.state.qvel[addr : addr + 3] = object_displacement / raw_dt

        contacts = None
        if is_contact or can_grasp:
            _, _, contacts = check_collision(
                robot,
                robot.state,
                home_poses,
                return_contacts=True,
            )

        # robot-object 사이 접촉점 순회
        object_contacts = []
        for contact in contacts or []:
            if contact.normal is None:
                continue

            if contact.record_a.body_name == object_body_name:
                contact.normal = -contact.normal
                object_contacts.append(contact)
            elif contact.record_b.body_name == object_body_name:
                object_contacts.append(contact)

        # 접촉점에서의 상대속도 계산은 compute_contact_force_sum() 내부에서 수행

        if object_contacts and grasp_changed and not can_grasp:  # grasp panel에 변화가 있을 때만
            object_mass = dynamics_dict[object_body_name]["mass"]
            object_center = robot.body_node_for(object_body_name).world_transform[:3, 3]
            gravity_force = np.array([0.0, 0.0, -object_mass * 9.81])
            # gravity_force = np.array([0.0, 0.0, 0.0]) # 작은 외력은 closure 임계점을 많이 낮추는 문제
            gravity_wrench = np.r_[np.cross(object_center, gravity_force), gravity_force]

            next_can_grasp, _ = evaluate_grasp_state(
                object_contacts,
                gravity_wrench,
                friction_coefficient,
            )

            can_grasp = next_can_grasp
            if can_grasp:
                contact_body_pos = robot.body_node_for(right_target_body).world_transform[:3, 3]
                offset = object_center - contact_body_pos
            else:
                offset = None
        elif grasp_changed and not can_grasp:
            can_grasp = False
            offset = None
        elif not can_grasp:
            offset = None

        dynamics_dict[object_body_name]["is_grasped"] = can_grasp

        if object_contacts and not can_grasp:
            object_force = sum_contact_force(  # 물체에 가해지는 힘 벡터를 합산
                contact_points=object_contacts,
            )

            # 힘 벡터가 테이블과 평행하도록 클리핑 (추후 수정)
            object_force[2] = 0.0
            force_norm = np.linalg.norm(object_force)

            if force_norm > 0:
                # 가속도를 적분하여 변위 계산
                object_acc = object_force / 0.35  # mass: 0.35
                # object_vel = object_acc * dt
                object_displacement = 0.5 * object_acc * physics_dt**2
                apply_body_translation(robot, object_body_name, object_displacement)

                if physics_dt > 0:
                    body_node = robot.body_node_for(object_body_name)
                    joint = body_node.joints[0]
                    addr = joint["qpos_addr"]
                    robot.state.qvel[addr : addr + 3] = object_displacement / physics_dt

        if trajectory_elapsed >= trajectory_duration:
            trajectory_start = None
            trajectory_goal = None
            # trajectory_start_time = None
            target_goal = None
            last_tick_time = None
            trajectory_elapsed = 0.0

        return True

    run_target_window(  # body frame 기준 joint/link 렌더링
        robot,
        on_target_changed=handle_target_changed,
        on_grasp_changed=handle_grasp_changed,
        on_tick=handle_tick,
    )  # e.e pose, grasp, tick 콜백 함수 모두 전달


if __name__ == "__main__":
    main()
