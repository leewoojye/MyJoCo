import numpy as np
import open3d as o3d
import time

from sim.model.collision.collision_check import check_collision
from sim.model.dynamics.integrator import (
    apply_pos,
    integrate_force,
    update_qvel,
)
from sim.model.solver.contact_constraint import (
    sum_contact_forces,
)
from sim.model.kinematics.fk import apply_fk
from sim.model.kinematics.ik import apply_ik
from sim.model.solver.form_closure import compute_grasp
from sim.model.solver.grasp_solver import object_body_contacts, update_grasp_state
from sim.model.kinematics.jacobian import update_body_twists
from sim.model.math3d.rotation import rpy2rotation_matrix
from sim.model.motion.trajectory import interpolate_position_cubic
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
    right_target_base_rot = (
        robot.body_node_for(right_target_body).world_transform[:3, :3].copy()
    )  # roll/pitch/yaw 조종의 기준이 되는 회전 상태 저장
    left_hand_pos = (
        robot.body_node_for(left_target_body).world_transform[:3, 3].copy()
    )  # position IK로 위치를 고정할 왼손 위치 추출

    # 이벤트 핸들러가 갱신하는 패널에서 주어진 입력
    target_goal = None  # target panel에서 주어진 x, y, z
    target_rot = np.zeros(3)  # 주어진 roll, pitch, yaw

    # 궤적 관련 변수
    # trajectory_start_time = None
    trajectory_start = None
    trajectory_goal = None  # 형성된 단일 궤적의 목표
    last_tick_time = None
    trajectory_elapsed = 0.0  # 특정 궤적 안에서 진행된 정도
    trajectory_duration = 0.1  # 단일 궤적 시간 고정
    object_name = "pr_cokeCan"  # 상호작용할 물체 지정 (추후 리펙토링)
    friction_coefficient = (
        0.2  # force closure solver 입장에서는 마찰계수가 커질수록 가능한 접촉 힘 cone이 넓어져서 grasp 성공에 용이
    )

    # hand grasp 관련 변수
    grasp_alpha = np.zeros(2)
    grasp_goal = None  # grasp panel에서 들어온 목표 alpha
    grasp_goal_is_thumb = False  # 엄지인지 나머지 네마디인지
    is_grasped = False  # grasp 여부를 나타내는 flag 변수
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
    def handle_target_changed(target):
        nonlocal target_goal, target_rot

        # 타겟패널입력은 새로운 궤적 생성 타이밍에 관여하며, 입력을 감지하면 trajectory 관련 변수를 갱신함
        target_goal = target[:3].copy()
        target_rot = target[3:].copy()

    def handle_grasp_changed(alpha, is_thumb):
        nonlocal grasp_goal, grasp_goal_is_thumb

        grasp_goal = np.asarray(alpha, dtype=float).reshape(2)
        grasp_goal_is_thumb = is_thumb

    #####################################
    # Controller
    #####################################
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
            is_grasped, \
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
            elif not np.allclose(
                trajectory_goal, target_goal
            ):  # 기존 궤적의 끝, 목표 지점 간 차이가 있다면 새로운 궤적을 생성, 끊김을 완화할 수 있을 것으로 기대
                trajectory_start = current_hand_pos.copy()
                trajectory_goal = target_goal.copy()
                trajectory_elapsed = 0.0

            trajectory_elapsed += raw_dt

            t = min(trajectory_elapsed, trajectory_duration)  # 궤적 범위를 넘지 않도록 클리핑
            target_pos = interpolate_position_cubic(trajectory_start, trajectory_goal, trajectory_duration, t)

        candidate_state = RobotState(
            robot.state.joint_names,
            robot.state.qpos_addrs,
            robot.state.qpos_widths,
            robot.state.qpos.copy(),
        )

        if target_goal is not None:
            # rot = rpy2rotation_matrix(target_rot[0], target_rot[1], target_rot[2])
            rot_offset = rpy2rotation_matrix(target_rot[0], target_rot[1], target_rot[2])
            # rot = rot_offset @ right_target_base_rot
            rot = right_target_base_rot @ rot_offset

            # 오른손에 pose IK 계산, 캔 잡는 모션을 더 용이하게
            apply_ik(
                robot,
                candidate_state,
                target_pos,
                home_poses,
                "pose",  # "pose"
                right_target_body,
                rot=rot,  # roll, yaw, pitch 변환 행렬
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

        # grasp 입력이 있고, 캔을 잡은 상태가 아니라면 손가락 포즈를 업데이트할 수 있으므로 손가락 관절을 계산함
        # 추후 수정
        if grasp_goal is not None and not is_grasped:
            next_alpha = float(grasp_goal[0] if grasp_goal_is_thumb else grasp_goal[1])  # thumb 여부를 고려해 goal 설정
            compute_grasp(
                robot, candidate_state, next_alpha, grasp_goal_is_thumb
            )  # 목표 alpha를 candidate_state에 반영

        # 접촉점 상대속도 기반 (강체 자체 상대속도 아님)
        # 흐름도: 상대속도로 각 접촉점에 가해지는 힘 계산(힘 크기는 고정시킨 상태)->계산된 힘 벡터 합산->물체의 질량으로 가속도 및 속도와 변위까지 계산
        # v_contact = v_body + w × (p - center) # 회전 고려
        # v_contact = v_body # 회전 무시

        #####################################
        # Collison check
        #####################################
        is_collision, is_contact, contacts = check_collision(
            robot,
            candidate_state,
            home_poses,
            return_contacts=False,
        )

        if is_collision and not is_contact:  # hard collision은 FK를 수행하지 않고 바로 return
            # trajectory_start_time = None
            trajectory_goal = None
            trajectory_start = None
            target_goal = None
            last_tick_time = None
            trajectory_elapsed = 0.0
            grasp_goal = None
            return np.r_[current_hand_pos.copy(), target_rot.copy()]

        # 이제부턴 허용되는 collision만 남은 상태
        robot.state.qpos[:] = candidate_state.qpos  # qpos 업데이트

        if grasp_goal is not None:
            if not is_grasped:  # 물체를 잡지 않은 상태, 즉 손가락을 움직일 수 있는 상태 (추후 수정)
                grasp_alpha[:] = grasp_goal
            grasp_goal = None

        # qpos 업데이트 이후 관절 속도도 업데이트 (추후 수정)
        # 후보: time-scaling velocity를 사용해보기
        if raw_dt > 0:
            for joint_name, addr in robot.state.qpos_addrs.items():
                width = robot.state.qpos_widths[joint_name]
                robot.state.qvel[addr : addr + width] = (
                    robot.state.qpos[addr : addr + width] - prev_qpos[addr : addr + width]
                ) / raw_dt  # 이동한 관절위치를 dt로 나누어 관절속도를 구함

        # FK 적용: 왼손, 오른손, 손가락 동시에
        apply_fk(robot, robot.state, home_poses)

        # 트위스트 업데이트 시점: 트위스트는 현재 qpos에 대한 자코비안 행렬과 관절 속도의 곱으로 표현되므로 qpos를 적용한 이후 업데이트함
        # 근데 트위스트를 업데이트하고 충돌 감지에서 걸린 경우 고려해야 -> 사전에 hard collision 경우는 rollback해서 괜찮음
        update_body_twists(robot, robot.state, object_name)

        if is_grasped:  # 물체를 잡고 있는 상태면 손과 물체가 동일한 변위를 갖게 함
            contact_body_pos = robot.body_node_for(right_target_body).world_transform[:3, 3].copy()
            object_node = robot.body_node_for(object_name)
            object_pos = object_node.world_transform[:3, 3].copy()  # 물체 위치

            if offset is None:
                offset = object_pos - contact_body_pos

            object_displacement = contact_body_pos + offset - object_pos
            # 잡힌 물체의 위치와 속도 업데이트
            apply_pos(robot, object_name, object_displacement)
            update_qvel(robot, object_name, object_displacement, raw_dt)

        #####################################
        # Solver
        #####################################
        contacts = None
        if is_contact or is_grasped:
            _, _, contacts = check_collision(
                robot,
                robot.state,
                home_poses,
                return_contacts=True,
            )

        object_contacts = object_body_contacts(contacts, object_name)

        # 접촉점에서의 상대속도 계산은 compute_contact_force_sum() 내부에서 수행

        is_grasped, offset = update_grasp_state(
            robot,
            object_name,
            right_target_body,
            object_contacts,
            grasp_changed,
            is_grasped,
            offset,
            dynamics_dict[object_name]["mass"],
            friction_coefficient,
        )

        dynamics_dict[object_name]["is_grasped"] = is_grasped

        #####################################
        # Integrator
        #####################################
        if object_contacts and not is_grasped:
            object_force = sum_contact_forces(  # 물체에 가해지는 힘 벡터를 합산
                contacts=object_contacts,
            )

            # 힘 벡터가 테이블과 평행하도록 클리핑 (추후 수정)
            object_force[2] = 0.0
            integrate_force(
                robot,
                object_name,
                object_force,
                dynamics_dict[object_name]["mass"],
                physics_dt,
            )

        if (
            trajectory_elapsed >= trajectory_duration
        ):  # 초반에 elapsed가 초과되어도 조금이라도 움직이게 하기 위해(?) handle_tick() 후반부에서 검사
            # trajectory_start_time = None
            trajectory_start = None
            trajectory_goal = None
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
