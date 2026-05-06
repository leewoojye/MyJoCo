import numpy as np
import open3d as o3d
import time

from sim.model.kinematics.fk import apply_fk
from sim.model.kinematics.ik import apply_ik
from sim.model.motion.trajectory import interpolate_position
from sim.model.robot.loader_mujoco import build_robot_geometries
from sim.model.robot.state import RobotState
from sim.view.renderer import run_ik_target_window


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
    vis.create_window(window_name="My Simulator - Initial Pose")
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
    home_poses = {
        node.name: node.world_transform.copy() for node in robot.root_body.iter_nodes()
    }
    home_poses["_qpos"] = robot.state.qpos.copy()

    trajectory_start = None
    trajectory_goal = None
    trajectory_start_time = None
    trajectory_duration = 1.0  # 단일 궤적 시간 고정

    # callback 함수
    # 패널입력, 시간을 받아 궤적 생성 ->
    def handle_target_changed(target_pos):
        nonlocal trajectory_start, trajectory_goal, trajectory_start_time
        trajectory_start = (
            robot.body_node_for("hx5_r_base").world_transform[:3, 3].copy()
        )
        # 타겟패널입력은 새로운 궤적 생성 타이밍에 관여하며, 입력을 감지하면 trajectory 관련 변수를 갱신함
        trajectory_goal = target_pos.copy()
        trajectory_start_time = time.perf_counter()

    # 패널 감지 콜백함수가 궤적 생성 시점에 관여한다면, tick(유사 clock) 콜백함수가 로봇의 실제 ik 적용과 렌더링을 맡음
    def handle_tick():
        nonlocal trajectory_start, trajectory_goal, trajectory_start_time
        if trajectory_goal is None:  # 정적, 동적 상태 판별
            return False
        elapsed = time.perf_counter() - trajectory_start_time
        t = min(elapsed, trajectory_duration)
        target_pos = interpolate_position(
            trajectory_start, trajectory_goal, trajectory_duration, t
        )
        apply_ik(robot, robot.state, target_pos, home_poses)
        if elapsed >= trajectory_duration:
            trajectory_start = None
            trajectory_goal = None
            trajectory_start_time = None
        return True

    # joint/link의 body frame 기준 행렬로 렌더링
    run_ik_target_window(
        robot, on_target_changed=handle_target_changed, on_tick=handle_tick
    )  # callback으로 연결


if __name__ == "__main__":
    main()
