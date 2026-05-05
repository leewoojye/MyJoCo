import numpy as np
import open3d as o3d

from sim.model.kinematics.fk import apply_fk
from sim.model.kinematics.ik import apply_ik
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
    # state1 = RobotState.from_model(robot.model)

    # 새로운 state 인스턴스 생성 예시
    state = robot.state  # robot state 필드 참조, set()으로 수정 가능
    # state = RobotState.from_model(robot.model)
    # state.set("Joint1", 0.5)
    # state.set("Joint2", -0.4)
    # state.set("Joint3", 0.7)
    # state.set("Joint4", -0.6)
    # state.set("Joint5", 0.3)
    # state.set("Joint6", 0.2)
    # state.set("rh_r1", 0.3)
    # state.set("rh_r2", -0.3)
    # state.set("rh_l1", -0.3)
    # state.set("rh_l2", 0.3)

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

    # 5. 조립된 전체 로봇 화면에 띄우기
    print(f"body nodes: {len(robot.body_nodes)}")
    print(f"joint states: {len(robot.state.joint_names)}")
    print(f"render geometries: {len(robot.open3d_geometries())}")
    # if "left_hand" in robot.end_effectors:
    #     print(f"left hand position: {robot.end_effectors['left_hand'].position}")

    # callback 함수
    def handle_target_changed(target):
        apply_ik(robot, robot.state, target, home_poses)

    # joint/link의 body frame 기준 행렬로 렌더링
    run_ik_target_window(
        robot, on_target_changed=handle_target_changed
    )  # callback으로 연결


if __name__ == "__main__":
    main()
