import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from sim.view.gui.target_panel import IKTargetPanel


def _link6_position(robot_geometries):
    r_hand = robot_geometries.body_node_for("hx5_r_base")
    if r_hand is None:
        raise ValueError("r_hand body node not found")
    return r_hand.world_transform[:3, 3]


def run_ik_target_window(
    robot_geometries,
    initial_target=None,
    on_target_changed=None,
    slider_range=(-0.5, 0.5),  # end-effector 조작 범위(단위: m)
    # 0.2: 안정권, 0.5 미세한 변화에도 매우 민감
):
    app = gui.Application.instance
    app.initialize()

    window = app.create_window("Experiment", 1280, 720)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)

    material = rendering.MaterialRecord()
    material.shader = "defaultLit"

    robot_geometry_names = []

    def add_robot_geometries():
        robot_geometry_names.clear()
        for index, geometry in enumerate(robot_geometries.open3d_geometries()):
            name = f"robot_{index}"
            scene.scene.add_geometry(name, geometry, material)
            robot_geometry_names.append(name)

    def refresh_robot_geometries():
        for name in robot_geometry_names:
            scene.scene.remove_geometry(name)
        add_robot_geometries()
        scene.force_redraw()

    def handle_target_changed(target):
        if on_target_changed is not None:
            on_target_changed(target)
        refresh_robot_geometries()

    add_robot_geometries()

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    scene.scene.add_geometry("axes", axes, material)

    panel = IKTargetPanel(
        initial_target
        if initial_target is not None
        else _link6_position(robot_geometries),
        on_target_changed=handle_target_changed,
        slider_range=slider_range,
    )

    window.add_child(scene)
    window.add_child(panel.widget)

    def on_layout(_):
        rect = window.content_rect
        panel_width = min(320, rect.width)
        scene.frame = gui.Rect(rect.x, rect.y, rect.width - panel_width, rect.height)
        panel.widget.frame = gui.Rect(
            rect.x + rect.width - panel_width,
            rect.y,
            panel_width,
            rect.height,
        )

    window.set_on_layout(on_layout)

    bounds = scene.scene.bounding_box
    scene.setup_camera(60.0, bounds, bounds.get_center())
    app.run()
