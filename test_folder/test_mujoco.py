import mujoco
import open3d as o3d
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
XML_PATH = ROOT_DIR / "robotis_mujoco_menagerie" / "robotis_omy" / "scene.xml"

END_EFFECTOR_BODIES = {
    # These are MuJoCo body origins. For a precise TCP, add named <site> tags in the XML.
    "left_hand": "hx5_l_base",
    "right_hand": "hx5_r_base",
    "left_thumb_tip": "finger_l_link4",
    "left_index_tip": "finger_l_link8",
    "left_middle_tip": "finger_l_link12",
    "left_ring_tip": "finger_l_link16",
    "left_little_tip": "finger_l_link20",
    "right_thumb_tip": "finger_r_link4",
    "right_index_tip": "finger_r_link8",
    "right_middle_tip": "finger_r_link12",
    "right_ring_tip": "finger_r_link16",
    "right_little_tip": "finger_r_link20",
}


class JointType(Enum):
    FREE = "free"
    BALL = "ball"
    SLIDE = "slide"
    HINGE = "hinge"
    UNKNOWN = "unknown"


class MuJoCoBodyNode:
    def __init__(self, name, body_id=None):
        self.name = name
        self.body_id = body_id

        # 1. MuJoCo XML 정보 보존 영역
        self.mass = 0.0
        self.inertia = []
        self.joints = []
        self.attributes = {}

        # 2. Open3D 시각화 객체 영역
        self.geometries = []
        self.geometry_records = []

        # 3. 계층 구조 보존 영역 (Kinematic Tree)
        self.parent = None
        self.children = []

        # 상대적 위치 변환 행렬 (body pos, quat 속성 보존)
        self.local_transform = np.eye(4)
        self.world_transform = np.eye(4)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def iter_nodes(self):
        yield self
        for child in self.children:
            yield from child.iter_nodes()

    def all_geometries(self):
        geometries = list(self.geometries)
        for child in self.children:
            geometries.extend(child.all_geometries())
        return geometries


@dataclass
class GeometryRecord:
    mesh: o3d.geometry.TriangleMesh
    geom_id: int
    geom_name: Optional[str]
    body_id: int
    body_name: Optional[str]
    geom_type: str
    mesh_id: Optional[int]
    mesh_name: Optional[str]
    transform: np.ndarray
    is_end_effector: bool = False


@dataclass
class EndEffector:
    name: str
    body_name: str
    body_id: int
    position: np.ndarray
    rotation: np.ndarray
    transform: np.ndarray
    geometry_records: List[GeometryRecord]


class RobotState:
    """Current robot joint state, separated from the body tree structure."""

    def __init__(self, joint_names, qpos_addrs, qpos_widths, initial_qpos):
        self.joint_names = list(joint_names)
        self.joint_index = {name: i for i, name in enumerate(self.joint_names)}
        self.qpos_addrs = dict(qpos_addrs)
        self.qpos_widths = dict(qpos_widths)
        self.qpos = np.asarray(initial_qpos, dtype=float).copy()

    @classmethod
    def from_model(cls, model, keyframe_name="home"):
        joint_names = []
        qpos_addrs = {}
        qpos_widths = {}

        for joint_id in range(model.njnt):
            joint_name = get_mujoco_name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
                f"joint_{joint_id}",
            )
            joint_names.append(joint_name)
            qpos_addrs[joint_name] = int(model.jnt_qposadr[joint_id])
            qpos_widths[joint_name] = qpos_width_for_joint_type(
                model.jnt_type[joint_id]
            )

        return cls(
            joint_names=joint_names,
            qpos_addrs=qpos_addrs,
            qpos_widths=qpos_widths,
            initial_qpos=initial_qpos_from_keyframe(model, keyframe_name),
        )

    def get(self, joint_name):
        qpos = self.get_qpos(joint_name)
        if qpos.size == 1:
            return float(qpos[0])
        return qpos

    def set(self, joint_name, value):
        addr = self.qpos_addrs[joint_name]
        width = self.qpos_widths[joint_name]
        values = np.asarray(value, dtype=float).reshape(-1)

        if values.size != width:
            raise ValueError(
                f"{joint_name} expects {width} qpos value(s), got {values.size}"
            )

        self.qpos[addr : addr + width] = values

    def get_qpos(self, joint_name):
        addr = self.qpos_addrs[joint_name]
        width = self.qpos_widths[joint_name]
        return self.qpos[addr : addr + width].copy()

    def as_dict(self):
        return {joint_name: self.get(joint_name) for joint_name in self.joint_names}


class RobotGeometries(list):
    """Open3D geometry list backed by a MuJoCo body tree."""

    def __init__(
        self, model, data, state, body_nodes, root_body, records, end_effectors
    ):
        super().__init__(root_body.all_geometries())
        self.model = model
        self.data = data
        self.state = state
        self.body_nodes = body_nodes
        self.root_body = root_body
        self.records = records
        self.end_effectors = end_effectors
        self.mesh_to_record = {id(record.mesh): record for record in records}

    def record_for(self, mesh):
        return self.mesh_to_record[id(mesh)]

    def body_node_for(self, name_or_id):
        if isinstance(name_or_id, int):
            return self.body_nodes.get(name_or_id)

        for node in self.body_nodes.values():
            if node.name == name_or_id:
                return node

        return None

    def open3d_geometries(self):
        return self.root_body.all_geometries()

    def update_from_mujoco(self):
        self.data.qpos[:] = self.state.qpos
        mujoco.mj_forward(self.model, self.data)

        for body_id, node in self.body_nodes.items():
            pos = self.data.xpos[body_id]
            mat = self.data.xmat[body_id].reshape(3, 3)
            node.world_transform = make_transform(pos, mat)

        for record in self.records:
            pos = self.data.geom_xpos[record.geom_id]
            mat = self.data.geom_xmat[record.geom_id].reshape(3, 3)
            new_transform = make_transform(pos, mat)
            delta = new_transform @ np.linalg.inv(record.transform)
            record.mesh.transform(delta)
            record.transform = new_transform.copy()

        for end_effector in self.end_effectors.values():
            pos = self.data.xpos[end_effector.body_id].copy()
            rot = self.data.xmat[end_effector.body_id].reshape(3, 3).copy()
            end_effector.position = pos
            end_effector.rotation = rot
            end_effector.transform = make_transform(pos, rot)


def create_open3d_mesh(model, mesh_id):
    """Create an Open3D mesh from MuJoCo's compiled mesh data."""
    vert_start = model.mesh_vertadr[mesh_id]
    vert_end = vert_start + model.mesh_vertnum[mesh_id]
    face_start = model.mesh_faceadr[mesh_id]
    face_end = face_start + model.mesh_facenum[mesh_id]

    vertices = model.mesh_vert[vert_start:vert_end].copy()
    faces = model.mesh_face[face_start:face_end].copy()

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.6, 0.6, 0.6])
    return mesh


def get_mujoco_name(model, obj_type, obj_id, fallback):
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else fallback


def enum_name(enum_type, value):
    try:
        return enum_type(int(value)).name
    except ValueError:
        return str(int(value))


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


def initial_qpos_from_keyframe(model, keyframe_name="home"):
    if model.nkey == 0:
        return np.zeros(model.nq)

    key_id = 0
    for candidate_id in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, candidate_id)
        if name == keyframe_name:
            key_id = candidate_id
            break

    return model.key_qpos[key_id].copy()


def quat_to_matrix(quat):
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def make_transform(position, rotation_matrix):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = position
    return T


def create_floor(size=4.0, z=0.0, thickness=0.01, color=(0.22, 0.23, 0.24)):
    floor = o3d.geometry.TriangleMesh.create_box(
        width=size,
        height=size,
        depth=thickness,
    )
    floor.translate((-size / 2, -size / 2, z - thickness))
    floor.paint_uniform_color(color)
    floor.compute_vertex_normals()
    return floor


def create_floor_from_mujoco_plane(model, geom_id, fallback_size=4.0, thickness=0.01):
    size_x, size_y = model.geom_size[geom_id][:2]
    width = size_x * 2 if size_x > 0 else fallback_size
    height = size_y * 2 if size_y > 0 else fallback_size

    floor = o3d.geometry.TriangleMesh.create_box(
        width=width,
        height=height,
        depth=thickness,
    )
    floor.translate((-width / 2, -height / 2, -thickness))
    floor.paint_uniform_color(model.geom_rgba[geom_id][:3])
    floor.compute_vertex_normals()
    return floor


def create_open3d_geometry_from_geom(model, data, geom_id):
    geom_type = model.geom_type[geom_id]
    mesh_id = None

    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        geometry = create_open3d_mesh(model, mesh_id)
        geometry.paint_uniform_color(model.geom_rgba[geom_id][:3])
    elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        geometry = create_floor_from_mujoco_plane(model, geom_id)
    else:
        return None, None, None

    transform = make_transform(
        data.geom_xpos[geom_id],
        data.geom_xmat[geom_id].reshape(3, 3),
    )
    geometry.transform(transform)
    return geometry, transform, mesh_id


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


def build_body_nodes(model, data):
    body_nodes: Dict[int, MuJoCoBodyNode] = {}

    for body_id in range(model.nbody):
        body_name = get_mujoco_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            f"body_{body_id}",
        )
        node = MuJoCoBodyNode(body_name, body_id)
        node.mass = float(model.body_mass[body_id])
        node.inertia = model.body_inertia[body_id].copy()
        node.local_transform = make_transform(
            model.body_pos[body_id],
            quat_to_matrix(model.body_quat[body_id]),
        )
        node.world_transform = make_transform(
            data.xpos[body_id],
            data.xmat[body_id].reshape(3, 3),
        )
        node.attributes = {
            "body_id": body_id,
            "parent_id": int(model.body_parentid[body_id]),
            "root_id": int(model.body_rootid[body_id]),
            "mocap_id": int(model.body_mocapid[body_id]),
            "pos": model.body_pos[body_id].copy(),
            "quat": model.body_quat[body_id].copy(),
            "inertial_pos": model.body_ipos[body_id].copy(),
            "inertial_quat": model.body_iquat[body_id].copy(),
            "geom_addr": int(model.body_geomadr[body_id]),
            "geom_num": int(model.body_geomnum[body_id]),
            "joint_addr": int(model.body_jntadr[body_id]),
            "joint_num": int(model.body_jntnum[body_id]),
            "dof_addr": int(model.body_dofadr[body_id]),
            "dof_num": int(model.body_dofnum[body_id]),
        }

        joint_addr = int(model.body_jntadr[body_id])
        joint_num = int(model.body_jntnum[body_id])
        if joint_addr >= 0 and joint_num > 0:
            node.joints = [
                create_joint_record(model, data, joint_id)
                for joint_id in range(joint_addr, joint_addr + joint_num)
            ]

        body_nodes[body_id] = node

    for body_id, node in body_nodes.items():
        parent_id = int(model.body_parentid[body_id])
        if body_id != 0 and parent_id in body_nodes:
            body_nodes[parent_id].add_child(node)

    return body_nodes, body_nodes[0]


def build_robot_geometries(xml_path=XML_PATH):
    # 1. MuJoCo로 XML 파싱 및 초기 구조 불러오기
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    state = RobotState.from_model(model)
    data.qpos[:] = state.qpos

    # ⭐️ 핵심: 초기 상태(관절 각도 0)의 절대 좌표계(Global Position)를 단 한 번 계산합니다.
    # 이 함수를 쓰면 부모-자식 링크 간의 복잡한 행렬 곱셈을 무조코가 대신 해줍니다.
    mujoco.mj_kinematics(model, data)

    body_nodes, root_body = build_body_nodes(model, data)
    records = []
    records_by_body_id = {}
    ee_body_ids = set()
    for body_name in END_EFFECTOR_BODIES.values():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id != -1:
            ee_body_ids.add(body_id)

    # 2. 로봇의 모든 기하학적 형태(geom)를 순회합니다.
    for i in range(model.ngeom):
        geometry, T, mesh_id = create_open3d_geometry_from_geom(model, data, i)
        if geometry is None:
            continue

        body_id = int(model.geom_bodyid[i])
        geom_type = enum_name(mujoco.mjtGeom, model.geom_type[i])
        mesh_name = None
        if mesh_id is not None:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)

        record = GeometryRecord(
            mesh=geometry,
            geom_id=i,
            geom_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i),
            body_id=body_id,
            body_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
            geom_type=geom_type,
            mesh_id=mesh_id,
            mesh_name=mesh_name,
            transform=T.copy(),
            is_end_effector=body_id in ee_body_ids,
        )
        records.append(record)
        records_by_body_id.setdefault(body_id, []).append(record)

        body_node = body_nodes[body_id]
        body_node.geometries.append(geometry)
        body_node.geometry_records.append(record)

    end_effectors = {}
    for ee_name, body_name in END_EFFECTOR_BODIES.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            continue

        pos = data.xpos[body_id].copy()
        rot = data.xmat[body_id].reshape(3, 3).copy()
        end_effectors[ee_name] = EndEffector(
            name=ee_name,
            body_name=body_name,
            body_id=body_id,
            position=pos,
            rotation=rot,
            transform=make_transform(pos, rot),
            geometry_records=records_by_body_id.get(body_id, []),
        )

    return RobotGeometries(
        model, data, state, body_nodes, root_body, records, end_effectors
    )


def main():
    # 순환 import 방지
    from sim.kinematics.fk import compute_fk, apply_fk

    robot = build_robot_geometries()
    # home configuration 저장
    home_poses = {}
    for node in robot.root_body.iter_nodes():
        home_poses[node.name] = node.world_transform.copy()

    # robot 클래스 state와 독립적인 state 인스턴스 생성
    state1 = RobotState.from_model(robot.model)

    # robot 자체 state 인스턴스를 수정
    state = robot.state
    state.set("Joint1", 0.5)
    state.set("Joint2", -0.4)
    state.set("Joint3", 0.7)
    state.set("Joint4", -0.6)
    state.set("Joint5", 0.3)
    state.set("Joint6", 0.2)
    state.set("rh_r1", 0.3)
    state.set("rh_r2", -0.3)
    state.set("rh_l1", -0.3)
    state.set("rh_l2", 0.3)

    robot = apply_fk(robot, state, home_poses)

    # 5. 조립된 전체 로봇 화면에 띄우기
    print("초기 로봇 렌더링 완료!")
    print(f"body nodes: {len(robot.body_nodes)}")
    print(f"joint states: {len(robot.state.joint_names)}")
    print(f"render geometries: {len(robot.open3d_geometries())}")
    if "left_hand" in robot.end_effectors:
        print(f"left hand position: {robot.end_effectors['left_hand'].position}")

    # joint/link의 body frame 기준 행렬로 렌더링
    draw_scene(robot)  # robot links, joints 그리고 배경 렌더링


if __name__ == "__main__":
    main()
