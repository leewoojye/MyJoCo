import mujoco
import open3d as o3d
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
XML_PATH = ROOT_DIR / "robotis_mujoco_menagerie" / "robotis_ffw" / "ffw_sh5.xml"

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


@dataclass
class GeometryRecord:
    mesh: o3d.geometry.TriangleMesh
    geom_id: int
    geom_name: Optional[str]
    body_id: int
    body_name: Optional[str]
    mesh_id: int
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


class RobotGeometries(list):
    """Open3D geometry list with MuJoCo metadata attached."""

    def __init__(self, model, data, records, end_effectors):
        super().__init__([record.mesh for record in records])
        self.model = model
        self.data = data
        self.records = records
        self.end_effectors = end_effectors
        self.mesh_to_record = {id(record.mesh): record for record in records}

    def record_for(self, mesh):
        return self.mesh_to_record[id(mesh)]

    def update_from_mujoco(self):
        mujoco.mj_forward(self.model, self.data)

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


def draw_scene(robot_geometries):
    scene_geometries = [
        create_floor(),
        create_grid(),
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25),
        *robot_geometries,
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


def build_robot_geometries(xml_path=XML_PATH):
    # 1. MuJoCo로 XML 파싱 및 초기 구조 불러오기
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # ⭐️ 핵심: 초기 상태(관절 각도 0)의 절대 좌표계(Global Position)를 단 한 번 계산합니다.
    # 이 함수를 쓰면 부모-자식 링크 간의 복잡한 행렬 곱셈을 무조코가 대신 해줍니다.
    mujoco.mj_kinematics(model, data)

    records = []
    records_by_body_id = {}
    ee_body_ids = set()
    for body_name in END_EFFECTOR_BODIES.values():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id != -1:
            ee_body_ids.add(body_id)

    # 2. 로봇의 모든 기하학적 형태(geom)를 순회합니다.
    for i in range(model.ngeom):
        # 해당 geom이 3D 메쉬(.stl 등)인지 확인합니다. (박스나 구 같은 기본 도형은 제외)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:

            # 메쉬 ID 가져오기
            mesh_id = int(model.geom_dataid[i])

            # MuJoCo가 XML의 scale과 meshdir을 반영해 컴파일한 메쉬 데이터를 사용합니다.
            mesh = create_open3d_mesh(model, mesh_id)

            # 3. MuJoCo에서 절대 위치(xpos)와 회전 행렬(xmat) 가져오기
            pos = data.geom_xpos[i] # [x, y, z] 평행 이동
            mat = data.geom_xmat[i].reshape(3, 3) # 3x3 회전 행렬

            # 4. Open3D 적용을 위한 4x4 동차 변환 행렬(Transformation Matrix) T 만들기
            T = make_transform(pos, mat)

            # 메쉬를 3D 공간의 올바른 위치와 각도로 변환
            mesh.transform(T)

            body_id = int(model.geom_bodyid[i])
            record = GeometryRecord(
                mesh=mesh,
                geom_id=i,
                geom_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i),
                body_id=body_id,
                body_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                mesh_id=mesh_id,
                mesh_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id),
                transform=T.copy(),
                is_end_effector=body_id in ee_body_ids,
            )
            records.append(record)
            records_by_body_id.setdefault(body_id, []).append(record)

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

    return RobotGeometries(model, data, records, end_effectors)


def main():
    geometries = build_robot_geometries()

    # 5. 조립된 전체 로봇 화면에 띄우기
    print("초기 로봇 렌더링 완료!")
    print(f"geometry meshes: {len(geometries)}")
    print(f"left hand position: {geometries.end_effectors['left_hand'].position}")
    draw_scene(geometries)


if __name__ == "__main__":
    main()
