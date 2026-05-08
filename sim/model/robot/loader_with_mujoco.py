from pathlib import Path
from typing import Dict

import mujoco
import numpy as np
import open3d as o3d

from sim.model.math3d.transform import create_transform_matrix
from sim.model.robot.body import BodyNode
from sim.model.robot.geometry import EndEffector, GeomRecord
from sim.model.robot.joint import create_joint_record, get_mujoco_name
from sim.model.robot.robot_model import RobotModel
from sim.model.robot.robot_state import RobotState

ROOT_DIR = Path(__file__).resolve().parents[3]
XML_PATH = ROOT_DIR / "robotis_mujoco_menagerie" / "robotis_ffw" / "scene_ffw_sh5.xml"

END_EFFECTOR_BODIES = {
    "left_hand": "hx5_l_base",
    "right_hand": "hx5_r_base",
    # "left_thumb_tip": "finger_l_link4",
    # "left_index_tip": "finger_l_link8",
    # "left_middle_tip": "finger_l_link12",
    # "left_ring_tip": "finger_l_link16",
    # "left_little_tip": "finger_l_link20",
    # "right_thumb_tip": "finger_r_link4",
    # "right_index_tip": "finger_r_link8",
    # "right_middle_tip": "finger_r_link12",
    # "right_ring_tip": "finger_r_link16",
    # "right_little_tip": "finger_r_link20",
}


def enum_name(enum_type, value):
    try:
        return enum_type(int(value)).name
    except ValueError:
        return str(int(value))


def quat_to_matrix(quat):
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


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

    # 문제점: gem_type 분기점이 적음
    # mjtGeom: MuJoCo가 정의한 기하 타입을 모아놓은 열거형
    # mjGEOM_MESH: 열거형 상수(고유한 정수)
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:  # ex. arm, finger link
        mesh_id = int(model.geom_dataid[geom_id])
        geometry = create_open3d_mesh(model, mesh_id)
        geometry.paint_uniform_color(model.geom_rgba[geom_id][:3])
    elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        geometry = create_floor_from_mujoco_plane(model, geom_id)
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        size = model.geom_size[geom_id]
        geometry = o3d.geometry.TriangleMesh.create_box(
            width=2 * size[0], height=2 * size[1], depth=2 * size[2]
        )
        geometry.translate(-size)
        geometry.paint_uniform_color(model.geom_rgba[geom_id][:3])
        geometry.compute_vertex_normals()
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = model.geom_size[geom_id][0]
        half_height = model.geom_size[geom_id][1]
        geometry = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius,
            height=2 * half_height,
            resolution=32,
        )
        geometry.paint_uniform_color(model.geom_rgba[geom_id][:3])
        geometry.compute_vertex_normals()
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return None, None, None
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return None, None, None
    else:
        return None, None, None

    transform = create_transform_matrix(
        data.geom_xmat[geom_id].reshape(3, 3),
        data.geom_xpos[geom_id],
    )
    geometry.transform(transform)
    return geometry, transform, mesh_id


def build_body_nodes(model, data):
    body_nodes: Dict[int, BodyNode] = {}

    for body_id in range(model.nbody):
        body_name = get_mujoco_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            f"body_{body_id}",
        )
        node = BodyNode(body_name, body_id)
        node.mass = float(model.body_mass[body_id])
        node.inertia = model.body_inertia[body_id].copy()
        node.local_transform = create_transform_matrix(
            quat_to_matrix(model.body_quat[body_id]),
            model.body_pos[body_id],
        )
        node.world_transform = create_transform_matrix(
            data.xmat[body_id].reshape(3, 3),
            data.xpos[body_id],
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
    # XML 파일을 읽어와 MjModel 객체를 생성
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    # MjModel과 달리 동적인 현재 상태를 담는 객체로, qpos, qvel 등 정보를 담음
    data = mujoco.MjData(model)
    state = RobotState.from_model(model)
    data.qpos[:] = state.qpos

    # 초기 상태의 절대 좌표계를 한 번 계산
    mujoco.mj_kinematics(model, data)

    body_nodes, root_body = build_body_nodes(model, data)
    ee_body_ids = set()
    for body_name in END_EFFECTOR_BODIES.values():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id != -1:
            ee_body_ids.add(body_id)

    # 로봇의 모든 geom을 순회
    for i in range(model.ngeom):
        geometry, T, mesh_id = create_open3d_geometry_from_geom(model, data, i)
        if geometry is None:
            continue

        body_id = int(model.geom_bodyid[i])
        geom_type = enum_name(mujoco.mjtGeom, model.geom_type[i])
        mesh_name = None
        if mesh_id is not None:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)

        record = GeomRecord(
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
        body_node = body_nodes[body_id]
        # display용 geom은 관례적으로 geom_group를 2로 할당함
        if model.geom_group[i] == 2:
            body_node.visual_records.append(record)
        # 어떤 충돌 타입에 관여하는지 나타내는 geom_contype, geom_conaffinity 필드로 collision records 배열을 구성
        if model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0:
            body_node.collision_records.append(record)

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
            transform=create_transform_matrix(rot, pos),
            geom_records=list(body_nodes[body_id].all_records()),
        )

    return RobotModel(model, data, state, body_nodes, root_body, end_effectors)
