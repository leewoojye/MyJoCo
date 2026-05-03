import time
import numpy as np
import trimesh
import viser
import viser.transforms as tf

robot = {
    "links": [
        {"name": "torso", "parent": None, "offset": [0.0, 0.0, 1.0],
         "geom": {"type": "box", "size": [0.35, 0.2, 0.6]}},
        {"name": "right_upper_arm", "parent": "torso", "joint": "right_shoulder_yaw",
         "offset": [0.25, -0.18, 0.25],
         "geom": {"type": "capsule", "radius": 0.04, "length": 0.3}},
        {"name": "right_forearm", "parent": "right_upper_arm", "joint": "right_elbow",
         "offset": [0.0, 0.0, -0.32],
         "geom": {"type": "capsule", "radius": 0.035, "length": 0.3}},
        {"name": "right_hand", "parent": "right_forearm", "joint": "right_wrist_pitch",
         "offset": [0.0, 0.0, -0.25],
         "geom": {"type": "sphere", "radius": 0.06}},
    ],
    "joints": [
        {"name": "right_shoulder_yaw", "type": "revolute", "axis": [0, 0, 1], "limit": [-1.57, 1.57]},
        {"name": "right_elbow", "type": "revolute", "axis": [1, 0, 0], "limit": [0.0, 2.4]},
        {"name": "right_wrist_pitch", "type": "revolute", "axis": [1, 0, 0], "limit": [-1.2, 1.2]},
    ]
}

joint_map = {j["name"]: j for j in robot["joints"]}
link_map = {l["name"]: l for l in robot["links"]}

server = viser.ViserServer()
server.scene.set_up_direction("+z")
server.scene.add_grid("/grid", width=4.0, height=4.0)
server.scene.configure_default_lights(cast_shadow=True)

def make_geom(geom):
    t = geom["type"]
    if t == "box":
        return trimesh.creation.box(extents=geom["size"])
    elif t == "capsule":
        return trimesh.creation.capsule(radius=geom["radius"], height=geom["length"], count=[16, 16])
    elif t == "sphere":
        return trimesh.creation.icosphere(subdivisions=3, radius=geom["radius"])
    raise ValueError(t)

def angle_to_wxyz(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    return tf.SO3.exp(axis * angle).wxyz

joint_handles = {}

root = server.scene.add_frame("/robot", axes_length=0.15, axes_radius=0.005)

# torso
torso = link_map["torso"]
torso_frame = server.scene.add_frame(
    "/robot/torso",
    position=np.array(torso["offset"], dtype=float),
    axes_length=0.12,
    axes_radius=0.004,
)
torso_mesh = make_geom(torso["geom"])
server.scene.add_mesh_simple(
    "/robot/torso/mesh",
    vertices=np.asarray(torso_mesh.vertices),
    faces=np.asarray(torso_mesh.faces),
    color=(170, 170, 180),
)
server.scene.add_label("/robot/torso/name", text="torso", position=(0, 0, 0.4))

# upper arm joint
s = link_map["right_upper_arm"]
shoulder_joint = server.scene.add_frame(
    "/robot/torso/right_shoulder_yaw",
    position=np.array(s["offset"], dtype=float),
    axes_length=0.08,
    axes_radius=0.003,
)
joint_handles["right_shoulder_yaw"] = shoulder_joint
upper_arm_frame = server.scene.add_frame("/robot/torso/right_shoulder_yaw/right_upper_arm")
upper_arm_mesh = make_geom(s["geom"])
server.scene.add_mesh_simple(
    "/robot/torso/right_shoulder_yaw/right_upper_arm/mesh",
    vertices=np.asarray(upper_arm_mesh.vertices),
    faces=np.asarray(upper_arm_mesh.faces),
    color=(90, 160, 255),
)

# forearm joint
e = link_map["right_forearm"]
elbow_joint = server.scene.add_frame(
    "/robot/torso/right_shoulder_yaw/right_upper_arm/right_elbow",
    position=np.array(e["offset"], dtype=float),
    axes_length=0.07,
    axes_radius=0.003,
)
joint_handles["right_elbow"] = elbow_joint
forearm_frame = server.scene.add_frame(
    "/robot/torso/right_shoulder_yaw/right_upper_arm/right_elbow/right_forearm"
)
forearm_mesh = make_geom(e["geom"])
server.scene.add_mesh_simple(
    "/robot/torso/right_shoulder_yaw/right_upper_arm/right_elbow/right_forearm/mesh",
    vertices=np.asarray(forearm_mesh.vertices),
    faces=np.asarray(forearm_mesh.faces),
    color=(255, 150, 90),
)

# wrist joint
w = link_map["right_hand"]
wrist_joint = server.scene.add_frame(
    "/robot/torso/right_shoulder_yaw/right_upper_arm/right_elbow/right_forearm/right_wrist_pitch",
    position=np.array(w["offset"], dtype=float),
    axes_length=0.06,
    axes_radius=0.0025,
)
joint_handles["right_wrist_pitch"] = wrist_joint
hand_frame = server.scene.add_frame(
    "/robot/torso/right_shoulder_yaw/right_upper_arm/right_elbow/right_forearm/right_wrist_pitch/right_hand"
)
hand_mesh = make_geom(w["geom"])
server.scene.add_mesh_simple(
    "/robot/torso/right_shoulder_yaw/right_upper_arm/right_elbow/right_forearm/right_wrist_pitch/right_hand/mesh",
    vertices=np.asarray(hand_mesh.vertices),
    faces=np.asarray(hand_mesh.faces),
    color=(120, 220, 140),
)

sliders = {}
for j in robot["joints"]:
    lo, hi = j["limit"]
    sliders[j["name"]] = server.gui.add_slider(
        j["name"], min=lo, max=hi, step=0.01, initial_value=0.0
    )

while True:
    for j in robot["joints"]:
        joint_handles[j["name"]].wxyz = angle_to_wxyz(j["axis"], sliders[j["name"]].value)
    time.sleep(0.02)