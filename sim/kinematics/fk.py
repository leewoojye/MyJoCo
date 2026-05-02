import numpy as np
from sim.math3d.rotation import skew
from sim.math3d.screw import screw_hat, unit_screw_axis
from scipy.linalg import expm
from test_folder.test_mujoco import RobotGeometries

# PoE: product of exponential
def forward_kinematics_all_links(robot: RobotGeometries, state, ctrl):
  T = np.eye(4)

  all_joints = []

  for node in robot.body_nodes.values():
      for joint in node.joints:
          all_joints.append(joint)

  all_joints.sort(key=lambda joint: joint["qpos_addr"])

  for joint in all_joints:
    q = joint["pos"]
    w = joint["axis"]
    S = unit_screw_axis(q,w,0,"revolute")
    S_exp = skew(S) * ctrl[joint["qpos_addr"]]
    T = T @ expm(S_exp)

  # end-effector 동차변환행렬(현재자세)를 마지막에 곱함
  node = robot.body_node_for("link6")
  current_pose = node.world_transform
  T = T @ current_pose
  return T