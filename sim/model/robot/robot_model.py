import mujoco
import numpy as np

from sim.model.robot.geometry import make_transform


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
