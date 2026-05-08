class RobotModel(list):
    def __init__(self, model, data, state, body_nodes, root_body, end_effectors):
        super().__init__(root_body.all_geometries())
        self.model = model
        self.data = data
        self.state = state
        self.body_nodes = body_nodes
        self.root_body = root_body
        self.end_effectors = end_effectors

    def body_node_for(self, name_or_id):
        if isinstance(name_or_id, int):
            return self.body_nodes.get(name_or_id)

        for node in self.body_nodes.values():
            if node.name == name_or_id:
                return node

        return None

    def open3d_geometries(self):
        return self.root_body.all_geometries()
