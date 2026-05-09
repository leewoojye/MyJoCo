# jacobian frame, wrench frame은 같아야 함
# space frame / body frame jacobian 구분
# τ = Jᵀ F
def force_to_torque(J_pos, force):  # J_pos: position jacobian
    return J_pos.T @ force


def wrench_to_torque(J, wrench):  # J: geometric jacobian
    return J.T @ wrench


def velocity_to_twist(J, qdot):
    return J @ qdot


# task space / work space
def compute_mass_matrix():
    return


def coriolis_matrix():
    return


def gravity_term():
    return


def inverse_dynamics():
    return


def forward_dynamics():
    return
