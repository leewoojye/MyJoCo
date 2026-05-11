import numpy as np
from scipy.optimize import linprog
from sim.model.solver.contact_constraint import force_to_wrench
from sim.model.solver.form_closure import check_form_closure


def check_force_closure():
    # 각 접촉점마다 friction cone 생성
    # sliding 접촉 상태라면 작용하는 마찰력은 최대가 됨

    # friction cone으로부터 wrench cone 생성
    # wrench cone의 positive span이 전체 공간을 덮는지 확인
    return


# 평형 조건 확인과 동시에 평형을 위한 접촉점별 힘(가중치)을 계산
def solve_contact_forces(contact_points, external_wrench, friction_coefficient):
    # external_wrench: 중력항
    external_wrench = np.asarray(external_wrench, dtype=float).reshape(6)
    wrenches = []

    # 각 접촉점마다 friction cone edge force 생성
    for contact in contact_points:
        if contact.normal is None or contact.point is None:
            continue

        normal = np.asarray(contact.normal, dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-8:
            continue

        normal = normal / normal_norm

        if abs(normal[0]) < 0.9:
            a = np.array([1.0, 0.0, 0.0])
        else:
            a = np.array([0.0, 1.0, 0.0])

        basis1 = np.cross(normal, a)
        basis1 = basis1 / np.linalg.norm(basis1)

        basis2 = np.cross(normal, basis1)
        basis2 = basis2 / np.linalg.norm(basis2)

        edge_forces = [  # 각 edge force를 wrench로 변환
            normal + friction_coefficient * basis1,
            normal - friction_coefficient * basis1,
            normal + friction_coefficient * basis2,
            normal - friction_coefficient * basis2,
        ]

        for edge_force in edge_forces:
            wrenches.append(force_to_wrench(contact.point, edge_force))

    if not wrenches:
        return False, None

    G = np.column_stack(wrenches)  # wrench edge를 합쳐 G 생성
    m = G.shape[1]  # wrench edge 계수

    # 정적 평형(Static Equilibrium) 상태를 위한 접촉력 계산...
    result = linprog(
        c=np.ones(m),  # minimize sum(f)
        A_eq=G,
        b_eq=-external_wrench,
        bounds=[(0.0, None)] * m,  # f >= 0 제약
        method="highs",
    )

    if not result.success:
        return False, None

    edge_weights = result.x
    return True, edge_weights


def evaluate_grasp_state(contact_points, external_wrench, friction_coefficient):
    if check_form_closure(contact_points):
        return True, None

    return solve_contact_forces(contact_points, external_wrench, friction_coefficient)
