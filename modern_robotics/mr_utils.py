"""Small Modern Robotics helpers used by the chapter examples.

The convention is the one used in Modern Robotics:
twists are 6-vectors [omega, v], where omega is angular velocity and v is the
linear part. Homogeneous transforms map body-frame coordinates into the space
frame unless stated otherwise.
"""

from __future__ import annotations

import math

import numpy as np


EPS = 1e-9


def near_zero(value: float, eps: float = EPS) -> bool:
    return abs(value) < eps


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if near_zero(norm):
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def skew(omega: np.ndarray) -> np.ndarray:
    wx, wy, wz = np.asarray(omega, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ]
    )


def unskew(omega_hat: np.ndarray) -> np.ndarray:
    return np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])


def rot_axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = normalize(axis)
    axis_hat = skew(axis)
    return (
        np.eye(3)
        + math.sin(theta) * axis_hat
        + (1.0 - math.cos(theta)) * (axis_hat @ axis_hat)
    )


def so3_log(R: np.ndarray) -> np.ndarray:
    acos_input = (np.trace(R) - 1.0) / 2.0
    acos_input = float(np.clip(acos_input, -1.0, 1.0))

    if acos_input >= 1.0 - EPS:
        return np.zeros((3, 3))

    if acos_input <= -1.0 + EPS:
        if not near_zero(1.0 + R[2, 2]):
            omega = np.array([R[0, 2], R[1, 2], 1.0 + R[2, 2]])
            omega /= math.sqrt(2.0 * (1.0 + R[2, 2]))
        elif not near_zero(1.0 + R[1, 1]):
            omega = np.array([R[0, 1], 1.0 + R[1, 1], R[2, 1]])
            omega /= math.sqrt(2.0 * (1.0 + R[1, 1]))
        else:
            omega = np.array([1.0 + R[0, 0], R[1, 0], R[2, 0]])
            omega /= math.sqrt(2.0 * (1.0 + R[0, 0]))
        return skew(math.pi * omega)

    theta = math.acos(acos_input)
    return theta / (2.0 * math.sin(theta)) * (R - R.T)


def transform(R: np.ndarray | None = None, p: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


def inv_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv


def adjoint(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    adj = np.zeros((6, 6))
    adj[:3, :3] = R
    adj[3:, :3] = skew(p) @ R
    adj[3:, 3:] = R
    return adj


def exp_twist(S: np.ndarray, theta: float) -> np.ndarray:
    S = np.asarray(S, dtype=float).reshape(6)
    omega = S[:3]
    v = S[3:]
    omega_norm = np.linalg.norm(omega)

    if near_zero(omega_norm):
        return transform(np.eye(3), v * theta)

    omega = omega / omega_norm
    v = v / omega_norm
    theta = theta * omega_norm

    omega_hat = skew(omega)
    R = rot_axis_angle(omega, theta)
    G = (
        np.eye(3) * theta
        + (1.0 - math.cos(theta)) * omega_hat
        + (theta - math.sin(theta)) * (omega_hat @ omega_hat)
    )
    return transform(R, G @ v)


def matrix_exp6(Vtheta: np.ndarray) -> np.ndarray:
    Vtheta = np.asarray(Vtheta, dtype=float).reshape(6)
    omega_theta = Vtheta[:3]
    theta = np.linalg.norm(omega_theta)
    if near_zero(theta):
        return transform(np.eye(3), Vtheta[3:])
    return exp_twist(Vtheta / theta, theta)


def se3_to_vec(se3mat: np.ndarray) -> np.ndarray:
    return np.r_[unskew(se3mat[:3, :3]), se3mat[:3, 3]]


def matrix_log6(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    omega_mat = so3_log(R)

    if np.linalg.norm(omega_mat) < EPS:
        se3mat = np.zeros((4, 4))
        se3mat[:3, 3] = p
        return se3mat

    theta = math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    G_inv = (
        np.eye(3)
        - 0.5 * omega_mat
        + (1.0 / theta - 0.5 / math.tan(theta / 2.0))
        * (omega_mat @ omega_mat)
        / theta
    )

    se3mat = np.zeros((4, 4))
    se3mat[:3, :3] = omega_mat
    se3mat[:3, 3] = G_inv @ p
    return se3mat


def fk_space(M: np.ndarray, Slist: np.ndarray, theta_list: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for S, theta in zip(np.asarray(Slist).T, theta_list):
        T = T @ exp_twist(S, float(theta))
    return T @ M


def fk_body(M: np.ndarray, Blist: np.ndarray, theta_list: np.ndarray) -> np.ndarray:
    T = M.copy()
    for B, theta in zip(np.asarray(Blist).T, theta_list):
        T = T @ exp_twist(B, float(theta))
    return T


def jacobian_space(Slist: np.ndarray, theta_list: np.ndarray) -> np.ndarray:
    Slist = np.asarray(Slist, dtype=float)
    theta_list = np.asarray(theta_list, dtype=float)
    J = Slist.copy()
    T = np.eye(4)
    for i in range(1, Slist.shape[1]):
        T = T @ exp_twist(Slist[:, i - 1], theta_list[i - 1])
        J[:, i] = adjoint(T) @ Slist[:, i]
    return J


def jacobian_body(Blist: np.ndarray, theta_list: np.ndarray) -> np.ndarray:
    Blist = np.asarray(Blist, dtype=float)
    theta_list = np.asarray(theta_list, dtype=float)
    J = Blist.copy()
    T = np.eye(4)
    for i in range(Blist.shape[1] - 2, -1, -1):
        T = T @ exp_twist(-Blist[:, i + 1], theta_list[i + 1])
        J[:, i] = adjoint(T) @ Blist[:, i]
    return J


def rotz(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def planar_arm_points(lengths: np.ndarray, theta: np.ndarray) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=float)
    theta = np.asarray(theta, dtype=float)
    points = [np.array([0.0, 0.0])]
    angle = 0.0
    current = np.array([0.0, 0.0])
    for length, joint_angle in zip(lengths, theta):
        angle += joint_angle
        current = current + length * np.array([math.cos(angle), math.sin(angle)])
        points.append(current.copy())
    return np.array(points)


def planar_pose(lengths: np.ndarray, theta: np.ndarray) -> np.ndarray:
    points = planar_arm_points(lengths, theta)
    phi = float(np.sum(theta))
    return np.array([phi, points[-1, 0], points[-1, 1]])


def planar_jacobian(lengths: np.ndarray, theta: np.ndarray) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=float)
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    cumulative = np.cumsum(theta)
    J = np.zeros((3, n))
    J[0, :] = 1.0
    for j in range(n):
        for k in range(j, n):
            J[1, j] += -lengths[k] * math.sin(cumulative[k])
            J[2, j] += lengths[k] * math.cos(cumulative[k])
    return J


def planar_screws(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray(lengths, dtype=float)
    joint_x = np.r_[0.0, np.cumsum(lengths[:-1])]
    Slist = []
    omega = np.array([0.0, 0.0, 1.0])
    for x in joint_x:
        q = np.array([x, 0.0, 0.0])
        v = -np.cross(omega, q)
        Slist.append(np.r_[omega, v])
    Slist = np.array(Slist).T
    M = transform(np.eye(3), np.array([np.sum(lengths), 0.0, 0.0]))
    Blist = adjoint(inv_transform(M)) @ Slist
    return M, Slist, Blist


def xyt_to_T(x: float, y: float, phi: float) -> np.ndarray:
    return transform(rotz(phi), np.array([x, y, 0.0]))


def T_to_xyt(T: np.ndarray) -> np.ndarray:
    phi = math.atan2(T[1, 0], T[0, 0])
    return np.array([phi, T[0, 3], T[1, 3]])


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi
