from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
if "--no-show" in sys.argv:
    os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from docs.modern_robotics.mr_utils import wrap_to_pi


def circle_intersection(B, D, b, c, branch=1):
    delta = D - B
    distance = np.linalg.norm(delta)
    a = (b**2 - c**2 + distance**2) / (2.0 * distance)
    height_sq = max(b**2 - a**2, 0.0)
    height = math.sqrt(height_sq)
    unit = delta / distance
    normal = np.array([-unit[1], unit[0]])
    return B + a * unit + branch * height * normal


def closure(alpha, beta, gamma, lengths):
    a, b, c, d = lengths
    B = np.array([a * math.cos(alpha), a * math.sin(alpha)])
    D = np.array([d, 0.0])
    return B + b * np.array([math.cos(beta), math.sin(beta)]) - D - c * np.array([math.cos(gamma), math.sin(gamma)])


def constraint_jacobian(beta, gamma, lengths):
    _, b, c, _ = lengths
    return np.array(
        [
            [-b * math.sin(beta), c * math.sin(gamma)],
            [b * math.cos(beta), -c * math.cos(gamma)],
        ]
    )


def solve_beta_gamma(alpha, beta_gamma0, lengths, max_iters=30):
    beta, gamma = beta_gamma0
    for _ in range(max_iters):
        f = closure(alpha, beta, gamma, lengths)
        if np.linalg.norm(f) < 1e-11:
            break
        J = constraint_jacobian(beta, gamma, lengths)
        step = np.linalg.solve(J, -f)
        beta += step[0]
        gamma += step[1]
    return np.array([wrap_to_pi(beta), wrap_to_pi(gamma)])


def four_bar_points(alpha, beta, gamma, lengths):
    a, b, c, d = lengths
    A = np.array([0.0, 0.0])
    B = A + a * np.array([math.cos(alpha), math.sin(alpha)])
    D = np.array([d, 0.0])
    C_from_B = B + b * np.array([math.cos(beta), math.sin(beta)])
    C_from_D = D + c * np.array([math.cos(gamma), math.sin(gamma)])
    C = 0.5 * (C_from_B + C_from_D)
    return np.vstack([A, B, C, D, A])


def velocity_solution(alpha, beta, gamma, lengths, alpha_dot=1.0):
    a, _, _, _ = lengths
    f_alpha = np.array([-a * math.sin(alpha), a * math.cos(alpha)])
    J = constraint_jacobian(beta, gamma, lengths)
    beta_dot, gamma_dot = np.linalg.solve(J, -f_alpha * alpha_dot)
    return beta_dot, gamma_dot


def main(show=True):
    lengths = np.array([0.35, 0.8, 0.7, 1.0])
    a, b, c, d = lengths
    D = np.array([d, 0.0])

    alphas = np.linspace(0.0, 2.0 * math.pi, 220)
    B0 = np.array([a, 0.0])
    C0 = circle_intersection(B0, D, b, c, branch=1)
    beta_gamma = np.array([math.atan2(C0[1] - B0[1], C0[0] - B0[0]), math.atan2(C0[1], C0[0] - d)])

    solutions = []
    residuals = []
    dets = []
    gamma_dots = []
    coupler_point_path = []

    for alpha in alphas:
        beta_gamma = solve_beta_gamma(alpha, beta_gamma, lengths)
        beta, gamma = beta_gamma
        points = four_bar_points(alpha, beta, gamma, lengths)
        coupler_point = 0.55 * points[1] + 0.45 * points[2]
        beta_dot, gamma_dot = velocity_solution(alpha, beta, gamma, lengths)

        solutions.append([beta, gamma])
        residuals.append(np.linalg.norm(closure(alpha, beta, gamma, lengths)))
        dets.append(np.linalg.det(constraint_jacobian(beta, gamma, lengths)))
        gamma_dots.append(gamma_dot)
        coupler_point_path.append(coupler_point)

    solutions = np.array(solutions)
    residuals = np.array(residuals)
    dets = np.array(dets)
    gamma_dots = np.array(gamma_dots)
    coupler_point_path = np.array(coupler_point_path)

    print("Closed-chain four-bar kinematics")
    print(f"  lengths [input, coupler, rocker, ground] = {lengths}")
    print(f"  max loop-closure residual = {residuals.max():.3e}")
    print(f"  min |det constraint Jacobian| = {np.min(np.abs(dets)):.3e}")
    print("  near-zero determinant indicates a closed-chain singular configuration.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    sample_indices = np.linspace(0, len(alphas) - 1, 7, dtype=int)
    for idx in sample_indices:
        alpha = alphas[idx]
        beta, gamma = solutions[idx]
        pts = four_bar_points(alpha, beta, gamma, lengths)
        axes[0].plot(pts[:, 0], pts[:, 1], "-o", alpha=0.75)
    axes[0].plot(coupler_point_path[:, 0], coupler_point_path[:, 1], "k--", lw=1.5, label="coupler point")
    axes[0].set_title("Loop-closure configurations")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(np.rad2deg(alphas), np.rad2deg(solutions[:, 1]), label="rocker gamma")
    axes[1].plot(np.rad2deg(alphas), gamma_dots, label="gamma_dot for alpha_dot=1")
    axes[1].set_title("Output angle and velocity")
    axes[1].set_xlabel("input alpha [deg]")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(np.rad2deg(alphas), np.abs(dets), color="tab:purple")
    axes[2].set_yscale("log")
    axes[2].set_title("Constraint Jacobian singularity metric")
    axes[2].set_xlabel("input alpha [deg]")
    axes[2].set_ylabel("|det J_c|")
    axes[2].grid(True, which="both")

    fig.tight_layout()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "ch07_closed_chain_four_bar.png"
    fig.savefig(output_path, dpi=160)
    print(f"Saved figure: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main(show="--no-show" not in sys.argv)
