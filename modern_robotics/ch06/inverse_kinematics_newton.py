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

from mr_utils import planar_arm_points, planar_jacobian, planar_pose, wrap_to_pi


def pose_error(target_pose, theta, lengths):
    current = planar_pose(lengths, theta)
    error = target_pose - current
    error[0] = wrap_to_pi(error[0])
    return error


def damped_least_squares_step(J, error, damping=0.04):
    return J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), error)


def solve_ik(lengths, target_pose, theta0, max_iters=80, tol=1e-5):
    theta = theta0.astype(float).copy()
    history = []
    path = []

    for _ in range(max_iters):
        error = pose_error(target_pose, theta, lengths)
        history.append(np.linalg.norm(error))
        path.append(planar_pose(lengths, theta)[1:])
        if history[-1] < tol:
            break

        J = planar_jacobian(lengths, theta)
        dtheta = damped_least_squares_step(J, error)
        max_step = 0.35
        step_norm = np.linalg.norm(dtheta)
        if step_norm > max_step:
            dtheta *= max_step / step_norm
        theta += dtheta
        theta = wrap_to_pi(theta)

    history.append(np.linalg.norm(pose_error(target_pose, theta, lengths)))
    path.append(planar_pose(lengths, theta)[1:])
    return theta, np.array(history), np.array(path)


def plot_arm(ax, lengths, theta, label, color):
    points = planar_arm_points(lengths, theta)
    ax.plot(points[:, 0], points[:, 1], "-o", lw=2.5, color=color, label=label)


def main(show=True):
    lengths = np.array([1.0, 0.75, 0.5])
    target_pose = np.array([math.radians(-25.0), 1.35, 0.55])

    guesses = [
        np.deg2rad(np.array([20.0, 40.0, -90.0])),
        np.deg2rad(np.array([95.0, -95.0, -20.0])),
    ]
    colors = ["tab:blue", "tab:orange"]
    results = [solve_ik(lengths, target_pose, guess) for guess in guesses]

    print("Newton-Raphson inverse kinematics")
    print(f"  target [phi, x, y] = {target_pose.round(4)}")
    for i, (theta, history, _) in enumerate(results, start=1):
        print(f"  solution {i}: theta deg = {np.rad2deg(theta).round(3)}")
        print(f"              final error norm = {history[-1]:.3e}, iterations = {len(history) - 1}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    target_x = target_pose[1]
    target_y = target_pose[2]
    target_phi = target_pose[0]
    ax1.scatter(target_x, target_y, marker="*", s=180, color="tab:red", label="target")
    ax1.arrow(target_x, target_y, 0.18 * math.cos(target_phi), 0.18 * math.sin(target_phi), color="tab:red", head_width=0.04)

    for i, ((theta, _, path), color) in enumerate(zip(results, colors), start=1):
        ax1.plot(path[:, 0], path[:, 1], "--", color=color, alpha=0.6)
        plot_arm(ax1, lengths, theta, f"solution {i}", color)

    ax1.set_title("Different IK solutions from different initial guesses")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.axis("equal")
    ax1.grid(True)
    ax1.legend()

    for i, ((_, history, _), color) in enumerate(zip(results, colors), start=1):
        ax2.semilogy(history, "-o", ms=3, color=color, label=f"guess {i}")
    ax2.set_title("Newton update convergence")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("pose error norm")
    ax2.grid(True, which="both")
    ax2.legend()

    fig.tight_layout()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "ch06_inverse_kinematics_newton.png"
    fig.savefig(output_path, dpi=160)
    print(f"Saved figure: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main(show="--no-show" not in sys.argv)
