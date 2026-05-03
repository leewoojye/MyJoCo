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

from docs.modern_robotics.mr_utils import (
    jacobian_body,
    jacobian_space,
    planar_arm_points,
    planar_jacobian,
    planar_pose,
    planar_screws,
)


def plot_arm(ax, lengths, theta, color="tab:blue"):
    points = planar_arm_points(lengths, theta)
    ax.plot(points[:, 0], points[:, 1], "-o", color=color, lw=2.5)
    return points


def ellipse_points(Jxy, samples=120):
    A = Jxy @ Jxy.T
    values, vectors = np.linalg.eigh(A)
    values = np.maximum(values, 0.0)
    angles = np.linspace(0.0, 2.0 * math.pi, samples)
    circle = np.vstack((np.cos(angles), np.sin(angles)))
    ellipse = vectors @ np.diag(np.sqrt(values)) @ circle
    return ellipse, np.sqrt(values)


def main(show=True):
    lengths = np.array([1.0, 0.75, 0.5])
    M, Slist, Blist = planar_screws(lengths)
    theta = np.deg2rad(np.array([40.0, -70.0, 55.0]))

    Js = jacobian_space(Slist, theta)
    Jb = jacobian_body(Blist, theta)
    Jplanar = planar_jacobian(lengths, theta)
    wrench = np.array([0.6, 8.0, -3.0])
    tau = Jplanar.T @ wrench

    print("Jacobian, statics, and manipulability")
    print(f"  rank(J_space) = {np.linalg.matrix_rank(Js)}")
    print(f"  rank(J_body)  = {np.linalg.matrix_rank(Jb)}")
    print(f"  planar pose Jacobian rows are [omega_z, v_x, v_y]")
    print(Jplanar.round(4))
    print(f"  wrench [moment_z, force_x, force_y] = {wrench}")
    print(f"  tau = J.T F = {tau.round(4)}")

    q1_grid = np.deg2rad(np.linspace(-160, 160, 100))
    q2_grid = np.deg2rad(np.linspace(-160, 160, 100))
    manipulability = np.zeros((len(q2_grid), len(q1_grid)))
    min_sigma = np.zeros_like(manipulability)

    for row, q2 in enumerate(q2_grid):
        for col, q1 in enumerate(q1_grid):
            Jxy = planar_jacobian(lengths, np.array([q1, q2, 0.0]))[1:, :]
            singular_values = np.linalg.svd(Jxy, compute_uv=False)
            manipulability[row, col] = np.prod(singular_values)
            min_sigma[row, col] = singular_values[-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    im = axes[0].imshow(
        manipulability,
        origin="lower",
        extent=[-160, 160, -160, 160],
        aspect="auto",
        cmap="viridis",
    )
    axes[0].set_title("Manipulability over theta1-theta2")
    axes[0].set_xlabel("theta1 [deg]")
    axes[0].set_ylabel("theta2 [deg]")
    fig.colorbar(im, ax=axes[0], label="sqrt(det(J J.T))")

    points = plot_arm(axes[1], lengths, theta)
    end_pose = planar_pose(lengths, theta)
    Jxy = Jplanar[1:, :]
    ellipse, radii = ellipse_points(Jxy)
    scale = 0.22
    axes[1].plot(end_pose[1] + scale * ellipse[0], end_pose[2] + scale * ellipse[1], color="tab:purple", lw=2)
    axes[1].arrow(end_pose[1], end_pose[2], 0.025 * wrench[1], 0.025 * wrench[2], color="tab:red", head_width=0.04)
    axes[1].set_title("Velocity ellipse and external wrench")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].axis("equal")
    axes[1].grid(True)
    axes[1].text(points[-1, 0], points[-1, 1], f"sigmas={radii.round(2)}")

    axes[2].bar(["joint 1", "joint 2", "joint 3"], tau, color=["tab:blue", "tab:orange", "tab:green"])
    axes[2].axhline(0.0, color="0.3", lw=1)
    axes[2].set_title("Static torques from tau = J.T F")
    axes[2].set_ylabel("torque")
    axes[2].grid(True, axis="y")

    fig.tight_layout()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "ch05_jacobian_statics_manipulability.png"
    fig.savefig(output_path, dpi=160)
    print(f"Saved figure: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main(show="--no-show" not in sys.argv)
