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
    fk_body,
    fk_space,
    planar_arm_points,
    planar_pose,
    planar_screws,
)


def plot_planar_arm(ax, lengths, theta, label, color):
    points = planar_arm_points(lengths, theta)
    ax.plot(points[:, 0], points[:, 1], "-o", color=color, lw=2.5, label=label)
    phi, x, y = planar_pose(lengths, theta)
    frame_len = 0.16
    ax.arrow(x, y, frame_len * math.cos(phi), frame_len * math.sin(phi), color="tab:red", head_width=0.035)
    ax.arrow(x, y, -frame_len * math.sin(phi), frame_len * math.cos(phi), color="tab:green", head_width=0.035)
    ax.text(x, y, label)


def main(show=True):
    lengths = np.array([1.0, 0.75, 0.5])
    M, Slist, Blist = planar_screws(lengths)

    theta = np.deg2rad(np.array([35.0, -55.0, 70.0]))
    T_space = fk_space(M, Slist, theta)
    T_body = fk_body(M, Blist, theta)

    print("PoE forward kinematics")
    print("  M home configuration:")
    print(M.round(4))
    print("  space-frame FK:")
    print(T_space.round(4))
    print("  body-frame FK:")
    print(T_body.round(4))
    print(f"  ||T_space - T_body|| = {np.linalg.norm(T_space - T_body):.3e}")

    rng = np.random.default_rng(7)
    random_thetas = rng.uniform(
        low=np.deg2rad([-160.0, -130.0, -150.0]),
        high=np.deg2rad([160.0, 130.0, 150.0]),
        size=(1200, 3),
    )
    workspace = np.array([planar_pose(lengths, th)[1:] for th in random_thetas])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_planar_arm(ax1, lengths, np.zeros(3), "home", "0.55")
    plot_planar_arm(ax1, lengths, theta, "sample", "tab:blue")
    plot_planar_arm(ax1, lengths, np.deg2rad([-45, 80, -35]), "other", "tab:orange")
    ax1.set_title("Serial-chain FK from joint screw axes")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.axis("equal")
    ax1.grid(True)
    ax1.legend()

    ax2.scatter(workspace[:, 0], workspace[:, 1], s=6, alpha=0.25, color="tab:blue")
    end = planar_pose(lengths, theta)[1:]
    ax2.scatter(end[0], end[1], s=80, color="tab:red", label="sample pose")
    ax2.set_title("Workspace sampled through the PoE model")
    ax2.set_xlabel("end-effector x")
    ax2.set_ylabel("end-effector y")
    ax2.axis("equal")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "ch04_forward_kinematics_poe.png"
    fig.savefig(output_path, dpi=160)
    print(f"Saved figure: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main(show="--no-show" not in sys.argv)
