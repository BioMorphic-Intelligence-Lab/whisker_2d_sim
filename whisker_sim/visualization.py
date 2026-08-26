from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .geometry import wrap_angle


def plot_trajectory(
    result,
    output_path=None,
    show=False,
):
    """Plot final GT and dead-reckoning trajectories."""
    room_closed = np.vstack(
        [
            result.room.vertices,
            result.room.vertices[0],
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        room_closed[:, 0],
        room_closed[:, 1],
        linewidth=2,
        label="Room",
    )

    ax.plot(
        result.true_states[:, 0],
        result.true_states[:, 1],
        linewidth=2,
        label="Ground truth",
    )

    ax.plot(
        result.odom_states[:, 0],
        result.odom_states[:, 1],
        "--",
        linewidth=2,
        label="Dead reckoning",
    )

    ax.set_aspect("equal", adjustable="box")

    x_min, x_max, y_min, y_max = result.room.plot_bounds(
        margin=0.25
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    ax.legend()

    ax.set_title(
        "{} | {}".format(
            result.room.name,
            result.room.difficulty,
        )
    )

    fig.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            output_path,
            dpi=180,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_odometry_error(
    result,
    output_path=None,
    show=False,
):
    """Plot accumulated dead-reckoning position and yaw error."""
    position_error = np.linalg.norm(
        result.odom_states[:, :2]
        - result.true_states[:, :2],
        axis=1,
    )

    yaw_error = np.rad2deg(
        np.array(
            [
                wrap_angle(odom_yaw - gt_yaw)
                for odom_yaw, gt_yaw in zip(
                    result.odom_states[:, 2],
                    result.true_states[:, 2],
                )
            ]
        )
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
    )

    axes[0].plot(
        result.times,
        position_error,
    )
    axes[0].set_ylabel(
        "Position error [m]"
    )
    axes[0].grid(True)

    axes[1].plot(
        result.times,
        yaw_error,
    )
    axes[1].set_ylabel(
        "Yaw error [deg]"
    )
    axes[1].set_xlabel(
        "Time [s]"
    )
    axes[1].grid(True)

    fig.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            output_path,
            dpi=180,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
