"""Live visualization for RealtimeSimulator.

The GUI never advances simulation time. It only reads the newest state produced
by the independent 50 Hz simulation/control thread.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from .geometry import rotation_matrix


def _drone_triangle(state, length=0.09, width=0.065):
    x, y, yaw = state

    body = np.array(
        [
            [length / 2.0, 0.0],
            [-length / 2.0, +width / 2.0],
            [-length / 2.0, -width / 2.0],
        ],
        dtype=float,
    )

    world = body @ rotation_matrix(yaw).T
    world[:, 0] += x
    world[:, 1] += y
    return world


def animate_realtime(simulator, render_fps=30.0):
    """Open the live window and run the simulator online."""
    cfg = simulator.config
    room = simulator.room

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title(
        "LIVE Whisker 2D Simulator - {}".format(room.name)
    )

    room_closed = np.vstack(
        [room.vertices, room.vertices[0]]
    )

    ax.plot(
        room_closed[:, 0],
        room_closed[:, 1],
        linewidth=2.5,
        label="Room boundary",
    )

    for wall_id, texture in enumerate(room.textures):
        start, end = room.segment(wall_id)
        midpoint = 0.5 * (start + end)
        ax.text(
            midpoint[0],
            midpoint[1],
            " {}".format(texture),
            fontsize=7,
            va="center",
        )

    gt_trail, = ax.plot([], [], linewidth=2.0, label="Ground truth")
    odom_trail, = ax.plot(
        [],
        [],
        linestyle="--",
        linewidth=1.8,
        label="Dead reckoning",
    )

    initial = simulator.get_snapshot()

    gt_drone = Polygon(
        _drone_triangle(initial.true_state),
        closed=True,
        alpha=0.75,
    )
    odom_drone = Polygon(
        _drone_triangle(initial.odom_state),
        closed=True,
        fill=False,
        linewidth=1.8,
    )
    ax.add_patch(gt_drone)
    ax.add_patch(odom_drone)

    left_whisker, = ax.plot([], [], linewidth=2.0)
    right_whisker, = ax.plot([], [], linewidth=2.0)

    left_base, = ax.plot(
        [],
        [],
        marker="s",
        linestyle="None",
        markersize=4,
    )
    right_base, = ax.plot(
        [],
        [],
        marker="s",
        linestyle="None",
        markersize=4,
    )

    left_contact, = ax.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=6,
    )
    right_contact, = ax.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=6,
    )

    status = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
    )

    x_min, x_max, y_min, y_max = room.plot_bounds(margin=0.25)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    ax.legend(loc="lower left")

    ax.set_title(
        "{} | TRUE ONLINE 50 Hz control\\n"
        "Space: pause/resume simulation | R: restart".format(
            room.name
        )
    )

    def set_whisker(
        line,
        base_artist,
        contact_artist,
        measurement,
        drone_state,
    ):
        rotation = rotation_matrix(drone_state[2])
        direction = rotation @ np.array([1.0, 0.0])

        base = measurement.base_world
        base_artist.set_data([base[0]], [base[1]])

        if measurement.contact:
            tip = measurement.point_world
            contact_artist.set_data(
                [tip[0]],
                [tip[1]],
            )
        else:
            tip = base + cfg.whisker_range * direction
            contact_artist.set_data([], [])

        line.set_data(
            [base[0], tip[0]],
            [base[1], tip[1]],
        )

    def update(_):
        snapshot = simulator.get_snapshot()
        result = simulator.get_result()

        gt_drone.set_xy(
            _drone_triangle(snapshot.true_state)
        )
        odom_drone.set_xy(
            _drone_triangle(snapshot.odom_state)
        )

        if len(result.times) > 0:
            gt_trail.set_data(
                result.true_states[:, 0],
                result.true_states[:, 1],
            )
            odom_trail.set_data(
                result.odom_states[:, 0],
                result.odom_states[:, 1],
            )

        measurements = {
            m.name: m for m in snapshot.measurements
        }

        left = measurements["left"]
        right = measurements["right"]

        set_whisker(
            left_whisker,
            left_base,
            left_contact,
            left,
            snapshot.true_state,
        )
        set_whisker(
            right_whisker,
            right_base,
            right_contact,
            right,
            snapshot.true_state,
        )

        gt = snapshot.true_state
        odom = snapshot.odom_state
        command = snapshot.velocity_body
        command_yaw_rate = np.rad2deg(snapshot.yaw_rate)

        position_error = np.linalg.norm(
            odom[:2] - gt[:2]
        )

        if snapshot.target_yaw is None:
            target_yaw_text = "-"
        else:
            target_yaw_text = "{:+.1f} deg".format(
                np.rad2deg(snapshot.target_yaw)
            )

        if left.contact:
            left_text = (
                "range={:4.1f} cm, depth={:4.0f} mm".format(
                    left.distance * 100.0,
                    (cfg.whisker_range - left.distance) * 1000.0,
                )
            )
        else:
            left_text = "no contact"

        if right.contact:
            right_text = (
                "range={:4.1f} cm, depth={:4.0f} mm".format(
                    right.distance * 100.0,
                    (cfg.whisker_range - right.distance) * 1000.0,
                )
            )
        else:
            right_text = "no contact"

        texture = (
            left.texture
            if left.contact
            else (
                right.texture
                if right.contact
                else "-"
            )
        )

        status.set_text(
            "sim time: {:5.2f} s\n"
            "FSM: {}\n"
            "target yaw: {}\n"
            "cmd body v: ({:+.3f}, {:+.3f}) m/s\n"
            "cmd yaw rate: {:+.2f} deg/s\n"
            "GT:   ({:+.3f}, {:+.3f}, {:+.2f} deg)\n"
            "odom: ({:+.3f}, {:+.3f}, {:+.2f} deg)\n"
            "position drift: {:5.1f} cm\n"
            "left:  {}\n"
            "right: {}\n"
            "texture: {}\n"
            "deadline lateness: {:5.2f} ms".format(
                snapshot.sim_time,
                snapshot.fsm_state,
                target_yaw_text,
                command[0],
                command[1],
                command_yaw_rate,
                gt[0],
                gt[1],
                np.rad2deg(gt[2]),
                odom[0],
                odom[1],
                np.rad2deg(odom[2]),
                position_error * 100.0,
                left_text,
                right_text,
                texture,
                snapshot.timing_lateness * 1000.0,
            )
        )

        # Close only after the final state has had a chance to render.
        if (
            simulator._stop_event.is_set()
            and snapshot.sim_time >= cfg.sim_time
        ):
            animation.event_source.stop()

        return (
            gt_trail,
            odom_trail,
            gt_drone,
            odom_drone,
            left_whisker,
            right_whisker,
            left_base,
            right_base,
            left_contact,
            right_contact,
            status,
        )

    interval_ms = max(
        1,
        int(round(1000.0 / float(render_fps))),
    )

    animation = FuncAnimation(
        fig,
        update,
        interval=interval_ms,
        blit=False,
        cache_frame_data=False,
    )

    def on_key(event):
        if event.key == " ":
            simulator.toggle_pause()

        elif event.key and event.key.lower() == "r":
            simulator.restart()

    def on_close(_event):
        simulator.stop()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    # IMPORTANT: start only once the figure and callbacks already exist.
    simulator.start()

    plt.tight_layout()
    plt.show()

    simulator.stop()
    return animation
