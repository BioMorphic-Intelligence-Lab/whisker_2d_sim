import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisker_sim import (
    ENVIRONMENT_NAMES,
    SimulationConfig,
    environment_descriptions,
    get_environment,
)
from whisker_sim.realtime_simulation import RealtimeSimulator
from whisker_sim.realtime_animation import animate_realtime
from whisker_sim.visualization import plot_odometry_error, plot_trajectory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the TRUE ONLINE 2D whisker-drone simulator."
    )
    parser.add_argument(
        "--environment",
        "-e",
        default="easy_rectangle",
        choices=ENVIRONMENT_NAMES,
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=30.0,
        help="Visualization FPS only; control remains at 50 Hz.",
    )
    parser.add_argument(
        "--list-environments",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_environments:
        for item in environment_descriptions():
            print(
                "{:16s} {:18s} {}".format(
                    item["name"],
                    item["difficulty"],
                    item["description"],
                )
            )
        return

    config = SimulationConfig()
    room = get_environment(args.environment)

    simulator = RealtimeSimulator(
        config=config,
        room=room,
    )

    print("TRUE ONLINE simulation")
    print("Environment: {}".format(room.name))
    print("Control/sensing/odometry: {:.1f} Hz".format(1.0 / config.dt))
    print("Visualization: {:.1f} Hz".format(args.render_fps))
    print("Whisker bases: left=(0.09,+0.025), right=(0.09,-0.025) m")

    animate_realtime(
        simulator,
        render_fps=args.render_fps,
    )

    result = simulator.get_result()

    print("Recorded control samples: {}".format(len(result.times)))
    print("Max scheduler lateness: {:.3f} ms".format(
        simulator.max_lateness * 1000.0
    ))
    print("Overruns (>20 ms): {}".format(simulator.overrun_count))

    if len(result.times) == 0:
        return

    output_dir = PROJECT_ROOT / "outputs" / room.name
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_trajectory(
        result,
        output_path=output_dir / "trajectory.png",
        show=False,
    )
    plot_odometry_error(
        result,
        output_path=output_dir / "odometry_error.png",
        show=False,
    )


if __name__ == "__main__":
    main()
