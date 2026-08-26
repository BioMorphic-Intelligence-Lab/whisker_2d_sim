import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisker_sim import (
    ENVIRONMENT_NAMES,
    SimulationConfig,
    get_environment,
    run_simulation,
)
from whisker_sim.visualization import plot_odometry_error, plot_trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Run the OFFLINE fast simulator for batch experiments."
    )
    parser.add_argument(
        "--environment",
        "-e",
        default="easy_rectangle",
        choices=ENVIRONMENT_NAMES,
    )
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    config = SimulationConfig()
    config.random_seed = args.seed
    room = get_environment(args.environment)

    result = run_simulation(
        config=config,
        room=room,
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "{}_seed{}".format(room.name, args.seed)
    )
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

    print("OFFLINE batch simulation complete.")
    print("Environment: {}".format(room.name))
    print("Samples: {}".format(len(result.times)))
    print("Saved to: {}".format(output_dir))


if __name__ == "__main__":
    main()
