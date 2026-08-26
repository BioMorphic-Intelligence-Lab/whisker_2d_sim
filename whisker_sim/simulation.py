from dataclasses import dataclass
import numpy as np

from .config import SimulationConfig
from .controller import TactileServoFSM
from .geometry import rotation_matrix, wrap_angle
from .map import RoomMap
from .odometry import DriftingOdometry
from .sensors import WhiskerRig


@dataclass
class SimulationResult:
    config: object
    room: object
    whiskers: object
    times: np.ndarray
    true_states: np.ndarray
    odom_states: np.ndarray
    command_velocity_body: np.ndarray
    command_yaw_rate: np.ndarray
    fsm_states: list
    fsm_state_elapsed: np.ndarray
    target_yaws: np.ndarray
    velocity_bias: np.ndarray
    gyro_bias: np.ndarray
    measurements: list


def run_simulation(config=None, room=None):
    """Run the velocity-controlled tactile exploration simulation."""
    config = config or SimulationConfig()
    room = room or RoomMap.default_rectangle()

    rng = np.random.default_rng(config.random_seed)

    whiskers = WhiskerRig(
        forward_offset=config.whisker_forward_offset,
        spacing=config.whisker_spacing,
        max_range=config.whisker_range,
        range_noise_std=config.whisker_range_noise_std,
    )

    controller = TactileServoFSM(config, rng)

    true_state = config.initial_state.copy()

    odometry = DriftingOdometry(
        true_state,
        config,
        rng,
    )

    times = np.arange(config.num_steps, dtype=float) * config.dt
    true_states = np.zeros((config.num_steps, 3), dtype=float)
    odom_states = np.zeros((config.num_steps, 3), dtype=float)
    command_velocity_body = np.zeros((config.num_steps, 2), dtype=float)
    command_yaw_rate = np.zeros(config.num_steps, dtype=float)
    fsm_states = []
    fsm_state_elapsed = np.zeros(config.num_steps, dtype=float)
    target_yaws = np.full(config.num_steps, np.nan, dtype=float)
    velocity_bias = np.zeros((config.num_steps, 2), dtype=float)
    gyro_bias = np.zeros(config.num_steps, dtype=float)
    measurement_log = []

    for step in range(config.num_steps):
        true_states[step] = true_state

        measurements = whiskers.sense(
            true_state,
            room,
            rng,
        )
        measurement_log.append(measurements)

        (
            velocity_body,
            yaw_rate,
            state_name,
            state_elapsed,
            target_yaw,
        ) = controller.update(
            measurements,
            current_yaw=true_state[2],
            dt=config.dt,
        )

        command_velocity_body[step] = velocity_body
        command_yaw_rate[step] = yaw_rate
        fsm_states.append(state_name)
        fsm_state_elapsed[step] = state_elapsed

        if target_yaw is not None:
            target_yaws[step] = target_yaw

        odom_state, _, _ = odometry.update(
            velocity_body,
            yaw_rate,
        )

        odom_states[step] = odom_state
        velocity_bias[step] = odometry.velocity_bias
        gyro_bias[step] = odometry.gyro_bias

        # Ideal low-level velocity tracking for this first 2D study.
        true_state[:2] += (
            rotation_matrix(true_state[2])
            @ velocity_body
            * config.dt
        )

        true_state[2] = wrap_angle(
            true_state[2]
            + yaw_rate * config.dt
        )

    return SimulationResult(
        config=config,
        room=room,
        whiskers=whiskers,
        times=times,
        true_states=true_states,
        odom_states=odom_states,
        command_velocity_body=command_velocity_body,
        command_yaw_rate=command_yaw_rate,
        fsm_states=fsm_states,
        fsm_state_elapsed=fsm_state_elapsed,
        target_yaws=target_yaws,
        velocity_bias=velocity_bias,
        gyro_bias=gyro_bias,
        measurements=measurement_log,
    )
