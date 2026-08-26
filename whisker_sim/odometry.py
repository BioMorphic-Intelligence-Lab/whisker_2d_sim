import numpy as np

from .geometry import rotation_matrix, wrap_angle


class DriftingOdometry:
    """Flowdeck-like dead reckoning with velocity and gyro bias drift."""

    def __init__(self, initial_state, config, rng):
        self.state = np.asarray(initial_state, dtype=float).copy()
        self.config = config
        self.rng = rng

        self.velocity_bias = config.initial_velocity_bias.copy()
        self.gyro_bias = float(config.initial_gyro_bias)

    def update(self, true_velocity_body, true_yaw_rate):
        dt = self.config.dt

        self.velocity_bias += self.rng.normal(
            0.0,
            self.config.velocity_bias_rw_std,
            size=2,
        ) * np.sqrt(dt)

        self.gyro_bias += self.rng.normal(
            0.0,
            self.config.gyro_bias_rw_std,
        ) * np.sqrt(dt)

        measured_velocity_body = (
            np.asarray(true_velocity_body, dtype=float)
            + self.velocity_bias
            + self.rng.normal(
                0.0,
                self.config.velocity_noise_std,
                size=2,
            )
        )

        measured_yaw_rate = (
            float(true_yaw_rate)
            + self.gyro_bias
            + self.rng.normal(
                0.0,
                self.config.gyro_noise_std,
            )
        )

        self.state[2] = wrap_angle(
            self.state[2] + measured_yaw_rate * dt
        )

        self.state[:2] += (
            rotation_matrix(self.state[2])
            @ measured_velocity_body
            * dt
        )

        return (
            self.state.copy(),
            measured_velocity_body.copy(),
            float(measured_yaw_rate),
        )
