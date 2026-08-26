from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimulationConfig:
    """Top-level configuration for the 2D whisker simulator."""

    dt: float = 0.02
    sim_time: float = 120.0
    random_seed: int = 10

    playback_speed: float = 1.0

    # ------------------------------------------------------------------
    # Whisker mounting geometry
    # Body frame: +x forward, +y left.
    #
    # Left base  = (0.09, +0.025) m
    # Right base = (0.09, -0.025) m
    # ------------------------------------------------------------------
    whisker_forward_offset: float = 0.09
    whisker_spacing: float = 0.05
    whisker_range: float = 0.20
    whisker_range_noise_std: float = 0.015

    # ------------------------------------------------------------------
    # Mission/FSM velocities
    # ------------------------------------------------------------------
    forward_speed: float = 0.20
    sweep_right_speed: float = 0.20
    sweep_duration: float = 2.0
    backward_speed: float = 0.20
    backward_duration: float = 2.0

    # Random direction change: temporary replacement for GPIS/planner.
    max_turn_rate: float = np.deg2rad(25.0)
    yaw_tolerance: float = np.deg2rad(1.0)
    random_turn_min: float = np.deg2rad(45.0)
    random_turn_max: float = np.deg2rad(135.0)

    # ------------------------------------------------------------------
    # Tactile contact-depth band
    #
    # depth = whisker_range - measured_base_to_wall_range
    #
    # 50-100 mm depth corresponds to a 100-150 mm base-to-wall range for
    # a 200 mm whisker.
    # ------------------------------------------------------------------
    tactile_min_depth: float = 0.05
    tactile_max_depth: float = 0.10

    tactile_forward_speed: float = 0.10
    tactile_backward_speed: float = 0.10

    # ------------------------------------------------------------------
    # Flowdeck-like velocity sensor
    # ------------------------------------------------------------------
    velocity_noise_std: float = 0.003
    velocity_bias_rw_std: float = 0.0008
    initial_velocity_bias: np.ndarray = field(
        default_factory=lambda: np.array([0.002, -0.0015], dtype=float)
    )

    # ------------------------------------------------------------------
    # Gyroscope
    # ------------------------------------------------------------------
    gyro_noise_std: float = np.deg2rad(0.08)
    gyro_bias_rw_std: float = np.deg2rad(0.025)
    initial_gyro_bias: float = np.deg2rad(0.08)

    # Start at world origin, facing +x.
    initial_state: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )

    @property
    def num_steps(self) -> int:
        return int(self.sim_time / self.dt)
