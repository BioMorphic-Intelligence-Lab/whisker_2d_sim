"""True online 50 Hz simulator.

The simulation/control loop runs in its own thread and advances from wall-clock
time. Visualization only reads the latest snapshot; it never drives the
controller and therefore cannot skip control iterations.
"""

from dataclasses import dataclass
import threading
import time

import numpy as np

from .config import SimulationConfig
from .controller import TactileServoFSM
from .geometry import rotation_matrix, wrap_angle
from .map import RoomMap
from .odometry import DriftingOdometry
from .sensors import WhiskerRig
from .simulation import SimulationResult


@dataclass
class RealtimeSnapshot:
    sim_time: float
    true_state: np.ndarray
    odom_state: np.ndarray
    velocity_body: np.ndarray
    yaw_rate: float
    fsm_state: str
    fsm_state_elapsed: float
    target_yaw: object
    velocity_bias: np.ndarray
    gyro_bias: float
    measurements: list
    control_step: int
    timing_lateness: float


class RealtimeSimulator:
    """Run sensing, FSM, odometry, and vehicle motion online at 50 Hz."""

    def __init__(self, config=None, room=None):
        self.config = config or SimulationConfig()
        self.room = room or RoomMap.default_rectangle()

        self._lock = threading.RLock()
        self._thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        self._reset_internal_state()

    def _reset_internal_state(self):
        cfg = self.config

        self.rng = np.random.default_rng(cfg.random_seed)

        self.whiskers = WhiskerRig(
            forward_offset=cfg.whisker_forward_offset,
            spacing=cfg.whisker_spacing,
            max_range=cfg.whisker_range,
            range_noise_std=cfg.whisker_range_noise_std,
        )

        self.controller = TactileServoFSM(cfg, self.rng)

        self.true_state = cfg.initial_state.copy()

        self.odometry = DriftingOdometry(
            self.true_state,
            cfg,
            self.rng,
        )

        self.sim_time = 0.0
        self.control_step = 0
        self.max_lateness = 0.0
        self.overrun_count = 0

        self.times = []
        self.true_states = []
        self.odom_states = []
        self.command_velocity_body = []
        self.command_yaw_rate = []
        self.fsm_states = []
        self.fsm_state_elapsed = []
        self.target_yaws = []
        self.velocity_bias = []
        self.gyro_bias = []
        self.measurement_log = []
        self.timing_lateness = []

        # Generate an initial no-motion sensor snapshot so the GUI has
        # something valid to draw before the first 20 ms control tick.
        initial_measurements = self.whiskers.sense(
            self.true_state,
            self.room,
            self.rng,
        )

        self.latest_snapshot = RealtimeSnapshot(
            sim_time=0.0,
            true_state=self.true_state.copy(),
            odom_state=self.odometry.state.copy(),
            velocity_body=np.zeros(2, dtype=float),
            yaw_rate=0.0,
            fsm_state=self.controller.state.name,
            fsm_state_elapsed=0.0,
            target_yaw=None,
            velocity_bias=self.odometry.velocity_bias.copy(),
            gyro_bias=float(self.odometry.gyro_bias),
            measurements=initial_measurements,
            control_step=0,
            timing_lateness=0.0,
        )

    @property
    def running(self):
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop_event.is_set()
        )

    @property
    def paused(self):
        return self._pause_event.is_set()

    def start(self):
        """Start the real-time 50 Hz control/simulation thread."""
        if self.running:
            return

        self._stop_event.clear()
        self._pause_event.clear()

        self._thread = threading.Thread(
            target=self._control_loop,
            name="whisker-sim-50hz",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Request simulator shutdown."""
        self._stop_event.set()

        if (
            self._thread is not None
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=2.0)

    def pause(self):
        """Pause both simulation time and control updates."""
        self._pause_event.set()

    def resume(self):
        """Resume the online simulator."""
        self._pause_event.clear()

    def toggle_pause(self):
        if self.paused:
            self.resume()
        else:
            self.pause()

    def restart(self):
        """Restart the online experiment from t=0.

        The thread is stopped, internal state/logs are reset, then the 50 Hz
        loop starts again.
        """
        self.stop()

        with self._lock:
            self._stop_event = threading.Event()
            self._pause_event = threading.Event()
            self._reset_internal_state()

        self.start()

    def _control_loop(self):
        """Soft-real-time periodic loop driven by time.perf_counter()."""
        dt = self.config.dt
        next_tick = time.perf_counter()

        while not self._stop_event.is_set():
            if self.sim_time >= self.config.sim_time:
                break

            if self._pause_event.is_set():
                # While paused, simulation time does not advance.
                time.sleep(0.005)
                next_tick = time.perf_counter() + dt
                continue

            now = time.perf_counter()

            if now < next_tick:
                time.sleep(next_tick - now)
                now = time.perf_counter()

            lateness = max(0.0, now - next_tick)

            # A lateness above one whole control period means the process
            # missed a 50 Hz deadline. We count it; we do NOT skip a control
            # update.
            if lateness > dt:
                self.overrun_count += 1

            self.max_lateness = max(self.max_lateness, lateness)

            with self._lock:
                self._step(lateness)

            # Fixed-period scheduler: advance the next deadline by exactly dt.
            # If one iteration was late, the next iteration may execute sooner
            # to catch back up; no controller sample is skipped.
            next_tick += dt

        self._stop_event.set()

    def _step(self, lateness):
        """Execute exactly one 20 ms sensing/control/dynamics iteration."""
        cfg = self.config

        # 1. Whisker sensing from the CURRENT ground-truth state.
        measurements = self.whiskers.sense(
            self.true_state,
            self.room,
            self.rng,
        )

        # 2. High-level velocity-control FSM.
        (
            velocity_body,
            yaw_rate,
            state_name,
            state_elapsed,
            target_yaw,
        ) = self.controller.update(
            measurements,
            current_yaw=self.true_state[2],
            dt=cfg.dt,
        )

        # 3. Flowdeck/gyro-like drifting dead reckoning.
        odom_state, _, _ = self.odometry.update(
            velocity_body,
            yaw_rate,
        )

        # 4. Log this control sample BEFORE integrating to the next pose.
        self.times.append(float(self.sim_time))
        self.true_states.append(self.true_state.copy())
        self.odom_states.append(odom_state.copy())
        self.command_velocity_body.append(
            np.asarray(velocity_body, dtype=float).copy()
        )
        self.command_yaw_rate.append(float(yaw_rate))
        self.fsm_states.append(state_name)
        self.fsm_state_elapsed.append(float(state_elapsed))
        self.target_yaws.append(
            np.nan if target_yaw is None else float(target_yaw)
        )
        self.velocity_bias.append(
            self.odometry.velocity_bias.copy()
        )
        self.gyro_bias.append(
            float(self.odometry.gyro_bias)
        )
        self.measurement_log.append(measurements)
        self.timing_lateness.append(float(lateness))

        # 5. Ideal low-level velocity tracking for one dt interval.
        self.true_state[:2] += (
            rotation_matrix(self.true_state[2])
            @ velocity_body
            * cfg.dt
        )

        self.true_state[2] = wrap_angle(
            self.true_state[2]
            + yaw_rate * cfg.dt
        )

        self.control_step += 1
        self.sim_time += cfg.dt

        # 6. Publish latest state for visualization.
        self.latest_snapshot = RealtimeSnapshot(
            sim_time=float(self.sim_time),
            true_state=self.true_state.copy(),
            odom_state=odom_state.copy(),
            velocity_body=np.asarray(
                velocity_body,
                dtype=float,
            ).copy(),
            yaw_rate=float(yaw_rate),
            fsm_state=state_name,
            fsm_state_elapsed=float(state_elapsed),
            target_yaw=target_yaw,
            velocity_bias=self.odometry.velocity_bias.copy(),
            gyro_bias=float(self.odometry.gyro_bias),
            measurements=measurements,
            control_step=self.control_step,
            timing_lateness=float(lateness),
        )

    def get_snapshot(self):
        """Thread-safe copy of the latest online state."""
        with self._lock:
            s = self.latest_snapshot

            return RealtimeSnapshot(
                sim_time=s.sim_time,
                true_state=s.true_state.copy(),
                odom_state=s.odom_state.copy(),
                velocity_body=s.velocity_body.copy(),
                yaw_rate=s.yaw_rate,
                fsm_state=s.fsm_state,
                fsm_state_elapsed=s.fsm_state_elapsed,
                target_yaw=s.target_yaw,
                velocity_bias=s.velocity_bias.copy(),
                gyro_bias=s.gyro_bias,
                measurements=list(s.measurements),
                control_step=s.control_step,
                timing_lateness=s.timing_lateness,
            )

    def get_result(self):
        """Return all samples accumulated so far in SimulationResult format."""
        with self._lock:
            n = len(self.times)

            if n == 0:
                return SimulationResult(
                    config=self.config,
                    room=self.room,
                    whiskers=self.whiskers,
                    times=np.zeros(0),
                    true_states=np.zeros((0, 3)),
                    odom_states=np.zeros((0, 3)),
                    command_velocity_body=np.zeros((0, 2)),
                    command_yaw_rate=np.zeros(0),
                    fsm_states=[],
                    fsm_state_elapsed=np.zeros(0),
                    target_yaws=np.zeros(0),
                    velocity_bias=np.zeros((0, 2)),
                    gyro_bias=np.zeros(0),
                    measurements=[],
                )

            return SimulationResult(
                config=self.config,
                room=self.room,
                whiskers=self.whiskers,
                times=np.asarray(self.times, dtype=float),
                true_states=np.asarray(
                    self.true_states,
                    dtype=float,
                ),
                odom_states=np.asarray(
                    self.odom_states,
                    dtype=float,
                ),
                command_velocity_body=np.asarray(
                    self.command_velocity_body,
                    dtype=float,
                ),
                command_yaw_rate=np.asarray(
                    self.command_yaw_rate,
                    dtype=float,
                ),
                fsm_states=list(self.fsm_states),
                fsm_state_elapsed=np.asarray(
                    self.fsm_state_elapsed,
                    dtype=float,
                ),
                target_yaws=np.asarray(
                    self.target_yaws,
                    dtype=float,
                ),
                velocity_bias=np.asarray(
                    self.velocity_bias,
                    dtype=float,
                ),
                gyro_bias=np.asarray(
                    self.gyro_bias,
                    dtype=float,
                ),
                measurements=list(self.measurement_log),
            )
