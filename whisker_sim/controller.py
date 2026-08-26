from enum import Enum, auto
import numpy as np

from .geometry import wrap_angle


class FlightState(Enum):
    """Minimal exploration FSM before adding GPIS."""

    FORWARD = auto()
    CF_ACTION = auto()
    BACKWARD = auto()
    CHANGE_DIRECTION = auto()


class TactileServoFSM:
    """Article-inspired contact-flight FSM with random reorientation."""

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.state = FlightState.FORWARD
        self.state_elapsed = 0.0
        self.target_yaw = None

    def _enter(self, new_state, current_yaw=None):
        self.state = new_state
        self.state_elapsed = 0.0

        if new_state == FlightState.CHANGE_DIRECTION:
            self.target_yaw = self._sample_random_target_yaw(current_yaw)
        else:
            self.target_yaw = None

    def _sample_random_target_yaw(self, current_yaw):
        magnitude = self.rng.uniform(
            self.config.random_turn_min,
            self.config.random_turn_max,
        )
        sign = self.rng.choice([-1.0, 1.0])
        return wrap_angle(current_yaw + sign * magnitude)

    @staticmethod
    def _measurement_dict(measurements):
        return {measurement.name: measurement for measurement in measurements}

    @staticmethod
    def _has_contact(measurements):
        return any(measurement.contact for measurement in measurements)

    def _cf_action(self, measurements):
        """Article-inspired tactile servo with a 50-100 mm depth band.

        The simulator reports geometric range from the whisker base to the
        wall. The controller converts that range to contact depth:

            contact_depth = whisker_length - measured_range

        For a 200 mm whisker, the 50-100 mm depth band is equivalent to a
        base-to-wall range of 100-150 mm.
        """
        by_name = self._measurement_dict(measurements)
        left = by_name["left"]
        right = by_name["right"]

        left_depth = (
            self.config.whisker_range - left.distance
            if left.contact and left.distance is not None
            else 0.0
        )
        right_depth = (
            self.config.whisker_range - right.distance
            if right.contact and right.distance is not None
            else 0.0
        )

        min_depth = self.config.tactile_min_depth
        max_depth = self.config.tactile_max_depth

        left_in_band = min_depth < left_depth < max_depth
        right_in_band = min_depth < right_depth < max_depth

        left_too_shallow = left_depth <= min_depth
        right_too_shallow = right_depth <= min_depth

        left_too_deep = left_depth >= max_depth
        right_too_deep = right_depth >= max_depth

        # Both whiskers are too shallow / no useful contact -> approach.
        if left_too_shallow and right_too_shallow:
            return np.array(
                [self.config.tactile_forward_speed, 0.0],
                dtype=float,
            ), 0.0

        # Both whiskers lie inside the desired 50-100 mm depth band -> sweep.
        if left_in_band and right_in_band:
            return np.array(
                [0.0, -self.config.sweep_right_speed],
                dtype=float,
            ), 0.0

        # Left has sufficient contact but right is too shallow -> yaw.
        if (left_in_band or left_too_deep) and right_too_shallow:
            return np.zeros(2, dtype=float), +self.config.max_turn_rate

        # Right has sufficient contact but left is too shallow -> opposite yaw.
        if (right_in_band or right_too_deep) and left_too_shallow:
            return np.zeros(2, dtype=float), -self.config.max_turn_rate

        # Remaining cases include excessive penetration -> back off slowly.
        return np.array(
            [-self.config.tactile_backward_speed, 0.0],
            dtype=float,
        ), 0.0

    def update(self, measurements, current_yaw, dt):
        """Advance the FSM and return velocity/yaw-rate commands."""

        if self.state == FlightState.FORWARD:
            if self._has_contact(measurements):
                self._enter(FlightState.CF_ACTION)

        elif self.state == FlightState.CF_ACTION:
            if self.state_elapsed >= self.config.sweep_duration:
                self._enter(FlightState.BACKWARD)

        elif self.state == FlightState.BACKWARD:
            if self.state_elapsed >= self.config.backward_duration:
                self._enter(FlightState.CHANGE_DIRECTION, current_yaw=current_yaw)

        elif self.state == FlightState.CHANGE_DIRECTION:
            yaw_error = wrap_angle(self.target_yaw - current_yaw)
            if abs(yaw_error) <= self.config.yaw_tolerance:
                self._enter(FlightState.FORWARD)

        if self.state == FlightState.FORWARD:
            velocity_body = np.array([self.config.forward_speed, 0.0])
            yaw_rate = 0.0

        elif self.state == FlightState.CF_ACTION:
            velocity_body, yaw_rate = self._cf_action(measurements)

        elif self.state == FlightState.BACKWARD:
            velocity_body = np.array([-self.config.backward_speed, 0.0])
            yaw_rate = 0.0

        elif self.state == FlightState.CHANGE_DIRECTION:
            velocity_body = np.zeros(2)
            yaw_error = wrap_angle(self.target_yaw - current_yaw)
            yaw_rate = np.sign(yaw_error) * self.config.max_turn_rate
            if abs(yaw_error) < abs(yaw_rate * dt):
                yaw_rate = yaw_error / dt

        else:
            raise RuntimeError("Unhandled FSM state.")

        state_name = self.state.name
        state_elapsed = self.state_elapsed
        target_yaw = self.target_yaw
        self.state_elapsed += dt

        return velocity_body, float(yaw_rate), state_name, state_elapsed, target_yaw
