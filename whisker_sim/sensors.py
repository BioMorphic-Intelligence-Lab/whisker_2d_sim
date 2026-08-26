import numpy as np

from .geometry import rotation_matrix
from .models import WhiskerMeasurement


class WhiskerRig:
    """Two parallel forward-pointing whiskers.

    Body-frame convention:
        +x = forward
        +y = left

    Default mounting:
        left base  = (0.09, +0.025) m
        right base = (0.09, -0.025) m

    Each whisker extends another 0.20 m in +x.
    """

    def __init__(
        self,
        forward_offset=0.09,
        spacing=0.05,
        max_range=0.20,
        range_noise_std=0.0015,
    ):
        self.forward_offset = float(forward_offset)
        self.spacing = float(spacing)
        self.max_range = float(max_range)
        self.range_noise_std = float(range_noise_std)

        half_spacing = 0.5 * self.spacing

        self.base_body = {
            "left": np.array(
                [self.forward_offset, +half_spacing],
                dtype=float,
            ),
            "right": np.array(
                [self.forward_offset, -half_spacing],
                dtype=float,
            ),
        }

        self.direction_body = np.array([1.0, 0.0], dtype=float)

    def sense(self, state, room, rng):
        """Generate noisy whisker range/contact measurements."""
        x, y, yaw = np.asarray(state, dtype=float)

        robot_position = np.array([x, y], dtype=float)
        rotation = rotation_matrix(yaw)
        direction_world = rotation @ self.direction_body

        measurements = []

        for name, base_body in self.base_body.items():
            base_world = robot_position + rotation @ base_body

            hit = room.raycast(
                base_world,
                direction_world,
                self.max_range,
            )

            if not hit.contact:
                measurements.append(
                    WhiskerMeasurement(
                        name=name,
                        contact=False,
                        distance=None,
                        point_world=None,
                        base_world=base_world,
                        texture=None,
                        wall_id=None,
                    )
                )
                continue

            measured_distance = hit.distance + rng.normal(
                0.0,
                self.range_noise_std,
            )

            measured_distance = float(
                np.clip(
                    measured_distance,
                    0.0,
                    self.max_range,
                )
            )

            measured_point = (
                base_world
                + measured_distance * direction_world
            )

            measurements.append(
                WhiskerMeasurement(
                    name=name,
                    contact=True,
                    distance=measured_distance,
                    point_world=measured_point,
                    base_world=base_world,
                    texture=hit.texture,
                    wall_id=hit.wall_id,
                )
            )

        return measurements
