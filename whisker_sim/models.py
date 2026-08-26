from dataclasses import dataclass
import numpy as np


@dataclass
class RaycastHit:
    """Result of a room raycast."""

    contact: bool
    distance: object
    point: object
    wall_id: object
    texture: object


@dataclass
class WhiskerMeasurement:
    """One whisker measurement at one simulation step."""

    name: str
    contact: bool
    distance: object
    point_world: object
    base_world: np.ndarray
    texture: object
    wall_id: object
