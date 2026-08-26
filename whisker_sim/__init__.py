"""Lightweight 2D whisker localization simulator."""

from .config import SimulationConfig
from .controller import FlightState, TactileServoFSM
from .environments import (
    ENVIRONMENT_NAMES,
    environment_descriptions,
    get_environment,
)
from .map import RoomMap
from .odometry import DriftingOdometry
from .sensors import WhiskerRig
from .simulation import SimulationResult, run_simulation

__all__ = [
    "SimulationConfig",
    "FlightState",
    "TactileServoFSM",
    "ENVIRONMENT_NAMES",
    "environment_descriptions",
    "get_environment",
    "RoomMap",
    "DriftingOdometry",
    "WhiskerRig",
    "SimulationResult",
    "run_simulation",
]
