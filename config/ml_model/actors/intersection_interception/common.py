"""Shared configuration adapter for the three interception variants.

Geometry and actor behavior intentionally come from the existing
``config.ml_model.actors.interception_car`` implementation.  This module
only gives Early/Medium/Late a common immutable configuration shape and
stable names for the shared runner; it does not duplicate that logic.
"""

from dataclasses import dataclass

from config.CARLA_actors.crash_car import (
    generate_left_turn_path as generate_interception_path,
    spawn_crash_car_relative_to_vehicle as spawn_interception_vehicle,
    stop_crash_car,
)
from config.ml_model.actors.interception_car import (
    CRASH_CAR_FORWARD_DISTANCE,
    CRASH_CAR_LANE_EXTENSION_M,
    CRASH_CAR_SETTLE_DISTANCE_M,
    CRASH_CAR_SIDE_OFFSET,
    CRASH_CAR_SPEED,
    compute_gap_to_crash_car_m as compute_distance_to_interception_vehicle_m,
    make_cutin_controller as make_interception_controller,
    resolve_actual_perception,
    resolve_expected,
)


# The variant family retains a shared timeout long enough for its distance
# triggers to remain distinguishable. The cut-in controller itself is the
# implementation from interception_car.py; this is only a passed parameter.
TRIGGER_MAX_SECONDS = 8.0


@dataclass(frozen=True)
class InterceptionScenarioConfig:
    """Variant identity plus its one experimental factor: trigger distance."""

    scenario_name: str
    scenario_slug: str
    severity: str
    trigger_distance_m: float
    description: str
    interception_vehicle_speed_mps: float = CRASH_CAR_SPEED
    interception_vehicle_forward_distance_m: float = CRASH_CAR_FORWARD_DISTANCE
    interception_vehicle_side_offset_m: float = CRASH_CAR_SIDE_OFFSET
    path_step_m: float = 2.0
    path_extension_m: float = CRASH_CAR_LANE_EXTENSION_M
    settle_distance_m: float = CRASH_CAR_SETTLE_DISTANCE_M
    trigger_max_seconds: float = TRIGGER_MAX_SECONDS


__all__ = [
    "InterceptionScenarioConfig",
    "compute_distance_to_interception_vehicle_m",
    "generate_interception_path",
    "make_interception_controller",
    "resolve_actual_perception",
    "resolve_expected",
    "spawn_interception_vehicle",
    "stop_crash_car",
]
