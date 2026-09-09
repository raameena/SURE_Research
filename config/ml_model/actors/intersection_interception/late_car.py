"""Late variant using interception_car.py's cut-in geometry and logic."""

from config.ml_model.actors.intersection_interception.common import (
    InterceptionScenarioConfig,
)


# Initial tuning value derived as twice the existing 15-foot reference.
# Validate the resulting actual distance/TTC in a live CARLA run.
TRIGGER_DISTANCE_M = 30.0 * 0.3048

SCENARIO_CONFIG = InterceptionScenarioConfig(
    scenario_name="Late Interception",
    scenario_slug="late_interception",
    severity="LATE",
    trigger_distance_m=TRIGGER_DISTANCE_M,
    description="Cut-in vehicle is released after the ego travels 30 feet.",
)
