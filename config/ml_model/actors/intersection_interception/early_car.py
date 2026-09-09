"""Early variant using interception_car.py's cut-in geometry and logic."""

from config.ml_model.actors.intersection_interception.common import (
    InterceptionScenarioConfig,
)


# Initial tuning value: release immediately when monitoring starts. Validate
# the resulting actual margin in CARLA using the recorded distance/TTC fields.
TRIGGER_DISTANCE_M = 0.0

SCENARIO_CONFIG = InterceptionScenarioConfig(
    scenario_name="Early Interception",
    scenario_slug="early_interception",
    severity="EARLY",
    trigger_distance_m=TRIGGER_DISTANCE_M,
    description="Cut-in vehicle is released immediately, providing the largest reaction margin.",
)
