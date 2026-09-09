"""Medium variant using interception_car.py's cut-in geometry and logic."""

from config.ml_model.actors.intersection_interception.common import (
    InterceptionScenarioConfig,
)


# Existing T-bone scenario reference value (15 feet). Live validation is
# still required because the ego is model-driven and its acceleration varies.
TRIGGER_DISTANCE_M = 15.0 * 0.3048

SCENARIO_CONFIG = InterceptionScenarioConfig(
    scenario_name="Medium Interception",
    scenario_slug="medium_interception",
    severity="MEDIUM",
    trigger_distance_m=TRIGGER_DISTANCE_M,
    description="Cut-in vehicle is released after the ego travels 15 feet.",
)
