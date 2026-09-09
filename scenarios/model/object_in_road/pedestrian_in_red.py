"""Object-in-Road scenario with a stationary pedestrian in a red shirt."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.ml_model.actors.object_in_road.red_pedestrian import (
    resolve_actual_perception,
    resolve_expected,
    spawn_object_ahead,
)
from scenarios.model.object_in_road.pedestrian_object import (
    OBJECT_DISTANCE_AHEAD_M,
    main as run_pedestrian_scenario,
)


def main():
    run_pedestrian_scenario(
        scenario_name="Pedestrian in Red",
        scenario_description=(
            f"Stationary pedestrian wearing a red shirt placed "
            f"{OBJECT_DISTANCE_AHEAD_M:.0f} m ahead of the ego vehicle."
        ),
        object_type_label="Pedestrian in Red",
        actor_spawn_object_ahead=spawn_object_ahead,
        actor_resolve_expected=resolve_expected,
        actor_resolve_actual_perception=resolve_actual_perception,
    )


if __name__ == "__main__":
    main()
