import sys
import logging
from pathlib import Path

# spawn_locations.py is inside: config/CARLA_actors/
# parents[2] goes back to: CARLA-Research/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Windows-only: single-keypress detection without waiting for Enter, so Esc
# can actually interrupt the loop -- input() would need Enter pressed after
# Esc to return at all, which defeats the point.
import msvcrt

from config.CARLA_actors.carla_config import (
    connect_to_carla,
    get_spawn_points,
    spawn_ego_vehicle_at_index,
    cleanup_actors
)

from config.CARLA_actors.sensors import (
    follow_vehicle_with_spectator
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

ESC_KEY = b"\x1b"
ENTER_KEY = b"\r"


def _wait_for_enter_or_esc():
    """
    Blocks until Enter or Esc is pressed. Returns True (Enter -- advance to
    the next spawn point) or False (Esc -- stop browsing).
    """
    while True:
        key = msvcrt.getch()

        if key == ESC_KEY:
            return False

        if key == ENTER_KEY:
            return True


def browse_spawn_points(vehicle_filter="vehicle.tesla.model3"):
    """
    Spawns a vehicle at spawn index 0, then steps forward one spawn index at
    a time each time Enter is pressed, moving the CARLA spectator camera to
    look at each one -- lets you eyeball every spawn point on the current
    map and note down the index of whichever one you like (spawn_index, the
    same value scenarios pass to spawn_ego_vehicle_at_index()/
    spawn_ego_vehicle_behind_index() elsewhere in this codebase). Press Esc
    to stop early.

    Only one vehicle exists at a time -- the previous one is destroyed
    before the next spawns. Only connect_to_carla()/get_spawn_points()/
    spawn_ego_vehicle_at_index()/cleanup_actors() and
    follow_vehicle_with_spectator() are reused here; synchronous mode,
    sensors, and traffic lights are irrelevant to just looking at spawn
    locations, so this doesn't touch any of that.
    """
    client, world, bp_lib = connect_to_carla()
    spawn_points = get_spawn_points(world)

    spawned_actors = []

    try:
        spawn_index = 0

        while spawn_index < len(spawn_points):
            cleanup_actors(spawned_actors)
            spawned_actors = []

            vehicle = spawn_ego_vehicle_at_index(
                world,
                bp_lib,
                spawn_index=spawn_index,
                vehicle_filter=vehicle_filter
            )
            spawned_actors.append(vehicle)

            follow_vehicle_with_spectator(world, vehicle)

            logging.info(
                f"Spawn index {spawn_index} (of 0-{len(spawn_points) - 1}) -- "
                f"press Enter for the next one, Esc to stop."
            )

            if not _wait_for_enter_or_esc():
                break

            spawn_index += 1

        else:
            logging.info("Reached the last spawn point.")

    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")

    finally:
        cleanup_actors(spawned_actors)


if __name__ == "__main__":
    browse_spawn_points()
