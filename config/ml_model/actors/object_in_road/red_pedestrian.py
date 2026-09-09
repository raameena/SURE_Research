"""Stationary Object-in-Road pedestrian using CARLA's red-clothed walker."""

import logging

from config.ml_model.actors.object_in_road.pedestrian import (
    resolve_actual_perception,
    resolve_expected,
    spawn_pedestrian_ahead,
)


# CARLA 0.9.16's walkers do not expose a configurable ``color`` attribute.
# This packaged adult walker has a fixed red upper-body clothing texture.
RED_SHIRT_BLUEPRINT_ID = "walker.pedestrian.0038"


def find_object_blueprint(bp_lib):
    """Return CARLA 0.9.16's fixed red-clothed pedestrian blueprint."""
    try:
        pedestrian_bp = bp_lib.find(RED_SHIRT_BLUEPRINT_ID)
    except (IndexError, RuntimeError) as error:
        raise RuntimeError(
            f"Required red-clothed pedestrian blueprint {RED_SHIRT_BLUEPRINT_ID!r} "
            "is unavailable in this CARLA build."
        ) from error

    if pedestrian_bp.has_attribute("is_invincible"):
        pedestrian_bp.set_attribute("is_invincible", "false")

    logging.info(
        "Red-shirt pedestrian blueprint selected: %s.",
        pedestrian_bp.id,
    )
    return pedestrian_bp


def spawn_object_ahead(world, bp_lib, ego_waypoint, distance_ahead_m):
    """Spawn the red-shirt pedestrian with the standard pedestrian geometry."""
    return spawn_pedestrian_ahead(
        world,
        bp_lib,
        ego_waypoint,
        distance_ahead_m,
        blueprint_selector=find_object_blueprint,
    )
