"""
pedestrian: the idle-in-road obstacle logic for scenarios/model/object_in_road/
pedestrian_object.py -- blueprint selection/spawning, and the scenario-local
error_detection resolvers (same role as config/ml_model/actors/object_in_road/chair.py
plays for scenarios/model/object_in_road/chair_object.py, and same overall
scenario shape: a single static obstacle spawned ahead of ego in its own
lane, straight-road approach, stop-only expected action).

Unlike chair.py's out-of-distribution prop, "pedestrian"/"walker" is a real
class in both of the model's own semantic segmentation heads (see controls.
ml_controls.SEMANTIC_CLASS_INFO/BEV_SEMANTIC_CLASS_INFO -- pred_bev_semantic
calls the class "walker", pred_semantic calls it "pedestrian") -- so,
following config/ml_model/actors/T_bone_car.py's precedent for this same
distinction, resolve_actual_perception() here just reads the model's own
already-existing "walker" score off pred_bev_semantic instead of chair.py's
generic occupancy-footprint proxy. See chair_object.py's
module docstring for the fuller "why a generic proxy only when there's no
real class" reasoning this mirrors.
"""

import logging

import carla

from config.CARLA_actors.pedestrian import hold_pedestrian_still

from controls.ml_controls import BEV_SEMANTIC_CLASS_INFO, MIN_SEMANTIC_PIXEL_THRESHOLD


# Same value as chair.OBJECT_RELEVANT_RANGE_M -- this scenario is otherwise
# identical in shape to chair_object.py, just with a different obstacle.
OBJECT_RELEVANT_RANGE_M = 18.0

# BEV_SEMANTIC_CLASS_INFO's "walker" class index -- resolved once at import
# time, same pattern as T_bone_car.py/interception_car.py's "vehicle" index.
_WALKER_BEV_CLASS_INDEX = next(
    index for index, info in BEV_SEMANTIC_CLASS_INFO.items()
    if info["class_name"] == "walker"
)


# -------------------------
# Scenario-local error_detection resolvers
# -------------------------

def resolve_expected(distance_to_object_m):
    """
    Same shape as chair.resolve_expected(): "stop" once the pedestrian is
    within OBJECT_RELEVANT_RANGE_M, "go" otherwise. expected_perception is
    "walker" (matching pred_bev_semantic's class name, an in-distribution
    class) rather than chair.py's generic "obstacle_detected", matching
    T_bone_car.resolve_expected()'s convention for a real class.
    """
    if distance_to_object_m <= OBJECT_RELEVANT_RANGE_M:
        return {"expected_perception": "walker", "expected_action": "stop"}

    return {"expected_perception": "none", "expected_action": "go"}


def resolve_actual_perception(pred_bev_semantic, vehicle, object_actor, tfpp_config):
    """
    "walker" if the model's own BEV output crosses MIN_SEMANTIC_PIXEL_THRESHOLD
    pixels of its "walker" class anywhere in the grid, "none" otherwise --
    identical in kind to T_bone_car.resolve_actual_perception() (a real
    trained class, so no spatial footprint-at-a-location projection like
    chair.resolve_actual_perception() needs). vehicle/object_actor/
    tfpp_config are accepted only to match monitor_with_model_and_act_for_object's
    resolve_actual_perception(...) call signature -- vehicle/tfpp_config are
    unused here.
    """
    walker_pixel_count = int((pred_bev_semantic == _WALKER_BEV_CLASS_INDEX).sum().item())

    if walker_pixel_count >= MIN_SEMANTIC_PIXEL_THRESHOLD:
        return "walker"

    return "none"


# -------------------------
# Object blueprint / spawn helpers
# -------------------------

def find_object_blueprint(bp_lib):
    """
    Picks the first walker.pedestrian.* blueprint in this CARLA version's
    library -- same "walker.pedestrian.*" filter config/CARLA_actors/
    pedestrian.py's spawn_pedestrian_relative_to_vehicle() uses, disabling
    is_invincible the same way (so this pedestrian, like that one, can
    actually be hit/react like a normal walker actor rather than the
    invincible default).
    """
    walker_blueprints = bp_lib.filter("walker.pedestrian.*")

    if not walker_blueprints:
        raise RuntimeError("No pedestrian blueprints found.")

    pedestrian_bp = walker_blueprints[0]

    if pedestrian_bp.has_attribute("is_invincible"):
        pedestrian_bp.set_attribute("is_invincible", "false")

    return pedestrian_bp


def spawn_pedestrian_ahead(
    world,
    bp_lib,
    ego_waypoint,
    distance_ahead_m,
    blueprint_selector,
):
    """
    Shared idle-pedestrian placement used by the normal and red-shirt actor
    modules. blueprint_selector owns appearance selection; every geometry,
    ground-height, and stationary-control step remains identical.

    Unlike chair.py's static prop, a spawned walker actor has its own
    locomotion and would otherwise be free to wander -- hold_pedestrian_still()
    (config/CARLA_actors/pedestrian.py) applies a zero-speed WalkerControl
    so it actually stays put, same as that module's own pedestrian before
    start_pedestrian_crossing() is called.
    """
    forward_waypoints = ego_waypoint.next(distance_ahead_m)

    if not forward_waypoints:
        raise RuntimeError(
            f"No waypoint found {distance_ahead_m}m ahead of the ego spawn point."
        )

    object_waypoint = forward_waypoints[0]
    ground_location = object_waypoint.transform.location

    object_bp = blueprint_selector(bp_lib)
    logging.debug(f"Object blueprint: {object_bp.id}")

    spawn_transform = carla.Transform(
        carla.Location(x=ground_location.x, y=ground_location.y, z=ground_location.z + 0.5),
        object_waypoint.transform.rotation
    )

    object_actor = world.spawn_actor(object_bp, spawn_transform)

    bounding_box = object_actor.bounding_box
    corrected_z = ground_location.z - bounding_box.location.z + bounding_box.extent.z
    corrected_location = carla.Location(x=ground_location.x, y=ground_location.y, z=corrected_z)

    object_actor.set_location(corrected_location)

    hold_pedestrian_still(object_actor)

    # Logs corrected_location directly rather than reading object_actor.get_location()
    # back -- same reasoning as chair.spawn_object_ahead()'s identical note.
    logging.debug(
        f"Object spawned {distance_ahead_m:.1f}m ahead of ego at "
        f"{corrected_location} (road_id={object_waypoint.road_id}, "
        f"lane_id={object_waypoint.lane_id})."
    )

    return object_actor


def spawn_object_ahead(world, bp_lib, ego_waypoint, distance_ahead_m):
    """Spawn the original unmodified pedestrian variant ahead of ego."""
    return spawn_pedestrian_ahead(
        world,
        bp_lib,
        ego_waypoint,
        distance_ahead_m,
        blueprint_selector=find_object_blueprint,
    )
