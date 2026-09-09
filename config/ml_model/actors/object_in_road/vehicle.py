"""
vehicle: the idle-in-road obstacle logic for scenarios/model/object_in_road/
vehicle_object.py -- blueprint selection/spawning, and the scenario-local
error_detection resolvers (same role as config/ml_model/actors/object_in_road/chair.py
plays for scenarios/model/object_in_road/chair_object.py, and same overall
scenario shape: a single static obstacle spawned ahead of ego in its own
lane, straight-road approach, stop-only expected action).

Unlike chair.py's out-of-distribution prop, "vehicle" is a real class in
both of the model's own semantic segmentation heads (see controls.
ml_controls.SEMANTIC_CLASS_INFO/BEV_SEMANTIC_CLASS_INFO) -- so, following
config/ml_model/actors/T_bone_car.py's precedent for this same distinction,
resolve_actual_perception() here just reads the model's own already-existing
"vehicle" score off pred_bev_semantic instead of chair.py's generic
occupancy-footprint proxy. See chair_object.py's module
docstring for the fuller "why a generic proxy only when there's no real
class" reasoning this mirrors.
"""

import logging

import carla

from config.CARLA_actors.crash_car import hold_crash_car_still

from controls.ml_controls import BEV_SEMANTIC_CLASS_INFO, MIN_SEMANTIC_PIXEL_THRESHOLD


# Same value as chair.OBJECT_RELEVANT_RANGE_M -- this scenario is otherwise
# identical in shape to chair_object.py, just with a different obstacle.
OBJECT_RELEVANT_RANGE_M = 18.0

# Concrete vehicle blueprints, not "vehicle.*" -- avoids picking whatever
# happens to sort first in the live library, and avoids matching
# spawn_ego_vehicle()'s own default ("vehicle.tesla.model3"), so the idle
# obstacle doesn't look identical to ego in recordings. Same list as
# config/CARLA_actors/crash_car.py's CRASH_CAR_BLUEPRINT_IDS -- already
# verified present in this project's CARLA version, so reused here rather
# than guessing new IDs. find_object_blueprint() below checks each against
# the live blueprint library at runtime, same pattern as chair.
# find_object_blueprint(), rather than assuming the first entry is present.
VEHICLE_BLUEPRINT_CANDIDATES = [
    "vehicle.dodge.charger_2020",
    "vehicle.mercedes.coupe_2020",
    "vehicle.audi.tt",
    "vehicle.mini.cooper_s_2021",
    "vehicle.chevrolet.impala",
]

# BEV_SEMANTIC_CLASS_INFO's "vehicle" class index -- resolved once at import
# time, same pattern as T_bone_car.py/interception_car.py.
_VEHICLE_BEV_CLASS_INDEX = next(
    index for index, info in BEV_SEMANTIC_CLASS_INFO.items()
    if info["class_name"] == "vehicle"
)


# -------------------------
# Scenario-local error_detection resolvers
# -------------------------

def resolve_expected(distance_to_object_m):
    """
    Same shape as chair.resolve_expected(): "stop" once the vehicle is
    within OBJECT_RELEVANT_RANGE_M, "go" otherwise. expected_perception is
    "vehicle" (an in-distribution class) rather than chair.py's generic
    "obstacle_detected", matching T_bone_car.resolve_expected()'s
    convention for a real class.
    """
    if distance_to_object_m <= OBJECT_RELEVANT_RANGE_M:
        return {"expected_perception": "vehicle", "expected_action": "stop"}

    return {"expected_perception": "none", "expected_action": "go"}


def resolve_actual_perception(pred_bev_semantic, vehicle, object_actor, tfpp_config):
    """
    "vehicle" if the model's own BEV output crosses MIN_SEMANTIC_PIXEL_THRESHOLD
    pixels of its "vehicle" class anywhere in the grid, "none" otherwise --
    identical in kind to T_bone_car.resolve_actual_perception() (a real
    trained class, so no spatial footprint-at-a-location projection like
    chair.resolve_actual_perception() needs). vehicle/object_actor/
    tfpp_config are accepted only to match monitor_with_model_and_act_for_object's
    resolve_actual_perception(...) call signature -- vehicle/tfpp_config are
    unused here.
    """
    vehicle_pixel_count = int((pred_bev_semantic == _VEHICLE_BEV_CLASS_INDEX).sum().item())

    if vehicle_pixel_count >= MIN_SEMANTIC_PIXEL_THRESHOLD:
        return "vehicle"

    return "none"


# -------------------------
# Object blueprint / spawn helpers
# -------------------------

def find_object_blueprint(bp_lib):
    """
    Picks the first VEHICLE_BLUEPRINT_CANDIDATES entry that actually exists
    in this CARLA version's vehicle.* library -- same pattern as
    chair.find_object_blueprint().
    """
    for candidate_id in VEHICLE_BLUEPRINT_CANDIDATES:
        matches = bp_lib.filter(candidate_id)

        if len(matches) > 0:
            return matches[0]

    raise RuntimeError(
        f"None of the candidate vehicle blueprints "
        f"{VEHICLE_BLUEPRINT_CANDIDATES} exist in this CARLA version's "
        f"vehicle.* library."
    )


def spawn_object_ahead(world, bp_lib, ego_waypoint, distance_ahead_m):
    """
    Spawns an idle vehicle distance_ahead_m along the same lane ego spawned
    in -- identical placement logic to chair.spawn_object_ahead() (see that
    function's docstring for why ego_waypoint.next() is used instead of a
    straight-line offset, and why z is resolved via the spawned actor's own
    bounding box rather than a fixed offset).

    Unlike chair.py's static prop, a spawned vehicle actor has physics/a
    drivetrain and would otherwise start rolling under gravity/idle
    throttle -- hold_crash_car_still() (config/CARLA_actors/crash_car.py)
    zeroes its velocity and applies a handbrake so it actually stays put,
    same as that module's own crash car before its maneuver is triggered.
    """
    forward_waypoints = ego_waypoint.next(distance_ahead_m)

    if not forward_waypoints:
        raise RuntimeError(
            f"No waypoint found {distance_ahead_m}m ahead of the ego spawn point."
        )

    object_waypoint = forward_waypoints[0]
    ground_location = object_waypoint.transform.location

    object_bp = find_object_blueprint(bp_lib)
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

    hold_crash_car_still(object_actor)

    # Logs corrected_location directly rather than reading object_actor.get_location()
    # back -- same reasoning as chair.spawn_object_ahead()'s identical note.
    logging.debug(
        f"Object spawned {distance_ahead_m:.1f}m ahead of ego at "
        f"{corrected_location} (road_id={object_waypoint.road_id}, "
        f"lane_id={object_waypoint.lane_id})."
    )

    return object_actor
