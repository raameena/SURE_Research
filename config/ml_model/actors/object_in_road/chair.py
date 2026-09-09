"""
chair: the out-of-distribution object logic for scenarios/model/object_in_road/
chair_object.py -- blueprint selection/spawning, and the scenario-local
error_detection resolvers (same pattern as scenarios/model/intersection/
simple_stop_go.py's LIGHT_TO_EXPECTED, but functions instead of a fixed
dict, since the relevant ground truth here -- distance to the object -- is
continuous, and turning the model's raw BEV output into a perception string
requires spatial logic with no equivalent to a class-name match).

See chair_object.py's module docstring for the full "why a generic
occupancy proxy instead of a class-name match" and "why expected_action is
stop-only for v1" reasoning -- kept there since it's about the scenario's
design, not this file's spawning/resolver mechanics.
"""

import logging

import carla

from controls.ml_controls import project_to_bev_pixel, BEV_SEMANTIC_CLASS_INFO


# ~15-20m out is the "do something about it" window for v1 -- picked, not
# derived from a braking-distance calculation (that's what timely_action
# checks separately, after the fact).
OBJECT_RELEVANT_RANGE_M = 18.0

# Candidates NOT in typical driving-scene training classes (vehicle/
# pedestrian/cyclist/cone) -- the point is testing perception on an
# out-of-distribution obstacle, not a known class. Checked against the live
# blueprint library at runtime (see find_object_blueprint()) rather than
# hardcoding one ID, since asset availability can differ across CARLA
# versions/content packs. Verified present in this project's CARLA 0.9.16
# static.prop.* library before writing this list.
OBJECT_BLUEPRINT_CANDIDATES = [
    "static.prop.plasticchair",
    "static.prop.plastictable",
    "static.prop.gnome",
    "static.prop.shoppingcart",
    "static.prop.creasedbox01",
]

# BEV_SEMANTIC_CLASS_INFO classes that represent an actually-drivable
# surface -- see chair_object.py's module docstring for why "obstacle
# perceived" is defined this way instead of a class-name match.
DRIVABLE_BEV_CLASS_NAMES = {"road", "lane_markers", "lane_markers_broken"}
DRIVABLE_BEV_CLASS_INDICES = {
    index for index, info in BEV_SEMANTIC_CLASS_INFO.items()
    if info["class_name"] in DRIVABLE_BEV_CLASS_NAMES
}


# -------------------------
# Scenario-local error_detection resolvers
# -------------------------

def resolve_expected(distance_to_object_m):
    """
    v1 scope cut: expected_action is "stop" once the object is within
    OBJECT_RELEVANT_RANGE_M, "go" otherwise -- there is no "go around"
    option to check against (see chair_object.py's module docstring).
    """
    if distance_to_object_m <= OBJECT_RELEVANT_RANGE_M:
        return {"expected_perception": "obstacle_detected", "expected_action": "stop"}

    return {"expected_perception": "none", "expected_action": "go"}


def resolve_actual_perception(pred_bev_semantic, vehicle, object_actor, tfpp_config):
    """
    "obstacle_detected" if any pixel in the object's projected BEV footprint
    (see controls.ml_controls.project_to_bev_pixel()) isn't one of
    DRIVABLE_BEV_CLASS_INDICES -- "none" if the object is currently outside
    the BEV grid's covered area, or every pixel in its footprint reads as
    drivable surface (the model saw nothing unusual there). The footprint
    (not a single pixel) is sampled using the object's own bounding-box
    extent, converted to a pixel radius, so a pixel or two of projection
    error doesn't make this all-or-nothing.
    """
    pixel = project_to_bev_pixel(vehicle, object_actor.get_location(), tfpp_config)

    if pixel is None:
        return "none"

    row, col = pixel

    extent = object_actor.bounding_box.extent
    half_height_px = max(1, round(extent.x * tfpp_config.pixels_per_meter))
    half_width_px = max(1, round(extent.y * tfpp_config.pixels_per_meter))

    row_start = max(0, row - half_height_px)
    row_end = min(tfpp_config.lidar_resolution_height, row + half_height_px + 1)
    col_start = max(0, col - half_width_px)
    col_end = min(tfpp_config.lidar_resolution_width, col + half_width_px + 1)

    footprint = pred_bev_semantic[row_start:row_end, col_start:col_end]

    for class_index in footprint.unique().tolist():
        if int(class_index) not in DRIVABLE_BEV_CLASS_INDICES:
            return "obstacle_detected"

    return "none"


# -------------------------
# Object blueprint / spawn helpers
# -------------------------

def find_object_blueprint(bp_lib):
    """
    Picks the first OBJECT_BLUEPRINT_CANDIDATES entry that actually exists
    in this CARLA version's static.prop.* library, rather than assuming a
    hardcoded ID is present.
    """
    for candidate_id in OBJECT_BLUEPRINT_CANDIDATES:
        matches = bp_lib.filter(candidate_id)

        if len(matches) > 0:
            return matches[0]

    raise RuntimeError(
        f"None of the candidate out-of-distribution object blueprints "
        f"{OBJECT_BLUEPRINT_CANDIDATES} exist in this CARLA version's "
        f"static.prop.* library."
    )


def spawn_object_ahead(world, bp_lib, ego_waypoint, distance_ahead_m):
    """
    Spawns the out-of-distribution object distance_ahead_m along the same
    lane ego spawned in -- walks ego_waypoint.next() rather than offsetting
    by a straight-line vector (matches carla_config.spawn_ego_vehicle_behind_index()'s
    use of waypoint.previous() for the same reason), since a straight
    Euclidean offset can drift off the actual lane centerline wherever the
    road curves, re-snapping project_to_road=True onto a different lane/road
    entirely -- confirmed empirically against this project's map before
    writing this: a raw sp10.location + forward_vector*30 landed on a
    different road_id/lane_id than ego's own spawn lane, while
    ego_waypoint.next(30) correctly stayed on the connected path ahead.

    z is resolved in two steps: spawn 0.5m above the target waypoint's road
    surface first (spawning exactly at surface height risks a same-frame
    collision against the road mesh for some prop meshes), then read the
    spawned actor's own bounding_box back and reposition its origin so the
    box's bottom face sits exactly on the road surface -- a fixed z offset
    would only be correct for one specific blueprint's own pivot placement,
    and these vary per prop.
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

    # Logs corrected_location directly rather than reading object_actor.get_location()
    # back -- immediately after spawn_actor()/set_location(), a freshly
    # spawned actor's transform isn't guaranteed to be reflected yet in a
    # get_location() read before the next world.tick() (confirmed
    # empirically -- see the identical note where this function is called).
    logging.debug(
        f"Object spawned {distance_ahead_m:.1f}m ahead of ego at "
        f"{corrected_location} (road_id={object_waypoint.road_id}, "
        f"lane_id={object_waypoint.lane_id})."
    )

    return object_actor
