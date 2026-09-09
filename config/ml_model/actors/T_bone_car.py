"""
T_bone_car: the T-bone crash car's spawn/path/trigger/control wiring for
scenarios/model/intersection/T_bone_crash.py, plus the scenario-local
error_detection resolvers (same role for that scenario as config/ml_model/
actors/object_in_road/chair.py plays for scenarios/model/object_in_road/
chair_object.py).

Unlike chair.py, the crash car's spawn/path/steering geometry already has a
full, working implementation in config/CARLA_actors/crash_car.py (built for
the hard-controls scenarios/hard_controls/intersection/car_crash_intersection.py)
-- that logic is reused here via import, not duplicated, so there is one
source of truth for the crash car's turn geometry. What this file actually
owns:

    1. The crash car's tuning constants (speed, spawn offsets, trigger
       timing) -- ported unchanged from car_crash_intersection.py's own
       module-level constants.
    2. make_crash_car_controller() -- the trigger/per-tick-steering logic
       that used to live inline as closures in car_crash_intersection.py's
       main() (record_green_start()/tick_crash_car()), relocated here as a
       reusable factory so scenarios/model/intersection/T_bone_crash.py can
       wire it in as controls.ml_controls.monitor_with_model_and_act_for_object's
       on_each_tick hook.
    3. resolve_expected()/resolve_actual_perception() -- the error_detection
       ground truth for this scenario. Unlike chair.py, this is an
       in-distribution perception check: "vehicle" is a real class in both
       of the model's own semantic segmentation heads (see controls.
       ml_controls.SEMANTIC_CLASS_INFO/BEV_SEMANTIC_CLASS_INFO), so there is
       no OOD occupancy-proxy workaround needed here -- resolve_actual_perception
       just reads the model's own already-existing "vehicle" class score off
       pred_bev_semantic, the same score interpret_semantic_output() already
       derives its "Vehicle detected" message from.
"""

from config.CARLA_actors.crash_car import (
    start_crash_car_maneuver,
    update_crash_car_control
)

from controls.ml_controls import BEV_SEMANTIC_CLASS_INFO, MIN_SEMANTIC_PIXEL_THRESHOLD


# -------------------------
# Crash car tuning -- ported unchanged from scenarios/hard_controls/
# intersection/car_crash_intersection.py's own module-level constants (see
# that file for the geometry/placeholder-speed reasoning behind these
# values; nothing here has been retuned).
# -------------------------

CRASH_CAR_FORWARD_DISTANCE = 20.0
CRASH_CAR_SIDE_OFFSET = -14.0

# Placeholder speed for the crash car's scripted turn -- see
# car_crash_intersection.py's identical comment. Still untuned; watch a run
# and adjust if the crash car doesn't reliably reach ego's lane in time.
CRASH_CAR_SPEED = 14.0

# Same trigger condition as car_crash_intersection.py: give the ego a head
# start on green before releasing the crash car, so it's already moving at
# speed when the crash car pulls out. Ported unchanged -- see this module's
# docstring and scenarios/model/intersection/T_bone_crash.py's module
# docstring for why this fixed-offset trigger may need to become a
# live-position pursuit calc if the model-driven ego's speed profile (which
# varies run to run, unlike the hard-controls scenario's deterministic one)
# makes it miss.
CRASH_CAR_TRIGGER_DISTANCE_METERS = 15.0 * 0.3048  # 15 feet
CRASH_CAR_TRIGGER_MAX_SECONDS = 2.0

# error_detection relevance gate (chair.py's OBJECT_RELEVANT_RANGE_M
# pattern) -- reuses car_crash_intersection.py's own CRASH_CAR_MAX_CHECK_DISTANCE
# (the straight-line cutoff hazard_controls used before bothering with a
# time-to-collision computation at all) rather than picking a new number,
# since that value was already chosen for this same crash car/intersection.
# v1 assumption per the task: a single relevant/not-relevant threshold, not
# a distance/urgency-tiered expected action -- see resolve_expected()'s
# docstring.
CRASH_CAR_RELEVANT_RANGE_M = 21.0

# BEV_SEMANTIC_CLASS_INFO's "vehicle" class index -- resolved once at import
# time rather than re-searched every resolve_actual_perception() call.
_VEHICLE_BEV_CLASS_INDEX = next(
    index for index, info in BEV_SEMANTIC_CLASS_INFO.items()
    if info["class_name"] == "vehicle"
)


# -------------------------
# Trigger + per-tick steering
# -------------------------

def make_crash_car_controller(
    crash_car,
    crash_car_path,
    vehicle,
    speed=CRASH_CAR_SPEED,
    trigger_distance_m=CRASH_CAR_TRIGGER_DISTANCE_METERS,
    trigger_max_ticks=None
):
    """
    Returns an on_tick(tick) callable, for use as controls.ml_controls.
    monitor_with_model_and_act_for_object's on_each_tick hook during ego's
    green-light phase.

    Ports car_crash_intersection.py's inline record_green_start()/
    tick_crash_car() closures: the crash car stays held still (see
    config.CARLA_actors.crash_car.spawn_crash_car_relative_to_vehicle's
    hold_crash_car_still() call at spawn time) until ego has either
    traveled trigger_distance_m since this controller's first tick, or
    trigger_max_ticks have passed -- whichever comes first -- then departs
    on its scripted left-turn maneuver and is steered along crash_car_path
    every tick after that.

    The first tick this is called is treated as "green just started"
    (T_bone_crash.py only ever wires this in as the green-phase monitor's
    on_each_tick, and that monitor call's own tick 0 is the instant
    force_traffic_light_state() set the light to green -- see
    run_fixed_traffic_light_sequence()), so there's no separate
    on_green_start callback needed the way the hazard-controls version
    required (that one had to watch for green inside a single continuous
    monitor loop spanning all light phases).
    """
    if trigger_max_ticks is None:
        raise ValueError("trigger_max_ticks is required.")

    state = {"maneuver_started": False, "green_start_location": None}

    def on_tick(tick):
        if state["maneuver_started"]:
            update_crash_car_control(
                crash_car,
                crash_car_path,
                tick,
                target_speed=speed
            )
            return

        if state["green_start_location"] is None:
            state["green_start_location"] = vehicle.get_location()

        distance_traveled = vehicle.get_location().distance(
            state["green_start_location"]
        )

        if distance_traveled >= trigger_distance_m or tick >= trigger_max_ticks:
            start_crash_car_maneuver(
                crash_car,
                crash_car_path,
                speed=speed
            )
            state["maneuver_started"] = True

    return on_tick


# -------------------------
# Scenario-local error_detection resolvers
# -------------------------

def resolve_expected(distance_to_crash_car_m):
    """
    v1 scope cut (task-flagged assumption to confirm): expected_action is
    "stop" once the crash car is within CRASH_CAR_RELEVANT_RANGE_M, "go"
    otherwise -- a single relevance threshold, same shape as chair.py's
    resolve_expected(), not a distance/urgency-tiered expected action like
    LIGHT_TO_EXPECTED's red/yellow/green tiers. Worth revisiting given how
    fast this scenario closes (a T-bone, not a slow approach), but not
    built now per the task's scope cut.
    """
    if distance_to_crash_car_m <= CRASH_CAR_RELEVANT_RANGE_M:
        return {"expected_perception": "vehicle", "expected_action": "stop"}

    return {"expected_perception": "none", "expected_action": "go"}


def resolve_actual_perception(pred_bev_semantic, vehicle, object_actor, tfpp_config):
    """
    "vehicle" if the model's own BEV output crosses controls.ml_controls.
    MIN_SEMANTIC_PIXEL_THRESHOLD pixels of its "vehicle" class anywhere in
    the grid, "none" otherwise -- reuses the same class index/threshold
    interpret_semantic_output() already derives its "Vehicle detected"
    message from (see this module's docstring for why no spatial
    footprint-at-a-location projection, unlike chair.resolve_actual_perception,
    is needed here: "vehicle" is a real trained class, not an OOD object).
    vehicle/object_actor/tfpp_config are accepted only to match
    monitor_with_model_and_act_for_object's resolve_actual_perception(pred_bev_semantic,
    vehicle, object_actor, tfpp_config) call signature -- unused here.
    """
    vehicle_pixel_count = int((pred_bev_semantic == _VEHICLE_BEV_CLASS_INDEX).sum().item())

    if vehicle_pixel_count >= MIN_SEMANTIC_PIXEL_THRESHOLD:
        return "vehicle"

    return "none"
