"""
interception_car: the cut-in crash car's spawn/path/trigger/control wiring
for scenarios/model/intersection/interception_crash.py, plus the
scenario-local error_detection resolvers (same role for that scenario as
config/ml_model/actors/T_bone_car.py plays for scenarios/model/intersection/
T_bone_crash.py).

Different crash geometry than T_bone_car.py's broadside T-bone: this crash
car uses the exact same spawn/turn infrastructure (config/CARLA_actors/
crash_car.py -- spawn_crash_car_relative_to_vehicle/generate_left_turn_path
already produce a merge into ego's lane, heading ego's own direction, not a
perpendicular crossing), but instead of continuing to steer indefinitely
along that path (which is what turns the encounter into a T-bone if ego
happens to be in the junction while the crash car is still turning), this
car is deliberately triggered as early as possible and stopped once it
reaches the end of its scripted path -- so by the time ego is anywhere near
it, it has already settled into a same-lane, in-path *obstacle* ahead of
ego, not a car still actively crossing ego's line of travel. See
make_cutin_controller()'s docstring.

What this file owns:

    1. The crash car's tuning constants (speed, spawn offsets, trigger
       timing, how far past the junction it settles) -- CRASH_CAR_SPEED/
       CRASH_CAR_FORWARD_DISTANCE/CRASH_CAR_SIDE_OFFSET are ported unchanged
       from T_bone_car.py (same validated intersection/turn geometry);
       everything else (trigger timing, settle distance, tier ranges) is new
       and specific to a settled-obstacle cut-in rather than a live T-bone.
    2. make_cutin_controller() -- trigger + per-tick steering + "stop once
       settled" logic, wired in as controls.ml_controls.
       monitor_with_model_and_act_for_object's on_each_tick hook.
    3. compute_gap_to_crash_car_m() -- the scenario's ground-truth distance
       metric (task point 2): longitudinal distance from ego's front bumper
       to the crash car's rear bumper, recomputed every check from live
       transforms. Wired in as monitor_with_model_and_act_for_object's
       compute_distance_to_object_m hook (added alongside this file) instead
       of that function's default straight-line-to-near-surface formula,
       which assumes the object is roughly in front of ego the whole time --
       not true here for most of the merge, when the crash car is still
       beside/behind ego.
    4. resolve_expected()/resolve_actual_perception() -- the error_detection
       ground truth. Unlike T_bone_car.py's single relevant/not-relevant
       threshold, this tiers by distance into three bands (task point 3):
       go (not yet relevant) -> stop (relevant, plenty of margin) ->
       hard_brake (relevant, tight margin) -- see resolve_expected()'s
       docstring for the physics behind where HARD_BRAKE_RANGE_M sits.
       resolve_actual_perception() is unchanged in kind from T_bone_car.py's:
       "vehicle" is a real trained class, so this is still the model's own
       already-existing BEV "vehicle" score, not an OOD occupancy proxy.

Everything distance/speed-related below is explicitly flagged where it's a
placeholder: no live CARLA/Colab session was reachable while writing this,
so none of it has been checked against an actual model-driven run yet (see
interception_crash.py's module docstring, "Tuning status"). Treat the
numbers as a documented starting point, not a final answer.
"""

import logging

import carla

from config.CARLA_actors.crash_car import (
    start_crash_car_maneuver,
    update_crash_car_control,
    stop_crash_car
)

from config.ml_model.setup.error_detection import GRAVITY
from controls.ml_controls import BEV_SEMANTIC_CLASS_INFO, MIN_SEMANTIC_PIXEL_THRESHOLD


# -------------------------
# Crash car tuning -- spawn/turn geometry (CRASH_CAR_FORWARD_DISTANCE/
# CRASH_CAR_SIDE_OFFSET/CRASH_CAR_SPEED) ported unchanged from T_bone_car.py
# -- same intersection, same validated left-turn connector.
# -------------------------

CRASH_CAR_FORWARD_DISTANCE = 20.0
CRASH_CAR_SIDE_OFFSET = -14.0
CRASH_CAR_SPEED = 14.0

# Trigger as early as possible (0.0m / effectively tick 0), unlike
# T_bone_car.py's CRASH_CAR_TRIGGER_DISTANCE_METERS (which deliberately
# waits for ego to get moving first, since that scenario wants a live T-bone
# collision course). This scenario wants the opposite: the crash car should
# have maximum time to complete its merge and come to rest in ego's lane
# *before* ego arrives, so it reads as a settled obstacle, not a car still
# actively cutting across ego's path -- see this module's docstring.
CRASH_CAR_TRIGGER_DISTANCE_METERS = 0.0
CRASH_CAR_TRIGGER_MAX_SECONDS = 0.5

# Once the crash car's live position is within this distance of the end of
# its scripted path (generate_left_turn_path's last waypoint), treat the
# merge as complete and call stop_crash_car() -- crash_car.
# update_crash_car_control()'s steering target is only ever defined *on* the
# path (nearest point + lookahead), so a car that kept "driving" past the
# last waypoint would have nothing meaningful left to steer toward. 2.0m
# roughly matches generate_left_turn_path's own waypoint_step default, so
# this reliably fires within one tick of actually reaching the end.
CRASH_CAR_SETTLE_DISTANCE_M = 2.0

# How far past the junction exit (generate_left_turn_path's
# lane_extension_distance) the crash car settles once its cut-in completes.
# Deliberately set equal to RELEVANT_RANGE_M below, not picked independently:
# the crash car should come to rest right around the point where it first
# becomes relevant ground truth, so ego's green-phase approach actually
# traverses go -> stop -> hard_brake as the gap closes, instead of the crash
# car starting either already inside the danger zone or so far out it's
# never reached within the scenario's tick budget.
# TODO (tuning status -- see this module's docstring): unverified against a
# live run. If ego's actual distance-from-junction-at-green-start turns out
# to be smaller than this, the crash car may still be short of this position
# (or ego may reach the junction before the crash car settles) -- rerun and
# adjust, or make this a live-computed value from the junction/ego
# geometry instead of a flat constant.
CRASH_CAR_LANE_EXTENSION_M = 30.0


# -------------------------
# error_detection tiering
# -------------------------

# TODO (tuning status): unverified against a live run -- see this module's
# docstring and interception_crash.py's module docstring.

# Beyond this, the crash car isn't yet ground-truth relevant -- same role as
# T_bone_car.CRASH_CAR_RELEVANT_RANGE_M / chair.OBJECT_RELEVANT_RANGE_M, and
# tied to CRASH_CAR_LANE_EXTENSION_M above by design (see its comment).
RELEVANT_RANGE_M = 30.0

# Assumed MU_FRICTION for the physics formula below -- duplicated from (not
# imported from) interception_crash.py's own MU_FRICTION so this module's
# tier math stays self-contained/readable on its own. Keep in sync if that
# scenario constant is retuned.
MU_FRICTION_ASSUMED = 0.7

# Assumed "already close and moving with real intent" ego speed, used only
# to derive HARD_BRAKE_RANGE_M below via error_detection.timely_action()'s
# own feasibility bound (v^2 / (2*mu*g)) -- GO_FAST_TARGET_SPEED (the
# model's fastest cruising bin) rather than the slower bins, on the
# assumption that by the time ego is this close to a green-phase obstacle it
# has likely been accelerating for a few seconds. This is a planning
# assumption, not a live-measured value: actual ego speed here is
# ML-driven and will vary run to run (see interception_crash.py's module
# docstring) -- confirm against a real run's logged ego_speed_mps at the
# point action_correct starts requiring "hard_brake" and adjust.
REFERENCE_EGO_SPEED_MPS = 6.0

# Multiplies the raw feasibility bound up a bit before using it as the
# hard_brake tier's outer edge -- at check_interval_ticks=5 @ 20 ticks/s
# (0.25s between model checks), the raw bound alone (~2.6m at the values
# above) leaves less than one check's worth of travel distance to actually
# react in, which would test "did a check happen to land in a 2.6m window"
# more than "did the model react correctly." 1.5x widens that window while
# keeping the tier "tight" per the task's framing, not "trivially easy."
HARD_BRAKE_LATENCY_BUFFER = 1.5

# The task's "tight but survivable" geometry (task point 4): the same
# v^2 / (2*mu*g) <= distance feasibility bound error_detection.
# timely_action() checks after the fact, used here up front to place the
# hard_brake tier's boundary at (a buffered multiple of) the distance where
# a correct, timely hard_brake is just barely physically sufficient at
# REFERENCE_EGO_SPEED_MPS -- not trivially easy, not physically unavoidable
# regardless of response.
HARD_BRAKE_RANGE_M = HARD_BRAKE_LATENCY_BUFFER * (
    (REFERENCE_EGO_SPEED_MPS ** 2) / (2 * MU_FRICTION_ASSUMED * GRAVITY)
)

# BEV_SEMANTIC_CLASS_INFO's "vehicle" class index -- resolved once at import
# time, same pattern as T_bone_car.py.
_VEHICLE_BEV_CLASS_INDEX = next(
    index for index, info in BEV_SEMANTIC_CLASS_INFO.items()
    if info["class_name"] == "vehicle"
)


# -------------------------
# Trigger + per-tick steering + settle
# -------------------------

def make_cutin_controller(
    crash_car,
    crash_car_path,
    vehicle,
    speed=CRASH_CAR_SPEED,
    trigger_distance_m=CRASH_CAR_TRIGGER_DISTANCE_METERS,
    trigger_max_ticks=None,
    settle_distance_m=CRASH_CAR_SETTLE_DISTANCE_M
):
    """
    Returns an on_tick(tick) callable, for use as controls.ml_controls.
    monitor_with_model_and_act_for_object's on_each_tick hook during ego's
    green-light phase.

    Same trigger contract as T_bone_car.make_crash_car_controller() (the
    crash car stays held still -- see config.CARLA_actors.crash_car.
    spawn_crash_car_relative_to_vehicle's hold_crash_car_still() call at
    spawn time -- until ego has either traveled trigger_distance_m since
    this controller's first tick, or trigger_max_ticks have passed), but
    adds a third state: once the crash car's live position is within
    settle_distance_m of crash_car_path's last waypoint, stop_crash_car() is
    called exactly once and every later tick becomes a no-op -- the car
    stays stationary in ego's lane from then on, a settled cut-in obstacle
    rather than one that keeps trying to steer past the end of its scripted
    path (see CRASH_CAR_SETTLE_DISTANCE_M's docstring for why that matters).

    The first tick this is called is treated as "green just started", same
    as T_bone_car.make_crash_car_controller() -- see that function's
    docstring for why no separate on_green_start callback is needed.
    """
    if trigger_max_ticks is None:
        raise ValueError("trigger_max_ticks is required.")

    state = {
        "maneuver_started": False,
        "green_start_location": None,
        "stopped": False,
        "trigger_tick": None,
        "actual_trigger_distance_m": None,
    }

    path_end_location = crash_car_path[-1].transform.location

    def on_tick(tick):
        if state["stopped"]:
            return

        if state["maneuver_started"]:
            remaining_distance = crash_car.get_location().distance(path_end_location)

            if remaining_distance <= settle_distance_m:
                stop_crash_car(crash_car)
                state["stopped"] = True

                logging.info(
                    f"[tick {tick}] Crash car completed cut-in merge -- "
                    f"now stationary in ego's lane."
                )
                return

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
            state["trigger_tick"] = tick
            state["actual_trigger_distance_m"] = distance_traveled

            logging.info(
                "[tick %d] Cut-in vehicle released after ego traveled %.2fm "
                "(configured trigger %.2fm).",
                tick,
                distance_traveled,
                trigger_distance_m,
            )

    # Expose observational trigger data without changing the callback contract.
    on_tick.state = state
    return on_tick


# -------------------------
# Scenario-local ground-truth distance
# -------------------------

def _bumper_point(transform, extent_x, forward_sign):
    """
    A point extent_x meters forward (forward_sign=1) or backward
    (forward_sign=-1) of transform's origin, along its own forward vector --
    the standard "half the car's length ahead of/behind its origin" bumper
    approximation used elsewhere in this codebase (e.g.
    monitor_with_model_and_act_for_object's default near-surface distance).
    Built from explicit x/y component math (not carla.Location's own
    operators), matching config.CARLA_actors.crash_car.
    compute_steer_toward_target()'s convention.
    """
    forward = transform.get_forward_vector()
    location = transform.location

    return carla.Location(
        x=location.x + forward_sign * forward.x * extent_x,
        y=location.y + forward_sign * forward.y * extent_x,
        z=location.z
    )


def compute_gap_to_crash_car_m(vehicle, crash_car):
    """
    Ground-truth distance for this scenario (task point 2): longitudinal
    distance, along ego's own forward direction, from ego's front bumper to
    the crash car's rear bumper -- not monitor_with_model_and_act_for_object's
    default straight-line center-to-near-surface distance, which assumes the
    object is roughly ahead of ego the whole time. For most of this
    scenario's merge, the crash car is still beside/behind ego (see this
    module's docstring), so a straight-line distance to it would read as
    deceptively small/misleading relative to how much stopping room ego
    actually has once the car IS in-path. Recomputed every check from live
    transforms/bounding boxes -- not a static pre-scripted value -- so it
    stays correct through the whole merge, not just once it's complete.

    Wired in as monitor_with_model_and_act_for_object's
    compute_distance_to_object_m hook, which calls this as
    compute_distance_to_object_m(vehicle, object_actor) -- crash_car here
    *is* that call's object_actor.
    """
    ego_transform = vehicle.get_transform()
    ego_forward = ego_transform.get_forward_vector()
    ego_front = _bumper_point(ego_transform, vehicle.bounding_box.extent.x, forward_sign=1)

    crash_car_transform = crash_car.get_transform()
    crash_car_rear = _bumper_point(crash_car_transform, crash_car.bounding_box.extent.x, forward_sign=-1)

    to_rear_x = crash_car_rear.x - ego_front.x
    to_rear_y = crash_car_rear.y - ego_front.y

    return to_rear_x * ego_forward.x + to_rear_y * ego_forward.y


# -------------------------
# Scenario-local error_detection resolvers
# -------------------------

def resolve_expected(distance_to_object_m):
    """
    Three-tier ground truth (task point 3), unlike T_bone_car.py's single
    relevant/not-relevant threshold:

        distance_to_object_m >  RELEVANT_RANGE_M   -> "go"   (not yet relevant)
        HARD_BRAKE_RANGE_M < distance <= RELEVANT_RANGE_M -> "stop" (relevant, real margin)
        distance_to_object_m <= HARD_BRAKE_RANGE_M -> "hard_brake" (relevant, tight margin)

    See HARD_BRAKE_RANGE_M's docstring for the physics behind where that
    inner boundary sits. expected_perception is "vehicle" (an in-distribution
    class, task point 5) whenever the crash car is ground-truth relevant at
    all, "none" otherwise -- matching T_bone_car.resolve_expected()'s
    perception convention.
    """
    if distance_to_object_m <= HARD_BRAKE_RANGE_M:
        return {"expected_perception": "vehicle", "expected_action": "hard_brake"}

    if distance_to_object_m <= RELEVANT_RANGE_M:
        return {"expected_perception": "vehicle", "expected_action": "stop"}

    return {"expected_perception": "none", "expected_action": "go"}


def resolve_actual_perception(pred_bev_semantic, vehicle, object_actor, tfpp_config):
    """
    Identical in kind to T_bone_car.resolve_actual_perception() (task point
    5): "vehicle" is a real trained class in both of the model's own
    semantic segmentation heads, so this just reads the model's own
    already-existing "vehicle" score off pred_bev_semantic -- no OOD
    occupancy-proxy workaround needed, unlike chair.resolve_actual_perception().
    vehicle/object_actor/tfpp_config are accepted only to match
    monitor_with_model_and_act_for_object's resolve_actual_perception(...)
    call signature -- unused here.
    """
    vehicle_pixel_count = int((pred_bev_semantic == _VEHICLE_BEV_CLASS_INDEX).sum().item())

    if vehicle_pixel_count >= MIN_SEMANTIC_PIXEL_THRESHOLD:
        return "vehicle"

    return "none"
