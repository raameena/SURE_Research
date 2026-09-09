import logging
import math
import time

import carla

from controls.actions import stop_vehicle, slow_vehicle, drive_forward, go_fast_vehicle, hard_brake, get_vehicle_speed_mps
from controls.light_controls import log_light_state, get_distance_to_stop_line
from controls.hazard_controls import compute_time_to_collision
from config.CARLA_actors.sensors import follow_vehicle_with_spectator
from config.ml_model.setup.model_loader import (
    get_model_control_prediction,
    get_model_prediction,
)
from config.ml_model.setup import error_detection

# -------------------------
# decide_action() tuning constants
# -------------------------
#
# The legacy path supplies a discrete target-speed bin.  The object_in_road
# fidelity path supplies PCLA's probability-weighted continuous target speed;
# this nearest-bin mapping remains only for research/error labels there.  It
# does not actuate the vehicle: PCLA's control_pid_direct does that from the
# processed speed and predicted checkpoint.
#
# "hard_brake" is therefore not a 5th bin the model can directly predict --
# it's the "stop" bin re-labeled by decide_action() when the situation is
# urgent enough (see HARD_BRAKE_URGENCY_RATIO below) that a plain "stop"
# label would understate how little margin is actually left. Scenarios that
# want to test for it (e.g. scenarios/model/intersection/interception_crash.py)
# pass ego_speed_mps/distance_to_target_m/mu_friction into decide_action()
# so it has the physics inputs to make that call; scenarios that don't
# (the 3-argument call) get the original stop/slow/go/go_fast-only behavior
# unchanged.

STOP_TARGET_SPEED = 0.0
SLOW_TARGET_SPEED = 2.0
GO_TARGET_SPEED = 3.0
GO_FAST_TARGET_SPEED = 6.0

TARGET_SPEED_TO_ACTION = {
    STOP_TARGET_SPEED: "stop",
    SLOW_TARGET_SPEED: "slow",
    GO_TARGET_SPEED: "go",
    GO_FAST_TARGET_SPEED: "go_fast",
}

# Fraction of the physically available stopping distance (error_detection.
# timely_action()'s v^2 / (2*mu*g) bound) that the "stop" decision's own
# required stopping distance must already consume before decide_action()
# relabels it "hard_brake" instead of "stop" -- e.g. 0.6 means "stop" is
# only just barely going to make it (60%+ of the physical limit already
# used up), not that stopping is impossible (that's a separate, later
# question -- see error_detection.timely_action()). Picked, not derived;
# tunable, and worth revisiting once a real scenario run's speed/distance
# profile is available to check it against.
HARD_BRAKE_URGENCY_RATIO = 0.6


def _round_probabilities(probabilities):
    """
    Rounds a {target_speed: probability} dict to 2dp for log readability.
    Logging/display only -- action_log entries keep full precision.
    """
    return {speed: round(probability, 2) for speed, probability in probabilities.items()}


def _format_optional(value):
    """
    Formats a value that may be None -- error_detection.evaluate_tick()'s
    perceieved_right/action_correct are always real booleans, but
    timely_action is None whenever action_correct is False (see
    error_detection.timely_action()'s docstring) or when the Colab endpoint
    was unreachable this check -- as "N/A" for log readability instead of
    "None".
    """
    return "N/A" if value is None else value

# Nearest-bin matching supports both the legacy discrete output and PCLA's
# probability-weighted continuous speed used for object_in_road labels.

ACTION_DISPATCH = {
    "stop": stop_vehicle,
    "slow": slow_vehicle,
    "go": drive_forward,
    "go_fast": go_fast_vehicle,
    "hard_brake": hard_brake,
}

# -------------------------
# Semantic perception interpretation
# -------------------------
#
# Class index -> {class_name, message} for the model's own two semantic
# segmentation heads, straight from the checkpoint's real class definitions
# in PCLA/pcla_agents/transfuserv4/config.py (self.classes/self.converter for
# pred_semantic, self.bev_converter for pred_bev_semantic). The two tensors
# do NOT share an index space -- e.g. index 1 means "vehicle" in
# pred_semantic but "road" in pred_bev_semantic -- so they get separate
# dicts rather than one shared one.
#
# message=None means the class is real and correctly named, but not
# considered notable enough to report every check -- road/sidewalk/lane
# markers/unlabeled are expected in nearly every frame, so surfacing them
# every check would just be noise.

SEMANTIC_CLASS_INFO = {
    0: {"class_name": "unlabeled", "message": None},
    1: {"class_name": "vehicle", "message": "Vehicle detected"},
    2: {"class_name": "road", "message": None},
    3: {"class_name": "traffic_light", "message": "Traffic light detected"},
    4: {"class_name": "pedestrian", "message": "Pedestrian detected"},
    5: {"class_name": "road_line", "message": None},
    6: {"class_name": "sidewalk", "message": None},
}

# pred_bev_semantic distinguishes traffic light color directly (unlike
# pred_semantic's single generic "light" class), so its messages are
# color-specific -- directly relevant to whether the model is actually
# perceiving a red light.
BEV_SEMANTIC_CLASS_INFO = {
    0: {"class_name": "unlabeled", "message": None},
    1: {"class_name": "road", "message": None},
    2: {"class_name": "sidewalk", "message": None},
    3: {"class_name": "lane_markers", "message": None},
    4: {"class_name": "lane_markers_broken", "message": None},
    5: {"class_name": "stop_sign", "message": "Stop sign detected"},
    6: {"class_name": "traffic_light_green", "message": "Traffic light (green) detected"},
    7: {"class_name": "traffic_light_yellow", "message": "Traffic light (yellow) detected"},
    8: {"class_name": "traffic_light_red", "message": "Traffic light (red) detected"},
    9: {"class_name": "vehicle", "message": "Vehicle detected"},
    10: {"class_name": "walker", "message": "Pedestrian detected"},
}

# Flat pixel count, not a fraction of the image -- pred_semantic's map
# (256x1024 = 262144px) and pred_bev_semantic's (256x256 = 65536px) are very
# different sizes, so the same threshold is relatively stricter for BEV.
# Tune per-tensor separately later if that turns out to matter in practice.
MIN_SEMANTIC_PIXEL_THRESHOLD = 50


def _notable_class_pixel_counts(class_map, class_info):
    """
    Returns {class_name: pixel_count} for every notable class (message is
    not None) in class_info -- the raw, pre-threshold signal
    _detect_notable_classes_from_counts() below thresholds into a
    detection. class_map is already a per-pixel argmaxed class-index map
    (done server-side, in the Colab bridge's /predict handler, before it's
    ever sent back -- see model_loader.get_model_prediction()'s docstring),
    not raw per-class logits, so pixel count (not a 0-1 confidence) is the
    actual signal available here.
    """
    return {
        info["class_name"]: int((class_map == class_index).sum().item())
        for class_index, info in class_info.items()
        if info["message"] is not None
    }


def _detect_notable_classes_from_counts(pixel_counts, class_info, min_pixel_threshold):
    """
    Reduces _notable_class_pixel_counts()'s raw counts down to the
    pre-written messages for whichever notable classes cleared
    min_pixel_threshold -- a handful of stray misclassified pixels
    shouldn't count as a detection. Takes the already-computed counts
    rather than class_map directly so interpret_semantic_output() doesn't
    scan the same tensor twice for the raw-count log line and the
    thresholded messages.
    """
    messages = []

    for class_index, info in class_info.items():
        if info["message"] is None:
            continue

        if pixel_counts[info["class_name"]] >= min_pixel_threshold:
            messages.append(info["message"])

    return messages


def interpret_semantic_output(
    pred_semantic,
    pred_bev_semantic,
    min_pixel_threshold=MIN_SEMANTIC_PIXEL_THRESHOLD
):
    """
    Interprets the model's own semantic segmentation heads in plain English.

    pred_semantic (forward-facing camera view) and pred_bev_semantic
    (bird's-eye view -- the only one of the two that distinguishes traffic
    light color) are already per-pixel argmaxed class-index maps (see
    _notable_class_pixel_counts()'s docstring), not raw per-class logits.

    Returns (messages, class_scores):
        messages: the plain-English messages for whatever notable classes
            are present above min_pixel_threshold pixels, e.g. ["Traffic
            light (red) detected", "Pedestrian detected"] -- or an empty
            list if nothing notable was found in either.
        class_scores: {"pred_semantic": {class_name: pixel_count, ...},
            "pred_bev_semantic": {class_name: pixel_count, ...}} -- every
            notable class's raw pre-threshold pixel count, kept as two
            separate per-tensor dicts (not merged/flattened) since both
            tensors have their own "vehicle" class at very different pixel
            scales (262144px vs 65536px total) -- merging would silently
            drop one of the two counts whenever both fire the same tick.
    """
    pred_semantic_counts = _notable_class_pixel_counts(pred_semantic, SEMANTIC_CLASS_INFO)
    pred_bev_semantic_counts = _notable_class_pixel_counts(pred_bev_semantic, BEV_SEMANTIC_CLASS_INFO)

    messages = []
    messages.extend(_detect_notable_classes_from_counts(pred_semantic_counts, SEMANTIC_CLASS_INFO, min_pixel_threshold))
    messages.extend(_detect_notable_classes_from_counts(pred_bev_semantic_counts, BEV_SEMANTIC_CLASS_INFO, min_pixel_threshold))

    class_scores = {
        "pred_semantic": pred_semantic_counts,
        "pred_bev_semantic": pred_bev_semantic_counts,
    }

    return messages, class_scores


# -------------------------
# BEV coordinate projection
# -------------------------

def project_to_bev_pixel(vehicle, world_location, tfpp_config):
    """
    Projects a world-frame carla.Location into (row, col) pixel coordinates
    of the model's BEV grid (pred_bev_semantic / pred_bounding_box's spatial
    grid), in the vehicle's current local frame.

    Uses the exact convention the checkpoint's own training pipeline uses to
    build its BEV targets (PCLA/pcla_agents/transfuserv4/transfuser_utils.
    bb_vehicle_to_image_system(), confirmed against data.py's get_targets(),
    which indexes the resulting grid as array[row, col]): row tracks forward
    distance (vehicle-local x), column tracks lateral/right distance
    (vehicle-local y), both scaled by tfpp_config.pixels_per_meter and
    shifted so (tfpp_config.min_x, tfpp_config.min_y) lands at pixel (0, 0).
    vehicle.get_transform().inverse_transform() (CARLA >=0.9.13) does the
    world->vehicle-local conversion directly -- no manual matrix math needed.

    Returns None if world_location falls outside the BEV grid's covered area
    this tick (behind ego, or beyond tfpp_config.max_x/max_y) -- there's no
    pred_bev_semantic signal for a location the grid doesn't cover.
    """
    local = vehicle.get_transform().inverse_transform(world_location)

    row = round(local.x * tfpp_config.pixels_per_meter - tfpp_config.min_x * tfpp_config.pixels_per_meter)
    col = round(local.y * tfpp_config.pixels_per_meter - tfpp_config.min_y * tfpp_config.pixels_per_meter)

    if not (0 <= row < tfpp_config.lidar_resolution_height and 0 <= col < tfpp_config.lidar_resolution_width):
        return None

    return row, col


# BEV_SEMANTIC_CLASS_INFO's three traffic-light-color messages, reduced to
# the perception vocabulary error_detection.evaluate_tick() compares against
# ("<color>_light" -- see scenarios/model/intersection/simple_stop_go.py's
# LIGHT_TO_EXPECTED). Order matters only for the rare case where more than
# one color fires in the same frame (misclassification, or more than one
# light visible) -- first match wins.
DETECTED_COLOR_MESSAGE_TO_PERCEPTION = [
    ("Traffic light (red) detected", "red_light"),
    ("Traffic light (yellow) detected", "yellow_light"),
    ("Traffic light (green) detected", "green_light"),
]


def _actual_perception_from_detections(detected_messages):
    """
    Reduces interpret_semantic_output()'s messages down to the single
    perception string error_detection compares against this tick, independent
    of what ground truth says is actually relevant -- "none" if the model
    didn't fire any traffic-light-color class this tick.
    """
    for message, perception in DETECTED_COLOR_MESSAGE_TO_PERCEPTION:
        if message in detected_messages:
            return perception

    return "none"


# -------------------------
# Pure decision function
# -------------------------

def decide_action(
    pred_target_speed,
    ego_speed_mps=None,
    distance_to_target_m=None,
    mu_friction=None
):
    """
    Maps the model's predicted target speed onto the nearest of the 4
    canonical speed bins, then returns that bin's action. Pure function --
    no CARLA calls -- so it can be unit tested or tuned without a live
    simulation.

    ego_speed_mps/distance_to_target_m/mu_friction are optional -- when all
    three are given and the nearest bin is "stop", this also checks whether
    "stop" is only barely sufficient (required physical stopping distance,
    the same v^2 / (2*mu*g) bound error_detection.timely_action() uses,
    already consumes HARD_BRAKE_URGENCY_RATIO or more of distance_to_target_m)
    and returns "hard_brake" instead if so. See this module's docstring
    comment above TARGET_SPEED_TO_ACTION for why "hard_brake" is a relabeling
    of the "stop" bin rather than a genuine 5th model output. Any caller that
    omits these (the plain 1-argument call) gets the original
    stop/slow/go/go_fast-only behavior, unchanged.
    """
    nearest_bin = min(
        TARGET_SPEED_TO_ACTION,
        key=lambda bin_speed: abs(bin_speed - pred_target_speed)
    )

    action = TARGET_SPEED_TO_ACTION[nearest_bin]

    if (
        action == "stop"
        and ego_speed_mps is not None
        and distance_to_target_m is not None
        and mu_friction is not None
        and distance_to_target_m > 0
    ):
        required_stopping_distance_m = (
            ego_speed_mps ** 2
        ) / (2 * mu_friction * error_detection.GRAVITY)

        if required_stopping_distance_m >= HARD_BRAKE_URGENCY_RATIO * distance_to_target_m:
            return "hard_brake"

    return action


# -------------------------
# One-time startup diagnostic
# -------------------------

def log_model_startup_prediction(model, vehicle, world, forward_distance=15.0):
    """
    Runs the model exactly once, at whatever the vehicle's current state is
    (intended to be called right after the ego vehicle spawns and its
    sensors attach, before any movement starts), and logs the raw
    prediction (full-precision bin probabilities, not just the winning
    discrete bin) and resulting action.

    This isolates "does the model output something stop-like when facing a
    red light at a standstill" from any movement/state-machine logic --
    it is a diagnostic only, not part of the main control loop, and does
    not apply any action to the vehicle.
    """
    pred_target_speed, _pred_semantic, _pred_bev_semantic, target_speed_probabilities = get_model_prediction(
        model,
        vehicle,
        world,
        forward_distance=forward_distance
    )

    if pred_target_speed is None:
        # No prior decision exists yet to hold at this point (this runs
        # once, before the control loop starts) -- surface the miss instead.
        logging.warning("[startup diagnostic] Colab endpoint unreachable -- no prediction available.")
        return None, None

    action = decide_action(pred_target_speed)

    logging.info(
        f"[startup diagnostic] pred_target_speed={pred_target_speed:.2f} -> action={action} "
        f"(bin_probabilities={_round_probabilities(target_speed_probabilities)})"
    )

    return pred_target_speed, action


# -------------------------
# Model-driven monitoring loop
# -------------------------

def monitor_with_model_and_act(
    world,
    vehicle,
    model,
    traffic_light,
    num_ticks,
    light_log,
    action_log,
    light_to_expected,
    mu_friction,
    check_interval_ticks=5,
    current_command_state=None,
    forward_distance=15.0
):
    """
    Drives the vehicle with the model's predictions instead of the hardcoded
    light-based rules, over num_ticks. Matches the role/signature style of
    hazard_controls.monitor_light_and_hazards_and_act().

    light_to_expected/mu_friction are scenario-owned (see scenarios/model/
    intersection/simple_stop_go.py's LIGHT_TO_EXPECTED/MU_FRICTION) --
    required here rather than defaulted, so a scenario can't silently
    inherit a value that wasn't actually tuned for it. light_to_expected
    maps the ground-truth light color read this tick (see
    controls.light_controls.traffic_light_state_to_string -- "red"/"yellow"/
    "green") to that phase's {"expected_perception", "expected_action"},
    since the correct answer changes with the light: expected_perception
    is what the model's semantic output should show as the light's actual
    color, and expected_action is the safe action for that phase (e.g.
    "stop" for red, "slow" for yellow, "go" for green) -- resolved fresh
    every check, not hardcoded to one phase. Forwarded to
    error_detection.evaluate_tick() (see #4 below).

    current_command_state is a single-key mutable dict ({"action": ...})
    owned by the caller and passed back in on every phase of the scenario
    (see scenarios/model/intersection/simple_stop_go.py) -- NOT a local
    variable here, because run_fixed_traffic_light_sequence calls this
    function once per light phase (currently red, then green), and a local
    would reset to its default at the start of every phase, discarding
    whatever the model had decided a moment before. If not given, a
    fresh dict defaulting to "go" is created (single-phase/standalone use).

    Every tick, regardless of whether this tick re-evaluates the model,
    current_command_state["action"] is (re)applied to the vehicle -- this is
    the *only* place vehicle control is applied. There is no separate
    one-shot "start moving" call anywhere else; tick 0 of the first phase
    applies the "go" default itself.

    Every check_interval_ticks:
        1. Ask the model for its prediction (get_model_prediction() --
           camera/LiDAR frame in, pred_target_speed out; no route/GPS).
        2. Decide the action (decide_action()) and store it into
           current_command_state -- it then stays in effect on every
           subsequent tick (via the per-tick apply above) until the next
           check changes it again.
        3. Interpret the model's own semantic segmentation output
           (interpret_semantic_output()) into plain-English detection
           messages (e.g. "Traffic light (red) detected"), stored in the
           action_log entry as "detections" -- plus every notable class's
           raw pre-threshold pixel count, logged every check as
           class_scores (see #5).
        4. Look up light_to_expected[light_color] (ground truth, read here
           only for after-the-fact comparison -- never passed into
           get_model_prediction() or anything the model sees) and check the
           model's perception and decision against that phase's expected
           values via error_detection.evaluate_tick() (see
           config.ml_model.setup.error_detection).
        5. Log the decision, the previous command, whether it changed, the
           ground-truth light color, and the three error_detection outcomes
           (perceieved_right / action_correct / timely_action) in one line,
           then the full speed_probabilities distribution (get_model_prediction()'s
           target_speed_probabilities, the same softmax output confidence
           above is already derived from -- exposed here in full so a
           near-tie between bins is visible, not just the winning one) in a
           second line, then class_scores (see #3) in a third. All logged
           every check, not just on change, so a rapid decision flip (e.g.
           stop -> go on the very next check) is visible instead of
           silently missing from the log.

    world.tick() (not world.wait_for_tick()) is called at the end of every
    tick -- this function requires the world to already be in synchronous
    mode (see carla_config.enable_synchronous_mode()), since inference here
    (~1s+) is far slower than one simulation step; without synchronous mode,
    CARLA free-runs on its own real-time clock while inference is blocking,
    so the vehicle keeps moving under the previous decision for that entire
    duration and every tick-count-based timing assumption (light phase
    durations, check_interval_ticks) silently stops meaning what it says.
    """

    if current_command_state is None:
        current_command_state = {"action": "go"}

    for tick in range(num_ticks):
        if tick % check_interval_ticks == 0:
            pred_target_speed, pred_semantic, pred_bev_semantic, target_speed_probabilities = get_model_prediction(
                model,
                vehicle,
                world,
                forward_distance=forward_distance
            )

            # Ground truth, for comparison only -- see docstring above.
            light_color = log_light_state(traffic_light, light_log, tick=tick)

            if pred_target_speed is None:
                # Colab endpoint unreachable this check -- leave
                # current_command_state untouched. The per-tick apply below
                # (the only place vehicle.apply_control() happens) keeps
                # reapplying whatever it already held, same as it does on
                # ticks between checks.
                held_action = current_command_state["action"]

                logging.warning(
                    f"tick={tick} Colab endpoint unreachable -- holding last decision ({held_action})."
                )

                action_log.append(
                    {
                        "tick": tick,
                        "timestamp": time.time(),
                        "pred_target_speed": None,
                        "target_speed_probabilities": None,
                        "decided_action": held_action,
                        "previous_command": held_action,
                        "changed": False,
                        "real_light_color": light_color,
                        "detections": [],
                        "perceieved_right": None,
                        "action_correct": None,
                        "timely_action": None
                    }
                )
            else:
                # Computed before decide_action() (not just at the
                # error_detection call below) so decide_action() can itself
                # weigh urgency -- see its docstring's "hard_brake" case.
                ego_speed_mps = get_vehicle_speed_mps(vehicle)
                distance_to_target_m = get_distance_to_stop_line(vehicle, traffic_light)

                decided_action = decide_action(
                    pred_target_speed,
                    ego_speed_mps=ego_speed_mps,
                    distance_to_target_m=distance_to_target_m,
                    mu_friction=mu_friction
                )

                detected_messages, class_scores = interpret_semantic_output(pred_semantic, pred_bev_semantic)

                previous_action = current_command_state["action"]
                action_changed = decided_action != previous_action

                current_command_state["action"] = decided_action

                # Ground truth: what should the model be perceiving/doing this
                # phase? traffic_light here is already the one
                # find_vehicle_traffic_light() matched to ego's lane/path, so
                # light_color alone (not a separate relevance check) picks the
                # right expectation -- resolved fresh every check since the
                # correct answer changes with the light (red/yellow/green).
                expected = light_to_expected[light_color]

                # Model's own perception, not ground truth -- reduces
                # interpret_semantic_output()'s BEV color messages down to the
                # single string error_detection compares against expected.
                actual_perception = _actual_perception_from_detections(detected_messages)

                error_result = error_detection.evaluate_tick(
                    expected_perception=expected["expected_perception"],
                    actual_perception=actual_perception,
                    expected_action=expected["expected_action"],
                    decision=decided_action,
                    ego_speed_mps=ego_speed_mps,
                    distance_to_target_m=distance_to_target_m,
                    mu_friction=mu_friction
                )

                action_log.append(
                    {
                        "tick": tick,
                        "timestamp": time.time(),
                        "pred_target_speed": pred_target_speed,
                        "target_speed_probabilities": target_speed_probabilities,
                        "decided_action": decided_action,
                        "previous_command": previous_action,
                        "changed": action_changed,
                        "real_light_color": light_color,
                        "detections": detected_messages,
                        "perceieved_right": error_result["perceieved_right"],
                        "action_correct": error_result["action_correct"],
                        "timely_action": error_result["timely_action"]
                    }
                )

                confidence = max(target_speed_probabilities.values())

                logging.info(
                    f"// DECISION // tick={tick} decision={decided_action} pred_speed={pred_target_speed:.2f} "
                    f"previous={previous_action} confidence={confidence:.2f} newdecision={action_changed} "
                    f"real_light_color={light_color} "
                    f"perceieved_right={_format_optional(error_result['perceieved_right'])} "
                    f"action_correct={_format_optional(error_result['action_correct'])} "
                    f"timely_action={_format_optional(error_result['timely_action'])}"
                )

                logging.info(f"// SPEED // tick={tick} speed_probabilities={_round_probabilities(target_speed_probabilities)}")

                logging.info(f"// DETECTION // tick={tick} class_scores={class_scores}")

        # Persistent application runs every tick.  The fidelity path reapplies
        # the last PCLA PID command; legacy callers retain action presets.
        ACTION_DISPATCH[current_command_state["action"]](vehicle)

        follow_vehicle_with_spectator(
            world,
            vehicle,
            distance=12,
            height=5,
            pitch=-8
        )

        # Explicit world.tick(), not world.wait_for_tick() -- see docstring.
        world.tick()

    return light_log, action_log


# -------------------------
# Model-driven monitoring loop -- static obstacle scenarios
# -------------------------

def _default_distance_to_object_m(vehicle, object_actor):
    """
    Default ground-truth distance: straight-line distance from ego to the
    object's near surface (center distance minus the object's own
    bounding-box extent), appropriate for an object ego is heading straight
    at (chair_object.py's stationary prop, T_bone_crash.py's broadside
    crash car). Not appropriate for an object beside/behind ego that hasn't
    merged into its lane yet -- see compute_distance_to_object_m's docstring
    below.
    """
    return (
        vehicle.get_location().distance(object_actor.get_location())
        - object_actor.bounding_box.extent.x
    )


def monitor_with_model_and_act_for_object(
    world,
    vehicle,
    model,
    object_actor,
    num_ticks,
    action_log,
    resolve_expected,
    resolve_actual_perception,
    mu_friction,
    check_interval_ticks=5,
    current_command_state=None,
    forward_distance=15.0,
    on_each_tick=None,
    compute_distance_to_object_m=None,
    inference_callback=None,
    collision_data=None,
    distance_log_name="distance_to_object",
    distance_display_name="Distance to Object",
    object_speed_log_name=None,
    object_speed_display_name=None,
    collision_actor_display_name="Object",
    use_pcla_control=False,
    navigation_state=None,
):
    """
    Object-scenario counterpart to monitor_with_model_and_act() -- same
    per-tick apply/check_interval_ticks/error_detection shape, but built
    around a single object_actor instead of a traffic_light, since there's
    no light phase/light_log involved here at all (see scenarios/model/
    object_in_road/chair_object.py).

    object_actor need not be static -- on_each_tick, if provided, is called
    once per tick (as on_each_tick(tick)) before that tick's command is
    applied, same contract as hazard_controls.monitor_light_and_hazards_and_act's
    on_each_tick -- for an object that needs continuous per-tick control of
    its own (e.g. scenarios/model/intersection/T_bone_crash.py's crash car,
    steered through a scripted turn every tick once triggered), unlike
    chair_object.py's stationary prop which never needs this.

    resolve_expected/resolve_actual_perception/mu_friction are
    scenario-owned (see LIGHT_TO_EXPECTED's docstring in
    monitor_with_model_and_act() for why these aren't defaulted here) --
    unlike the traffic-light scenario's light_to_expected (a plain dict,
    since ground truth is one of 3 fixed light colors), both are callbacks
    here because the relevant ground truth (distance to the object) is
    continuous, and turning the model's raw BEV output into a perception
    string requires scenario-specific spatial logic that has no equivalent
    for a class-name match (see chair_object.py's module docstring):

        resolve_expected(distance_to_object_m) -> {"expected_perception", "expected_action"}
        resolve_actual_perception(pred_bev_semantic, vehicle, object_actor, tfpp_config) -> str

    current_command_state: see monitor_with_model_and_act()'s docstring --
    same contract, just without the light-phase framing (there's only ever
    one phase here, so a fresh dict is the common case, but the caller can
    still pass one through if this ever needs to chain with another phase).

    compute_distance_to_object_m(vehicle, object_actor) -> float, optional:
    overrides how ground-truth distance_to_object_m is computed each check.
    Defaults to _default_distance_to_object_m() (straight-line distance to
    the object's near surface) -- correct for an object ego is heading
    straight at, but not for one that starts beside/behind ego and merges
    into its lane over time (e.g. scenarios/model/intersection/
    interception_crash.py's cut-in crash car, where the relevant ground
    truth is the longitudinal gap from ego's front bumper to the crash
    car's rear bumper once it's actually in ego's lane, not raw Euclidean
    distance to a car that's currently off to the side).

    Every check_interval_ticks:
        1. Ask the model for its prediction. The PCLA path uses
           get_model_control_prediction() and, when navigation_state is
           present, supplies its route-derived target and command.
        2. Decide the action (decide_action()) and store it into
           current_command_state, same as monitor_with_model_and_act().
        3. Interpret the model's own semantic segmentation output
           (interpret_semantic_output()) for the log's "detections" list --
           this scenario's object won't appear here (see chair_object.py),
           but other notable classes (e.g. another vehicle) still would --
           plus every notable class's raw pre-threshold pixel count, logged
           every check as class_scores (see #5).
        4. Compute ground-truth distance_to_object_m (vehicle location to
           object_actor location, minus the object's own bounding-box
           extent so it reflects distance to the object's near surface, not
           its center), resolve_expected() from it, resolve_actual_perception()
           from the model's own pred_bev_semantic, and check both via
           error_detection.evaluate_tick().
        5. Log the decision, the previous command, whether it changed, the
           ground-truth distance to the object, and the three
           error_detection outcomes in one line -- same shape as
           monitor_with_model_and_act()'s log line, with real_light_color
           swapped for distance_to_object_m -- then the full
           speed_probabilities distribution in a second line and
           class_scores (see #3) in a third, every check regardless of
           whether the object is currently in range -- watching stop/slow
           logits as distance closes (or fails to rise at all) is the point.

    world.tick() (not world.wait_for_tick()) is called at the end of every
    tick -- see monitor_with_model_and_act()'s docstring for why (requires
    synchronous mode).

    navigation_state is an optional PCLA route adapter. Object in Road passes
    one so each PCLA inference receives a world-route-derived target and
    lagged RoadOption command. Existing intersection callers omit it and keep
    the model loader's backward-compatible synthetic navigation fallback.
    """

    if current_command_state is None:
        current_command_state = {
            "action": "stop" if use_pcla_control else "go"
        }

    if use_pcla_control and "control" not in current_command_state:
        # Mirrors PCLA's safe initialization while the first complete LiDAR
        # sweep/prediction is not yet available.
        current_command_state["control"] = carla.VehicleControl(
            steer=0.0,
            throttle=0.0,
            brake=1.0,
        )

    if compute_distance_to_object_m is None:
        compute_distance_to_object_m = _default_distance_to_object_m

    fixed_delta_seconds = world.get_settings().fixed_delta_seconds
    previous_inference_frame = None

    for tick in range(num_ticks):
        inference_entry = None
        class_scores = None
        confidence = None
        navigation_input = None

        if on_each_tick is not None:
            on_each_tick(tick)

        if tick % check_interval_ticks == 0:
            snapshot = world.get_snapshot()
            carla_frame = snapshot.frame
            control_prediction = None
            if use_pcla_control:
                if navigation_state is not None:
                    navigation_input = navigation_state.run_step(
                        vehicle.get_transform()
                    )
                control_prediction = get_model_control_prediction(
                    model,
                    vehicle,
                    world,
                    forward_distance=forward_distance,
                    target_point=(
                        navigation_input.target_point
                        if navigation_input is not None else None
                    ),
                    command=(
                        navigation_input.command_value
                        if navigation_input is not None else None
                    ),
                )
                if control_prediction is None:
                    pred_target_speed = None
                    pred_semantic = None
                    pred_bev_semantic = None
                    target_speed_probabilities = None
                else:
                    pred_target_speed = control_prediction.processed_target_speed
                    pred_semantic = control_prediction.pred_semantic
                    pred_bev_semantic = control_prediction.pred_bev_semantic
                    target_speed_probabilities = control_prediction.target_speed_probabilities
            else:
                pred_target_speed, pred_semantic, pred_bev_semantic, target_speed_probabilities = get_model_prediction(
                    model,
                    vehicle,
                    world,
                    forward_distance=forward_distance
                )

            # Ground truth: see compute_distance_to_object_m's docstring
            # above for what "distance" means for this scenario's object.
            distance_to_object_m = compute_distance_to_object_m(vehicle, object_actor)
            ego_speed_mps = get_vehicle_speed_mps(vehicle)
            object_velocity = object_actor.get_velocity()
            object_speed_mps = math.sqrt(
                object_velocity.x ** 2
                + object_velocity.y ** 2
                + object_velocity.z ** 2
            )
            acceleration = vehicle.get_acceleration()
            ego_acceleration_mps2 = math.sqrt(
                acceleration.x ** 2 + acceleration.y ** 2 + acceleration.z ** 2
            )
            expected = resolve_expected(distance_to_object_m)

            if pred_target_speed is None:
                # Colab endpoint unreachable this check -- leave
                # current_command_state untouched. The per-tick apply below
                # (the only place vehicle.apply_control() happens) keeps
                # reapplying whatever it already held, same as it does on
                # ticks between checks.
                held_action = current_command_state["action"]

                logging.warning(
                    f"[TICK {tick:03d}] Remote inference failed; holding last decision ({held_action.upper()})."
                )

                inference_entry = {
                    "tick": tick,
                    "timestamp": time.time(),
                    "carla_frame": carla_frame,
                    "simulation_time_s": tick * fixed_delta_seconds,
                    "ego_speed_mps": ego_speed_mps,
                    "ego_acceleration_mps2": ego_acceleration_mps2,
                    "pred_target_speed": None,
                    "target_speed_probabilities": None,
                    "decided_action": held_action,
                    "previous_command": held_action,
                    "changed": False,
                    "distance_to_object_m": distance_to_object_m,
                    "object_speed_mps": object_speed_mps,
                    "detections": [],
                    "class_scores": None,
                    "expected_perception": expected["expected_perception"],
                    "actual_perception": None,
                    "expected_safe_action": expected["expected_action"],
                    "perceieved_right": None,
                    "action_correct": None,
                    "timely_action": None,
                }
            else:
                # Computed before decide_action() (not just at the
                # error_detection call below) so decide_action() can itself
                # weigh urgency -- see its docstring's "hard_brake" case.
                decided_action = decide_action(
                    pred_target_speed,
                    ego_speed_mps=ego_speed_mps,
                    distance_to_target_m=distance_to_object_m,
                    mu_friction=mu_friction
                )

                detected_messages, class_scores = interpret_semantic_output(pred_semantic, pred_bev_semantic)

                previous_action = current_command_state["action"]
                action_changed = decided_action != previous_action

                current_command_state["action"] = decided_action

                if use_pcla_control:
                    control_result = model.tfpp_controller.run(
                        control_prediction.pred_checkpoint,
                        control_prediction.processed_target_speed,
                        control_prediction.signed_forward_speed,
                    )
                    current_command_state["control"] = control_result.control

                actual_perception = resolve_actual_perception(
                    pred_bev_semantic,
                    vehicle,
                    object_actor,
                    model.tfpp_config
                )

                error_result = error_detection.evaluate_tick(
                    expected_perception=expected["expected_perception"],
                    actual_perception=actual_perception,
                    expected_action=expected["expected_action"],
                    decision=decided_action,
                    ego_speed_mps=ego_speed_mps,
                    distance_to_target_m=distance_to_object_m,
                    mu_friction=mu_friction
                )

                confidence = max(target_speed_probabilities.values())
                inference_entry = {
                    "tick": tick,
                    "timestamp": time.time(),
                    "carla_frame": carla_frame,
                    "simulation_time_s": tick * fixed_delta_seconds,
                    "ego_speed_mps": ego_speed_mps,
                    "ego_acceleration_mps2": ego_acceleration_mps2,
                    "pred_target_speed": pred_target_speed,
                    "target_speed_probabilities": target_speed_probabilities,
                    "decided_action": decided_action,
                    "previous_command": previous_action,
                    "changed": action_changed,
                    "distance_to_object_m": distance_to_object_m,
                    "object_speed_mps": object_speed_mps,
                    "detections": detected_messages,
                    "class_scores": class_scores,
                    "expected_perception": expected["expected_perception"],
                    "actual_perception": actual_perception,
                    "expected_safe_action": expected["expected_action"],
                    "perceieved_right": error_result["perceieved_right"],
                    "action_correct": error_result["action_correct"],
                    "timely_action": error_result["timely_action"],
                }
                if use_pcla_control:
                    inference_entry.update({
                        "pred_checkpoint_x": control_result.checkpoint_x,
                        "pred_checkpoint_y": control_result.checkpoint_y,
                        "processed_target_speed_mps": control_prediction.processed_target_speed,
                        "brake_probability": control_prediction.brake_probability,
                    })

            if navigation_input is not None:
                inference_entry.update({
                    "target_point_x": float(navigation_input.target_point[0]),
                    "target_point_y": float(navigation_input.target_point[1]),
                    "road_command": navigation_input.command_name,
                    "active_route_waypoint_index": (
                        navigation_input.active_route_waypoint_index
                    ),
                    "active_route_world_x": navigation_input.active_route_world_x,
                    "active_route_world_y": navigation_input.active_route_world_y,
                })

        # Persistent command application -- the only place vehicle.apply_control()
        # (via controls.actions) happens. Runs every tick so the vehicle keeps
        # doing whatever was last decided instead of reverting after one tick.
        if use_pcla_control:
            vehicle.apply_control(current_command_state["control"])
        else:
            ACTION_DISPATCH[current_command_state["action"]](vehicle)

        if inference_entry is not None:
            if use_pcla_control:
                # This is the command issued by the new inference (or the
                # deliberately held prior model command after a failed call),
                # not CARLA's previous-tick server readback.
                applied_control = current_command_state["control"]
            else:
                applied_control = vehicle.get_control()
            new_collision_events = [
                event for event in (collision_data or [])
                if (previous_inference_frame is None or event["frame"] > previous_inference_frame)
                and event["frame"] <= inference_entry["carla_frame"]
            ]
            inference_entry.update({
                "throttle": float(applied_control.throttle),
                "brake": float(applied_control.brake),
                "steer": float(applied_control.steer),
                "time_to_collision_s": compute_time_to_collision(vehicle, object_actor),
                "collision": bool(new_collision_events),
            })
            previous_inference_frame = inference_entry["carla_frame"]
            action_log.append(inference_entry)

            if inference_callback is not None:
                inference_callback(inference_entry)

            if navigation_input is not None:
                logging.info(
                    "[PCLA NAVIGATION] target_point=(%.4f, %.4f) "
                    "road_command=%s active_route_waypoint_index=%d "
                    "active_route_world=(%.4f, %.4f)",
                    navigation_input.target_point[0],
                    navigation_input.target_point[1],
                    navigation_input.command_name,
                    navigation_input.active_route_waypoint_index,
                    navigation_input.active_route_world_x,
                    navigation_input.active_route_world_y,
                )

            if pred_target_speed is not None:
                camera_scores = class_scores["pred_semantic"]
                bev_scores = class_scores["pred_bev_semantic"]
                logging.info(
                    f"[TICK {tick:03d}] time={inference_entry['simulation_time_s']:.2f}s | "
                    f"{distance_log_name}={distance_to_object_m:.2f}m | "
                    f"ego_speed={ego_speed_mps:.2f}m/s | "
                    + (
                        f"{object_speed_log_name}={object_speed_mps:.2f}m/s | "
                        if object_speed_log_name else ""
                    )
                    + f"target_speed={pred_target_speed:.2f}m/s "
                    f"confidence={confidence:.2f} | action={current_command_state['action'].upper()} | "
                    f"throttle={applied_control.throttle:.2f} brake={applied_control.brake:.2f} "
                    f"steer={applied_control.steer:.2f}"
                )
                logging.info(
                    f"[PERCEPTION] camera: vehicle={camera_scores['vehicle']} "
                    f"pedestrian={camera_scores['pedestrian']} "
                    f"traffic_light={camera_scores['traffic_light']} | "
                    f"BEV: vehicle={bev_scores['vehicle']} walker={bev_scores['walker']} "
                    f"stop_sign={bev_scores['stop_sign']} red_light={bev_scores['traffic_light_red']} "
                    f"green_light={bev_scores['traffic_light_green']}"
                )
                if use_pcla_control:
                    logging.info(
                        "[PCLA CONTROL] processed_target_speed=%.4fm/s "
                        "brake_probability=%.6f checkpoint=(%.4f, %.4f) "
                        "predicted_angle=%.6f",
                        control_prediction.processed_target_speed,
                        control_prediction.brake_probability,
                        control_result.checkpoint_x,
                        control_result.checkpoint_y,
                        control_result.predicted_angle,
                    )

                if inference_entry["changed"]:
                    separator = "=" * 61
                    speed_line = ""
                    if object_speed_display_name:
                        speed_line = (
                            f"{object_speed_display_name}: "
                            f"{object_speed_mps:.2f} m/s\n"
                        )
                    logging.info(
                        "\n%s\nDECISION CHANGE\nTick: %d\n%s -> %s\n\n"
                        "%s: %.2f m\nEgo Speed: %.2f m/s\n%s"
                        "Target Speed: %.2f m/s\nConfidence: %.2f\n%s",
                        separator, tick, inference_entry["previous_command"].upper(),
                        inference_entry["decided_action"].upper(), distance_display_name,
                        distance_to_object_m, ego_speed_mps, speed_line,
                        pred_target_speed, confidence, separator,
                    )

                if inference_entry["timely_action"] is False:
                    logging.warning(
                        "Insufficient stopping distance at tick %d: "
                        "%s=%.2fm, ego_speed=%.2fm/s.",
                        tick, distance_log_name, distance_to_object_m, ego_speed_mps,
                    )

            if new_collision_events:
                event = new_collision_events[-1]
                collision_actor = event.get("impact_actor_type") or "unknown actor"
                logging.warning(
                    "\n!!!!!!!!!!!!!!!!!!!!!!! COLLISION !!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    "Tick: %d\nEgo Speed: %.2f m/s\n%s: %s\n"
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                    tick, ego_speed_mps, collision_actor_display_name, collision_actor,
                )

        follow_vehicle_with_spectator(
            world,
            vehicle,
            distance=12,
            height=5,
            pitch=-8
        )

        # Explicit world.tick(), not world.wait_for_tick() -- see docstring.
        world.tick()

    return action_log
