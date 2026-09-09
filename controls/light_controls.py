import carla
import logging
import time

from controls.actions import slow_vehicle, stop_vehicle
from config.CARLA_actors.sensors import follow_vehicle_with_spectator
from config.CARLA_actors.carla_config import advance_simulation


# -------------------------
# Traffic light state helpers
# -------------------------

def traffic_light_state_to_string(state):
    """
    Converts a CARLA TrafficLightState into a readable string.
    """

    if state == carla.TrafficLightState.Red:
        return "red"

    if state == carla.TrafficLightState.Yellow:
        return "yellow"

    if state == carla.TrafficLightState.Green:
        return "green"

    if state == carla.TrafficLightState.Off:
        return "off"

    if state == carla.TrafficLightState.Unknown:
        return "unknown"

    return str(state)


def get_light_state(traffic_light):
    """
    Gets the current traffic light state from CARLA.
    Returns the raw CARLA state.
    """

    if traffic_light is None:
        raise ValueError("traffic_light is None. Cannot get traffic light state.")

    return traffic_light.get_state()


def get_light_color(traffic_light):
    """
    Gets the current traffic light color as a readable string.
    Example: red, yellow, green
    """

    state = get_light_state(traffic_light)

    return traffic_light_state_to_string(state)


def log_light_state(traffic_light, light_log, tick=None, phase=None):
    """
    Reads the current traffic light state and stores it in light_log.

    light_log should be a list.
    Each entry records:
        traffic light id
        phase label (e.g. "yellow"/"red"/"green"), if given -- tick numbers
        reset to 0 for each phase, so this disambiguates "tick 20" across
        phases when the same light_log/action_log is shared across phases
        tick number
        timestamp
        light color
    """

    light_color = get_light_color(traffic_light)

    log_entry = {
        "traffic_light_id": traffic_light.id,
        "phase": phase,
        "tick": tick,
        "timestamp": time.time(),
        "light_color": light_color
    }

    light_log.append(log_entry)

    logging.debug(
        f"Traffic light {traffic_light.id} state logged: {light_color}"
    )

    return light_color


def get_distance_to_stop_line(vehicle, traffic_light):
    """
    Returns the vehicle's current distance (meters) to the nearest of
    traffic_light's stop waypoints -- ground-truth input for
    error_detection.timely_action()'s physical stopping-distance check.

    traffic_light is expected to already be one whose stop waypoints match
    ego's path (e.g. the object find_vehicle_traffic_light() returns),
    which is guaranteed to have at least one stop waypoint.
    """

    stop_waypoints = traffic_light.get_stop_waypoints()

    if not stop_waypoints:
        return None

    vehicle_location = vehicle.get_location()

    return min(
        stop_waypoint.transform.location.distance(vehicle_location)
        for stop_waypoint in stop_waypoints
    )


# -------------------------
# Traffic light finder
# -------------------------

def find_vehicle_traffic_light(
    world,
    vehicle,
    max_ticks=150,
    creep_throttle=0.2,
    search_distance=80.0,
    waypoint_step=2.0,
    match_distance=4.0
):
    """
    Finds the traffic light affecting the ego vehicle's current lane/path.

    This function does NOT move the vehicle.

    It searches forward from the ego vehicle's current waypoint and selects
    the traffic light whose stop waypoint matches the ego vehicle's path.
    """

    logging.info("Searching for traffic light...")

    # Keep these for compatibility with the main scenario call.
    # They are not used because the vehicle should not creep forward.
    _ = max_ticks
    _ = creep_throttle

    stop_vehicle(vehicle)
    advance_simulation(world)

    carla_map = world.get_map()
    vehicle_location = vehicle.get_location()

    current_waypoint = carla_map.get_waypoint(
        vehicle_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if current_waypoint is None:
        logging.warning("No driving waypoint found for ego vehicle.")
        return None

    # -------------------------
    # Build forward ego path
    # -------------------------

    ego_path_waypoints = [current_waypoint]
    active_waypoint = current_waypoint
    distance_traveled = 0.0

    while distance_traveled < search_distance:
        next_waypoints = active_waypoint.next(waypoint_step)

        if not next_waypoints:
            break

        active_waypoint = next_waypoints[0]
        ego_path_waypoints.append(active_waypoint)
        distance_traveled += waypoint_step

    # -------------------------
    # Get traffic light candidates
    # -------------------------

    candidate_lights = world.get_traffic_lights_from_waypoint(
        current_waypoint,
        search_distance
    )

    if not candidate_lights:
        logging.warning("No traffic light found ahead.")
        return None

    # -------------------------
    # Match light stop waypoint to ego path
    # -------------------------

    best_light = None
    best_path_index = None
    best_match_distance = float("inf")

    for traffic_light in candidate_lights:
        stop_waypoints = traffic_light.get_stop_waypoints()

        if not stop_waypoints:
            continue

        for stop_waypoint in stop_waypoints:
            stop_location = stop_waypoint.transform.location

            for path_index, ego_waypoint in enumerate(ego_path_waypoints):
                ego_location = ego_waypoint.transform.location

                same_road = stop_waypoint.road_id == ego_waypoint.road_id
                same_lane = stop_waypoint.lane_id == ego_waypoint.lane_id
                distance = stop_location.distance(ego_location)

                if same_road and same_lane and distance <= match_distance:
                    if best_path_index is None:
                        best_light = traffic_light
                        best_path_index = path_index
                        best_match_distance = distance

                    elif path_index < best_path_index:
                        best_light = traffic_light
                        best_path_index = path_index
                        best_match_distance = distance

                    elif path_index == best_path_index and distance < best_match_distance:
                        best_light = traffic_light
                        best_path_index = path_index
                        best_match_distance = distance

    if best_light is None:
        logging.warning("No traffic light matched the ego vehicle path.")
        return None

    stop_vehicle(vehicle)
    advance_simulation(world)

    logging.info(
        f"Found traffic light at path index {best_path_index}: ID {best_light.id}"
    )

    return best_light

# -------------------------
# Traffic light state controls
# -------------------------

def set_traffic_light_state(traffic_light, state):
    """
    Sets a traffic light to a specific CARLA traffic light state.

    Example states:
        carla.TrafficLightState.Red
        carla.TrafficLightState.Yellow
        carla.TrafficLightState.Green
    """

    if traffic_light is None:
        raise ValueError("traffic_light is None. Cannot set traffic light state.")

    traffic_light.set_state(state)

    logging.info(
        f"Traffic light {traffic_light.id} set to {traffic_light_state_to_string(state)}."
    )


def set_light_red(traffic_light):
    """
    Sets the traffic light to red.
    """

    set_traffic_light_state(
        traffic_light,
        carla.TrafficLightState.Red
    )


def set_light_green(traffic_light):
    """
    Sets the traffic light to green.
    """

    set_traffic_light_state(
        traffic_light,
        carla.TrafficLightState.Green
    )


def set_light_yellow(traffic_light):
    """
    Sets the traffic light to yellow.
    """

    set_traffic_light_state(
        traffic_light,
        carla.TrafficLightState.Yellow
    )

# -------------------------
# Deterministic traffic light sequence
# -------------------------

def force_traffic_light_state(
    world,
    traffic_light,
    state,
    state_name
):
    """
    Forces the traffic light into a specific state and freezes it.

    This prevents CARLA from randomly changing the light timing.
    """

    if traffic_light is None:
        raise ValueError("traffic_light is None. Cannot force traffic light state.")

    traffic_light.set_state(state)
    traffic_light.freeze(True)

    advance_simulation(world)

    detected_color = get_light_color(traffic_light)

    logging.info(f"Light set to {state_name}.")

    return detected_color


def run_fixed_traffic_light_sequence(
    world,
    vehicle,
    traffic_light,
    monitor_function,
    light_log,
    action_log,
    check_interval_ticks=5,
    ticks_per_second=20,
    yellow_seconds=3,
    red_seconds=4,
    green_ticks=150
):
    """
    Runs a deterministic traffic light sequence:

        Yellow for 1 second
        Red for 3 seconds
        Green for the rest of the scenario

    The traffic light is frozen during each phase so CARLA does not
    automatically cycle it.

    Pass yellow_seconds=0 to skip the yellow phase entirely (starts directly
    at red) instead of just zeroing its monitored ticks -- otherwise the
    light would still be forced to Yellow for one tick before Red overwrites
    it.
    """

    yellow_ticks = int(yellow_seconds * ticks_per_second)
    red_ticks = red_seconds * ticks_per_second

    logging.info("///// Starting fixed traffic light sequence. /////")

    # -------------------------
    # Phase 1: Yellow
    # -------------------------

    if yellow_ticks > 0:
        force_traffic_light_state(
            world,
            traffic_light,
            carla.TrafficLightState.Yellow,
            "Yellow"
        )

        monitor_function(
            world,
            vehicle,
            traffic_light,
            num_ticks=yellow_ticks,
            light_log=light_log,
            action_log=action_log,
            check_interval_ticks=check_interval_ticks
        )

    # -------------------------
    # Phase 2: Red
    # -------------------------

    force_traffic_light_state(
        world,
        traffic_light,
        carla.TrafficLightState.Red,
        "Red"
    )

    monitor_function(
        world,
        vehicle,
        traffic_light,
        num_ticks=red_ticks,
        light_log=light_log,
        action_log=action_log,
        check_interval_ticks=check_interval_ticks
    )

    # -------------------------
    # Phase 3: Green
    # -------------------------

    force_traffic_light_state(
        world,
        traffic_light,
        carla.TrafficLightState.Green,
        "Green"
    )

    monitor_function(
        world,
        vehicle,
        traffic_light,
        num_ticks=green_ticks,
        light_log=light_log,
        action_log=action_log,
        check_interval_ticks=check_interval_ticks
    )

    logging.info("///// Fixed traffic light sequence complete. /////")

# -------------------------
# Traffic light monitoring
# -------------------------

def monitor_light_state(
    world,
    traffic_light,
    vehicle=None,
    light_log=None,
    num_ticks=20,
    check_interval_ticks=20
):
    """
    Monitors the traffic light state over time.

    This is useful because the scenario can:
        1. set the light color
        2. read what CARLA says the color is
        3. log that color
        4. later use that color to choose the vehicle action

    check_interval_ticks:
        Controls how often the light state is logged.
        If CARLA runs around 20 FPS, then 20 ticks is about 1 second.
    """

    if light_log is None:
        light_log = []

    current_color = None

    for tick in range(num_ticks):
        if tick % check_interval_ticks == 0:
            current_color = log_light_state(
                traffic_light,
                light_log,
                tick=tick
            )

        if vehicle is not None:
            follow_vehicle_with_spectator(
                world,
                vehicle,
                distance=12,
                height=5,
                pitch=-8
            )

        advance_simulation(world)

    return current_color, light_log


# -------------------------
# Traffic light cleanup
# -------------------------

def unfreeze_traffic_light(traffic_light):
    """
    Unfreezes a traffic light so CARLA can control it normally again.
    """

    if traffic_light is not None:
        traffic_light.freeze(False)
        logging.info(f"Traffic light {traffic_light.id} unfrozen.")