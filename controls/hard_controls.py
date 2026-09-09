import logging
import time

from controls.actions import (
    drive_forward,
    stop_vehicle,
    slow_vehicle,
    hard_brake
)

from controls.light_controls import (
    get_light_color,
    log_light_state
)

from config.CARLA_actors.sensors import follow_vehicle_with_spectator


# -------------------------
# Light-based hardcoded actions
# -------------------------

def red_light_action(vehicle):
    """
    Hardcoded action for a red light.

    For now:
        red light -> stop vehicle
    """

    logging.info("RED light detected -> STOP")

    stop_vehicle(vehicle)

    return "stop"


def green_light_action(vehicle, throttle=0.5):
    """
    Hardcoded action for a green light.

    For now:
        green light -> drive forward
    """

    logging.info("GREEN light detected -> GO")

    drive_forward(
        vehicle,
        throttle=throttle
    )

    return "go"


def yellow_light_action(vehicle):
    """
    Hardcoded action for a yellow light.

    For now:
        yellow light -> slow down

    Later, this can be changed depending on distance to the light.
    """

    logging.info("YELLOW light detected -> SLOW")

    slow_vehicle(
        vehicle,
        throttle=0.2
    )

    return "slow"


def unknown_light_action(vehicle):
    """
    Hardcoded action for an unknown/off light.

    Conservative safety behavior:
        unknown light -> hard brake
    """

    logging.info("UNKNOWN light detected -> HARD BRAKE")

    hard_brake(vehicle)

    return "hard_brake"


# -------------------------
# Main hardcoded controller
# -------------------------

def run_light_based_action(vehicle, light_color):
    """
    Chooses and applies the vehicle action based on the detected light color.

    This is the rule-based controller.

    Later, this function can be replaced with an ML model controller.
    """

    if light_color == "red":
        return red_light_action(vehicle)

    if light_color == "green":
        return green_light_action(vehicle)

    if light_color == "yellow":
        return yellow_light_action(vehicle)

    return unknown_light_action(vehicle)


def observe_light_and_act(vehicle, traffic_light):
    """
    Reads the current traffic light color from CARLA,
    then applies the correct hardcoded vehicle action.

    Returns:
        light_color
        action_taken
    """

    light_color = get_light_color(traffic_light)

    logging.info(f"Observed traffic light color: {light_color}")

    action_taken = run_light_based_action(
        vehicle,
        light_color
    )

    return light_color, action_taken


# -------------------------
# Monitoring loop
# -------------------------

def monitor_light_and_act(
    world,
    vehicle,
    traffic_light,
    num_ticks,
    light_log=None,
    action_log=None,
    check_interval_ticks=20
):
    """
    Monitors the traffic light and applies actions over time.

    Every check_interval_ticks:
        1. Read the traffic light color from CARLA
        2. Log the light color
        3. Apply the correct hardcoded vehicle action
        4. Store the action taken

    If CARLA is running around 20 FPS:
        check_interval_ticks=20 is about once per second.
    """

    if light_log is None:
        light_log = []

    if action_log is None:
        action_log = []

    latest_light_color = None
    latest_action = None

    for tick in range(num_ticks):
        if tick % check_interval_ticks == 0:
            latest_light_color = log_light_state(
                traffic_light,
                light_log,
                tick=tick
            )

            latest_action = run_light_based_action(
                vehicle,
                latest_light_color
            )

            action_entry = {
                "tick": tick,
                "timestamp": time.time(),
                "traffic_light_id": traffic_light.id,
                "light_color": latest_light_color,
                "action_taken": latest_action
            }

            action_log.append(action_entry)

            logging.info(
                f"Update: light={latest_light_color}, action={latest_action}"
            )

        follow_vehicle_with_spectator(
            world,
            vehicle,
            distance=12,
            height=5,
            pitch=-8
        )

        world.wait_for_tick()

    return light_log, action_log