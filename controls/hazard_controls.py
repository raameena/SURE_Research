import carla
import logging
import math
import time

from controls.light_controls import get_light_color, log_light_state
from controls.actions import stop_vehicle, slow_vehicle
from config.CARLA_actors.sensors import follow_vehicle_with_spectator

# -------------------------
# Vehicle movement helper
# -------------------------

def move_vehicle_forward(vehicle, throttle=0.4):
    """
    Moves the vehicle straight forward.
    """

    vehicle.apply_control(
        carla.VehicleControl(
            throttle=throttle,
            steer=0.0,
            brake=0.0
        )
    )


# -------------------------
# Hazard actor collection
# -------------------------

def get_hazard_actors(world, hazard_types=None):
    """
    Gets hazard actors from the CARLA world.

    Supported hazard types:
        pedestrian
        bicyclist
        vehicle
    """

    if hazard_types is None:
        hazard_types = ["pedestrian"]

    hazard_actors = []
    all_actors = world.get_actors()

    for hazard_type in hazard_types:
        hazard_type = hazard_type.lower()

        if hazard_type in ["pedestrian", "walker"]:
            hazard_actors.extend(
                list(all_actors.filter("walker.pedestrian.*"))
            )

        elif hazard_type in ["bicyclist", "bicycle", "bike"]:
            bicycle_filters = [
                "vehicle.bh.crossbike",
                "vehicle.diamondback.century",
                "vehicle.gazelle.omafiets"
            ]

            for bicycle_filter in bicycle_filters:
                hazard_actors.extend(
                    list(all_actors.filter(bicycle_filter))
                )

        elif hazard_type in ["vehicle", "car"]:
            hazard_actors.extend(
                list(all_actors.filter("vehicle.*"))
            )

    return hazard_actors


# -------------------------
# Hazard geometry helpers
# -------------------------

def get_actor_position_relative_to_vehicle(vehicle, actor):
    """
    Computes where an actor is relative to the ego vehicle.

    Returns:
        forward_distance:
            how far the actor is in front of the vehicle

        side_distance:
            how far the actor is left/right from the vehicle center path
    """

    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location

    actor_location = actor.get_location()

    forward_vector = vehicle_transform.get_forward_vector()
    right_vector = vehicle_transform.get_right_vector()

    vector_to_actor = actor_location - vehicle_location

    forward_distance = (
        vector_to_actor.x * forward_vector.x +
        vector_to_actor.y * forward_vector.y +
        vector_to_actor.z * forward_vector.z
    )

    side_distance = (
        vector_to_actor.x * right_vector.x +
        vector_to_actor.y * right_vector.y +
        vector_to_actor.z * right_vector.z
    )

    return forward_distance, abs(side_distance)


def compute_time_to_collision(vehicle, actor):
    """
    Estimates time-to-collision (seconds) between the ego vehicle and an
    actor, based on their closing speed along the line between them.

    Returns None if the actor is not currently closing distance with the
    ego vehicle (moving away, or closing slower than 0.1 m/s).

    This exists because a fixed distance box (see is_actor_in_ego_safety_zone)
    assumes a slow-moving hazard like a pedestrian or bicyclist: it reacts
    at the same distance regardless of how fast the hazard is closing. A
    fast-moving vehicle hazard can cover that distance in well under a
    second, so vehicle hazards should be judged by time-to-collision
    instead of raw distance.
    """

    vehicle_location = vehicle.get_location()
    actor_location = actor.get_location()

    separation_vector = actor_location - vehicle_location

    separation_distance = math.sqrt(
        separation_vector.x ** 2 +
        separation_vector.y ** 2
    )

    if separation_distance < 1e-3:
        return 0.0

    separation_direction_x = separation_vector.x / separation_distance
    separation_direction_y = separation_vector.y / separation_distance

    vehicle_velocity = vehicle.get_velocity()
    actor_velocity = actor.get_velocity()

    relative_velocity_x = actor_velocity.x - vehicle_velocity.x
    relative_velocity_y = actor_velocity.y - vehicle_velocity.y

    # Positive closing_speed means the actor is approaching the ego vehicle.
    closing_speed = -(
        relative_velocity_x * separation_direction_x +
        relative_velocity_y * separation_direction_y
    )

    if closing_speed <= 0.1:
        return None

    return separation_distance / closing_speed


def is_actor_in_ego_safety_zone(
    vehicle,
    actor,
    forward_distance_limit=15.0,
    side_distance_limit=3.5,
    use_time_to_collision=False,
    time_to_collision_limit=3.5,
    max_check_distance=60.0
):
    """
    Checks if an actor is inside the ego vehicle safety zone.

    Default (use_time_to_collision=False):
        Static distance box: ahead of the car, close enough, within the
        car path width. Suited to slow hazards (pedestrian, bicyclist)
        that stay roughly the same distance ahead while crossing.

    use_time_to_collision=True:
        Ignores the static box and instead flags the actor as hazardous
        only if it is on a collision course within time_to_collision_limit
        seconds, per compute_time_to_collision(). Suited to fast hazards
        (e.g. a crash car) where a fixed distance threshold would either
        trigger too late (if sized for a pedestrian) or too early/constantly
        (if sized generously), regardless of whether it's actually closing.
        max_check_distance is a coarse cutoff so distant, irrelevant actors
        don't need a time-to-collision computation every tick.
    """

    forward_distance, side_distance = get_actor_position_relative_to_vehicle(
        vehicle,
        actor
    )

    if use_time_to_collision:
        straight_line_distance = math.sqrt(
            forward_distance ** 2 +
            side_distance ** 2
        )

        if straight_line_distance > max_check_distance:
            return False, forward_distance, side_distance, None

        time_to_collision = compute_time_to_collision(vehicle, actor)

        is_on_collision_course = (
            time_to_collision is not None and
            time_to_collision <= time_to_collision_limit
        )

        return is_on_collision_course, forward_distance, side_distance, time_to_collision

    actor_is_in_front = forward_distance > 0
    actor_is_close_enough = forward_distance <= forward_distance_limit
    actor_is_in_path_width = side_distance <= side_distance_limit

    return (
        actor_is_in_front and
        actor_is_close_enough and
        actor_is_in_path_width
    ), forward_distance, side_distance, None


def detect_hazard_in_path(
    world,
    vehicle,
    hazard_actors=None,
    hazard_types=None,
    forward_distance_limit=15.0,
    side_distance_limit=3.5,
    use_time_to_collision=False,
    time_to_collision_limit=3.5,
    max_check_distance=60.0
):
    """
    Detects whether a hazard is in the ego vehicle safety zone.

    If hazard_actors is provided:
        only those specific actors are checked.

    If hazard_actors is None:
        actors are collected from the world using hazard_types.

    See is_actor_in_ego_safety_zone() for what use_time_to_collision changes.
    """

    if hazard_actors is None:
        hazard_actors = get_hazard_actors(
            world,
            hazard_types=hazard_types
        )

    closest_hazard_info = None
    closest_forward_distance = float("inf")

    for actor in hazard_actors:
        if actor is None:
            continue

        if not actor.is_alive:
            continue

        is_in_zone, forward_distance, side_distance, time_to_collision = is_actor_in_ego_safety_zone(
            vehicle,
            actor,
            forward_distance_limit=forward_distance_limit,
            side_distance_limit=side_distance_limit,
            use_time_to_collision=use_time_to_collision,
            time_to_collision_limit=time_to_collision_limit,
            max_check_distance=max_check_distance
        )

        if is_in_zone and forward_distance < closest_forward_distance:
            closest_forward_distance = forward_distance

            closest_hazard_info = {
                "actor_id": actor.id,
                "actor_type": actor.type_id,
                "forward_distance": forward_distance,
                "side_distance": side_distance,
                "straight_line_distance": math.sqrt(
                    forward_distance ** 2 + side_distance ** 2
                ),
                "time_to_collision": time_to_collision,
                "timestamp": time.time()
            }

    if closest_hazard_info is not None:
        return True, closest_hazard_info

    return False, None


# -------------------------
# Hazard-aware hard controller
# -------------------------

def monitor_light_and_hazards_and_act(
    world,
    vehicle,
    traffic_light,
    num_ticks,
    light_log,
    action_log,
    check_interval_ticks=5,
    hazard_actors=None,
    hazard_types=None,
    forward_distance_limit=15.0,
    side_distance_limit=3.5,
    use_time_to_collision=False,
    time_to_collision_limit=3.5,
    max_check_distance=60.0,
    on_green_start=None,
    on_each_tick=None
):
    """
    Monitors hazards and traffic light state, then acts.

    Priority:
        1. Hazard detection
        2. Traffic light color

    This means the car will stop for a pedestrian even if the light is green.

    on_each_tick, if provided, is called once per tick (as on_each_tick(tick))
    before hazard/light logic runs, regardless of hazard or light state. This
    exists for hazard actors that need continuous per-tick control (e.g. a
    crash car steering through a scripted turn), unlike the pedestrian/
    bicyclist hazards which only need a single one-time velocity command.
    """

    last_logged_action = None
    last_hazard_detected = False
    green_callback_has_run = False

    for tick in range(num_ticks):
        if on_each_tick is not None:
            on_each_tick(tick)

        hazard_detected, hazard_info = detect_hazard_in_path(
            world,
            vehicle,
            hazard_actors=hazard_actors,
            hazard_types=hazard_types,
            forward_distance_limit=forward_distance_limit,
            side_distance_limit=side_distance_limit,
            use_time_to_collision=use_time_to_collision,
            time_to_collision_limit=time_to_collision_limit,
            max_check_distance=max_check_distance
        )

        light_color = get_light_color(traffic_light)

        if tick % check_interval_ticks == 0:
            log_light_state(
                traffic_light,
                light_log,
                tick=tick
            )

        # -------------------------
        # Priority 1: Hazard
        # -------------------------

        if hazard_detected:
            stop_vehicle(vehicle)

            if not last_hazard_detected:
                if hazard_info["time_to_collision"] is not None:
                    time_to_collision_text = f"{hazard_info['time_to_collision']:.2f}s"
                else:
                    time_to_collision_text = "n/a"

                logging.info(
                    f"{hazard_types} detected at tick {tick}: "
                    f"forward_distance={hazard_info['forward_distance']:.2f}m, "
                    f"side_distance={hazard_info['side_distance']:.2f}m, "
                    f"straight_line_distance={hazard_info['straight_line_distance']:.2f}m, "
                    f"time_to_collision={time_to_collision_text}"
                )
                logging.info("STOP CAR.")

            action_log.append(
                {
                    "tick": tick,
                    "timestamp": time.time(),
                    "priority": "hazard",
                    "action": "stop",
                    "light_color": light_color,
                    "hazard_info": hazard_info
                }
            )

            last_logged_action = "stop_for_hazard"
            last_hazard_detected = True

        # -------------------------
        # Priority 2: Traffic light
        # -------------------------

        else:
            if last_hazard_detected:
                logging.info(f"{hazard_types} cleared.")

            last_hazard_detected = False

            if light_color == "red":
                stop_vehicle(vehicle)

                if last_logged_action != "stop_for_red":
                    logging.info("Red light detected.")
                    logging.info("STOP CAR.")

                action_name = "stop"
                last_logged_action = "stop_for_red"

            elif light_color == "yellow":
                slow_vehicle(vehicle)

                if last_logged_action != "slow_for_yellow":
                    logging.info("Yellow light detected.")
                    logging.info("SLOW CAR.")

                action_name = "slow"
                last_logged_action = "slow_for_yellow"

            elif light_color == "green":
                if on_green_start is not None and not green_callback_has_run:
                    on_green_start()
                    green_callback_has_run = True

                move_vehicle_forward(vehicle)

                if last_logged_action != "go_for_green":
                    logging.info("Green light detected.")
                    logging.info("Moving car forward.")

                action_name = "go"
                last_logged_action = "go_for_green"

            else:
                stop_vehicle(vehicle)

                if last_logged_action != "stop_unknown_light":
                    logging.info("Unknown light state detected.")
                    logging.info("STOP CAR.")

                action_name = "stop"
                last_logged_action = "stop_unknown_light"

            action_log.append(
                {
                    "tick": tick,
                    "timestamp": time.time(),
                    "priority": "traffic_light",
                    "action": action_name,
                    "light_color": light_color,
                    "hazard_info": None
                }
            )

            follow_vehicle_with_spectator(
                world,
                vehicle,
                distance=12,
                height=5,
                pitch=-8
            )

        world.wait_for_tick()