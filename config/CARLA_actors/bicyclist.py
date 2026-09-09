import carla
import logging
import math


# -------------------------
# Helper
# -------------------------

def yaw_from_vector(vector):
    """
    Converts a CARLA direction vector into a yaw angle in degrees.
    """

    return math.degrees(
        math.atan2(vector.y, vector.x)
    )


def hold_bicyclist_still(bicyclist):
    """
    Holds the bicyclist still before the scenario begins.
    """

    if bicyclist is None:
        return

    bicyclist.set_autopilot(False)

    bicyclist.set_target_velocity(
        carla.Vector3D(0.0, 0.0, 0.0)
    )

    bicyclist.set_target_angular_velocity(
        carla.Vector3D(0.0, 0.0, 0.0)
    )

    bicyclist.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True
        )
    )

    logging.info("Bicyclist held still.")


# -------------------------
# Bicyclist spawning
# -------------------------

def spawn_bicyclist_relative_to_vehicle(
    world,
    bp_lib,
    vehicle,
    forward_distance=12.0,
    side_offset=-6.0,
    crossing_direction="left_to_right"
):
    """
    Spawns a bicyclist near the ego vehicle.

    forward_distance:
        How far in front of the ego vehicle the bicyclist starts.

    side_offset:
        Negative = left side of ego vehicle
        Positive = right side of ego vehicle

    crossing_direction:
        left_to_right
        right_to_left
    """

    bicycle_blueprint_ids = [
        "vehicle.bh.crossbike",
        "vehicle.diamondback.century",
        "vehicle.gazelle.omafiets"
    ]

    bicycle_bp = None

    for blueprint_id in bicycle_blueprint_ids:
        matches = bp_lib.filter(blueprint_id)

        if matches:
            bicycle_bp = matches[0]
            break

    if bicycle_bp is None:
        raise RuntimeError("No bicycle blueprints found.")

    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location

    forward_vector = vehicle_transform.get_forward_vector()
    right_vector = vehicle_transform.get_right_vector()

    if crossing_direction == "left_to_right":
        crossing_vector = right_vector

    elif crossing_direction == "right_to_left":
        crossing_vector = carla.Vector3D(
            x=-right_vector.x,
            y=-right_vector.y,
            z=0.0
        )

    else:
        raise ValueError(
            "crossing_direction must be 'left_to_right' or 'right_to_left'."
        )

    crossing_yaw = yaw_from_vector(crossing_vector)

    # Try requested location first, then nearby backup spots.
    forward_options = [
        forward_distance,
        forward_distance + 2.0,
        forward_distance - 2.0,
        forward_distance + 4.0,
        forward_distance - 4.0
    ]

    side_options = [
        side_offset,
        -side_offset,
        side_offset + 1.0,
        side_offset - 1.0,
        -side_offset + 1.0,
        -side_offset - 1.0
    ]

    z_options = [
        0.3,
        0.6,
        1.0
    ]

    for current_forward_distance in forward_options:
        for current_side_offset in side_options:
            for z_offset in z_options:
                spawn_location = carla.Location(
                    x=vehicle_location.x
                    + forward_vector.x * current_forward_distance
                    + right_vector.x * current_side_offset,

                    y=vehicle_location.y
                    + forward_vector.y * current_forward_distance
                    + right_vector.y * current_side_offset,

                    z=vehicle_location.z + z_offset
                )

                spawn_rotation = carla.Rotation(
                    pitch=0.0,
                    yaw=crossing_yaw,
                    roll=0.0
                )

                spawn_transform = carla.Transform(
                    spawn_location,
                    spawn_rotation
                )

                logging.info(
                    f"Trying bicyclist spawn at "
                    f"forward={current_forward_distance}, "
                    f"side={current_side_offset}, "
                    f"z_offset={z_offset}"
                )

                bicyclist = world.try_spawn_actor(
                    bicycle_bp,
                    spawn_transform
                )

                if bicyclist is not None:
                    hold_bicyclist_still(bicyclist)

                    logging.info(
                        f"Bicyclist spawned: ID {bicyclist.id} "
                        f"at forward={current_forward_distance}, "
                        f"side={current_side_offset}"
                    )

                    return bicyclist

    raise RuntimeError(
        "Bicyclist failed to spawn after trying multiple nearby locations. "
        "Try changing BICYCLIST_FORWARD_DISTANCE or BICYCLIST_SIDE_OFFSET."
    )


# -------------------------
# Bicyclist movement
# -------------------------

def start_bicyclist_crossing(
    bicyclist,
    vehicle,
    speed=6.0,
    crossing_direction="left_to_right"
):
    """
    Starts moving the bicyclist across the ego vehicle path.

    speed:
        Target crossing speed in meters per second.
        6.0 m/s is about 13.4 mph.
    """

    if bicyclist is None:
        raise ValueError("bicyclist is None. Cannot start crossing.")

    vehicle_transform = vehicle.get_transform()
    right_vector = vehicle_transform.get_right_vector()

    if crossing_direction == "left_to_right":
        direction = right_vector

    elif crossing_direction == "right_to_left":
        direction = carla.Vector3D(
            x=-right_vector.x,
            y=-right_vector.y,
            z=0.0
        )

    else:
        raise ValueError(
            "crossing_direction must be 'left_to_right' or 'right_to_left'."
        )

    # Release the bicyclist before applying target velocity.
    bicyclist.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=0.0,
            hand_brake=False
        )
    )

    target_velocity = carla.Vector3D(
        x=direction.x * speed,
        y=direction.y * speed,
        z=0.0
    )

    bicyclist.set_target_velocity(target_velocity)

    logging.info(f"Bicyclist crossing started at {speed} m/s.")


def stop_bicyclist(bicyclist):
    """
    Stops the bicyclist.
    """

    if bicyclist is None:
        return

    bicyclist.set_target_velocity(
        carla.Vector3D(0.0, 0.0, 0.0)
    )

    bicyclist.set_target_angular_velocity(
        carla.Vector3D(0.0, 0.0, 0.0)
    )

    bicyclist.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True
        )
    )

    logging.info("Bicyclist stopped.")