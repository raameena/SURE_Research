import carla
import logging


# -------------------------
# Helper
# -------------------------

def hold_pedestrian_still(pedestrian):
    """
    Fully stops and holds the pedestrian in place before the scenario starts.
    """

    if pedestrian is None:
        return

    walker_control = carla.WalkerControl()
    walker_control.direction = carla.Vector3D(0.0, 0.0, 0.0)
    walker_control.speed = 0.0
    walker_control.jump = False

    pedestrian.apply_control(walker_control)

    logging.debug("Pedestrian held still.")


# -------------------------
# Pedestrian spawning
# -------------------------

def spawn_pedestrian_relative_to_vehicle(
    world,
    bp_lib,
    vehicle,
    forward_distance=12.0,
    side_offset=-5.0
):
    """
    Spawns a pedestrian near the ego vehicle.

    This version tries multiple nearby spawn locations because CARLA may reject
    a pedestrian spawn if the exact location is blocked or invalid.
    """

    walker_blueprints = bp_lib.filter("walker.pedestrian.*")

    if not walker_blueprints:
        raise RuntimeError("No pedestrian blueprints found.")

    pedestrian_bp = walker_blueprints[0]

    if pedestrian_bp.has_attribute("is_invincible"):
        pedestrian_bp.set_attribute("is_invincible", "false")

    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location

    forward_vector = vehicle_transform.get_forward_vector()
    right_vector = vehicle_transform.get_right_vector()

    # Try the requested location first, then nearby alternatives.
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
        0.5,
        1.0,
        1.5
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

                spawn_transform = carla.Transform(
                    spawn_location,
                    vehicle_transform.rotation
                )

                logging.info(
                    f"Trying pedestrian spawn at "
                    f"forward={current_forward_distance}, "
                    f"side={current_side_offset}, "
                    f"z_offset={z_offset}"
                )

                pedestrian = world.try_spawn_actor(
                    pedestrian_bp,
                    spawn_transform
                )

                if pedestrian is not None:
                    hold_pedestrian_still(pedestrian)
                    world.wait_for_tick()

                    logging.info(
                        f"Pedestrian spawned: ID {pedestrian.id} "
                        f"at forward={current_forward_distance}, "
                        f"side={current_side_offset}"
                    )

                    return pedestrian

    raise RuntimeError(
        "Pedestrian failed to spawn after trying multiple nearby locations. "
        "Try changing PEDESTRIAN_FORWARD_DISTANCE or PEDESTRIAN_SIDE_OFFSET."
    )


# -------------------------
# Pedestrian movement
# -------------------------

def start_pedestrian_crossing(
    pedestrian,
    vehicle,
    speed=1.4,
    crossing_direction="left_to_right"
):
    """
    Starts moving the pedestrian across the ego vehicle path.

    crossing_direction:
        left_to_right
        right_to_left
    """

    if pedestrian is None:
        raise ValueError("pedestrian is None. Cannot start crossing.")

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

    walker_control = carla.WalkerControl()
    walker_control.direction = direction
    walker_control.speed = speed
    walker_control.jump = False

    pedestrian.apply_control(walker_control)

    logging.info("Pedestrian crossing started.")


def stop_pedestrian(pedestrian):
    """
    Stops the pedestrian.
    """

    if pedestrian is None:
        return

    walker_control = carla.WalkerControl()
    walker_control.speed = 0.0
    walker_control.jump = False

    pedestrian.apply_control(walker_control)

    logging.info("Pedestrian stopped.")
