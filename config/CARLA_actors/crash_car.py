import carla
import logging
import math


# -------------------------
# Helper
# -------------------------

def hold_crash_car_still(crash_car):
    """
    Fully stops and holds the crash car in place before the scenario starts.
    """

    if crash_car is None:
        return

    crash_car.set_target_velocity(
        carla.Vector3D(0.0, 0.0, 0.0)
    )

    crash_car.set_target_angular_velocity(
        carla.Vector3D(0.0, 0.0, 0.0)
    )

    crash_car.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True
        )
    )

    logging.debug("Crash car held still.")


def get_speed_mps(actor):
    """
    Returns an actor's current speed in meters per second.
    """

    velocity = actor.get_velocity()

    return math.sqrt(
        velocity.x ** 2 +
        velocity.y ** 2 +
        velocity.z ** 2
    )


def find_ego_junction_waypoint(carla_map, vehicle, search_distance=40.0, step=2.0):
    """
    Walks forward from the ego vehicle's current waypoint until it reaches
    the junction the ego vehicle is approaching.

    Returns the first waypoint inside that junction, or None if no junction
    is found within search_distance.
    """

    vehicle_location = vehicle.get_location()

    waypoint = carla_map.get_waypoint(
        vehicle_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if waypoint is None:
        return None

    distance_traveled = 0.0

    while not waypoint.is_junction and distance_traveled < search_distance:
        next_waypoints = waypoint.next(step)

        if not next_waypoints:
            return None

        waypoint = next_waypoints[0]
        distance_traveled += step

    if not waypoint.is_junction:
        return None

    return waypoint


def find_ego_straight_connector(vehicle, junction_waypoint, match_distance=6.0):
    """
    Finds the junction connector (entry, exit) representing the ego
    vehicle's own straight-through path: the connector whose entry sits at
    the ego vehicle's junction entry point and whose exit continues most
    closely in the ego vehicle's current forward direction.

    Returns None if no connector is found near the ego's junction entry.
    """

    ego_forward = vehicle.get_transform().get_forward_vector()

    junction = junction_waypoint.get_junction()
    lane_pairs = junction.get_waypoints(carla.LaneType.Driving)

    ego_connectors = [
        (entry_waypoint, exit_waypoint)
        for entry_waypoint, exit_waypoint in lane_pairs
        if entry_waypoint.transform.location.distance(
            junction_waypoint.transform.location
        ) <= match_distance
    ]

    if not ego_connectors:
        return None

    def straight_alignment(pair):
        entry_waypoint, exit_waypoint = pair

        direction = exit_waypoint.transform.location - entry_waypoint.transform.location
        length = math.sqrt(direction.x ** 2 + direction.y ** 2)

        if length < 1e-3:
            return -1.0

        return (
            direction.x * ego_forward.x +
            direction.y * ego_forward.y
        ) / length

    return max(ego_connectors, key=straight_alignment)


def find_crash_car_turn_connector(
    vehicle,
    junction_waypoint,
    side_offset,
    exit_alignment_threshold=0.9,
    exit_merge_tolerance=3.0,
    entry_perpendicular_threshold=0.5
):
    """
    Finds the junction connector road representing an unprotected left turn
    from a perpendicular street on the ego vehicle's left into the ego
    vehicle's own lane: same location and same direction of travel as the
    ego's own straight-through exit.

    This replaces an earlier "walk to the leftmost lane via get_left_lane()"
    approach. On the live map that was checked against, the leftmost lane
    of the nearest perpendicular-left approach (by lateral offset) turned
    out to be a lane whose only junction connectors led into the *opposing*
    direction of the ego's road (a "continue straight across" maneuver),
    while the true left-turn-into-ego's-lane connector originated from a
    different, adjacent lane a few meters away. Matching directly against
    the maneuver (an exit that lands at the ego's own exit point, heading
    the ego's own direction) sidesteps that: it finds the correct entry
    lane by construction instead of guessing "leftmost".

    side_offset is used only to break ties if more than one such connector
    exists (e.g. a wide intersection with multiple crossing roads): the
    candidate whose entry lateral distance from the ego vehicle is closest
    to side_offset is preferred. Negative side_offset = left of ego.

    Returns (connector_entry_waypoint, connector_exit_waypoint), or None if
    no matching connector is found.
    """

    ego_transform = vehicle.get_transform()
    ego_location = ego_transform.location
    ego_forward = ego_transform.get_forward_vector()
    ego_right = ego_transform.get_right_vector()

    ego_straight_connector = find_ego_straight_connector(
        vehicle,
        junction_waypoint
    )

    if ego_straight_connector is None:
        return None

    ego_straight_entry, ego_straight_exit = ego_straight_connector

    junction = junction_waypoint.get_junction()
    lane_pairs = junction.get_waypoints(carla.LaneType.Driving)

    candidates = []

    for entry_waypoint, exit_waypoint in lane_pairs:
        # Skip the ego's own connector.
        if (
            entry_waypoint.road_id == ego_straight_entry.road_id and
            entry_waypoint.lane_id == ego_straight_entry.lane_id
        ):
            continue

        exit_forward = exit_waypoint.transform.get_forward_vector()

        exit_alignment = (
            exit_forward.x * ego_forward.x +
            exit_forward.y * ego_forward.y
        )

        # The exit must head the same direction as the ego vehicle...
        if exit_alignment < exit_alignment_threshold:
            continue

        exit_distance = exit_waypoint.transform.location.distance(
            ego_straight_exit.transform.location
        )

        # ...and land at (not just near) the ego's own exit point, so the
        # crash car actually merges into the ego's lane rather than some
        # other parallel lane.
        if exit_distance > exit_merge_tolerance:
            continue

        entry_forward = entry_waypoint.transform.get_forward_vector()

        entry_alignment = abs(
            entry_forward.x * ego_forward.x +
            entry_forward.y * ego_forward.y
        )

        # The entry should be roughly perpendicular to the ego's road, i.e.
        # a real turn rather than a merge from a parallel lane.
        if entry_alignment > entry_perpendicular_threshold:
            continue

        to_entry = entry_waypoint.transform.location - ego_location

        entry_side_distance = (
            to_entry.x * ego_right.x +
            to_entry.y * ego_right.y
        )

        # Only keep approaches on the ego vehicle's left (negative side distance).
        if entry_side_distance >= 0:
            continue

        candidates.append((entry_waypoint, exit_waypoint, entry_side_distance))

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: abs(candidate[2] - side_offset))

    best_entry_waypoint, best_exit_waypoint, _ = candidates[0]

    return best_entry_waypoint, best_exit_waypoint


def compute_steer_toward_target(vehicle_transform, target_location, max_steer=1.0, steer_gain=1.6):
    """
    Computes a steer value (-1.0 to 1.0) that turns the vehicle toward
    target_location, using the vehicle's own forward/right vectors so the
    sign convention always matches carla.VehicleControl.steer.
    """

    vehicle_location = vehicle_transform.location
    forward_vector = vehicle_transform.get_forward_vector()
    right_vector = vehicle_transform.get_right_vector()

    to_target = target_location - vehicle_location

    longitudinal = (
        to_target.x * forward_vector.x +
        to_target.y * forward_vector.y
    )

    lateral = (
        to_target.x * right_vector.x +
        to_target.y * right_vector.y
    )

    heading_error = math.atan2(lateral, longitudinal)

    steer = heading_error * steer_gain

    return max(-max_steer, min(max_steer, steer))


# -------------------------
# Crash car spawning
# -------------------------

CRASH_CAR_BLUEPRINT_IDS = [
    "vehicle.dodge.charger_2020",
    "vehicle.mercedes.coupe_2020",
    "vehicle.audi.tt",
    "vehicle.mini.cooper_s_2021",
    "vehicle.chevrolet.impala"
]


def spawn_crash_car_relative_to_vehicle(
    world,
    bp_lib,
    vehicle,
    forward_distance=20.0,
    side_offset=-14.0
):
    """
    Spawns the crash car on the street perpendicular to the ego vehicle's
    current road, on the ego vehicle's left, in that street's left lane,
    facing into the intersection.

    Unlike pedestrian/bicyclist spawning, this snaps to CARLA's actual road
    waypoints (rather than a blind forward/right offset from the ego
    transform) because generate_left_turn_path() must later follow that
    same lane through the junction into the ego's lane.

    forward_distance:
        How far back from the junction entry the crash car starts, along
        the cross-street lane (gives it room to accelerate before entering
        the intersection).

    side_offset:
        Used only to break ties if more than one perpendicular left-side
        approach exists at the junction. Negative = left side of ego.

    This still tries multiple nearby locations (several backing distances
    and vertical offsets) before giving up, for the same robustness reasons
    as the pedestrian/bicyclist spawn functions.
    """

    crash_car_bp = None

    for blueprint_id in CRASH_CAR_BLUEPRINT_IDS:
        matches = bp_lib.filter(blueprint_id)

        if matches:
            crash_car_bp = matches[0]
            break

    if crash_car_bp is None:
        four_wheeled_vehicles = [
            bp for bp in bp_lib.filter("vehicle.*")
            if bp.has_attribute("number_of_wheels")
            and int(bp.get_attribute("number_of_wheels")) == 4
        ]

        if not four_wheeled_vehicles:
            raise RuntimeError("No four-wheeled vehicle blueprints found.")

        crash_car_bp = four_wheeled_vehicles[0]

    carla_map = world.get_map()

    junction_waypoint = find_ego_junction_waypoint(carla_map, vehicle)

    if junction_waypoint is None:
        raise RuntimeError(
            "Could not find a junction ahead of the ego vehicle to spawn "
            "the crash car at. Check SPAWN_POINT_INDEX is near a real "
            "intersection."
        )

    connector = find_crash_car_turn_connector(
        vehicle,
        junction_waypoint,
        side_offset
    )

    if connector is None:
        raise RuntimeError(
            "Could not find a junction connector that starts on a "
            "perpendicular street on the ego vehicle's left and ends in "
            "the ego vehicle's own lane, heading the ego's own direction. "
            "This intersection may not support a crash car T-bone via "
            "unprotected left turn from the left."
        )

    entry_waypoint, _connector_exit_waypoint = connector

    # Try the requested backing distance first, then nearby alternatives.
    distance_options = [
        forward_distance,
        forward_distance + 5.0,
        forward_distance - 5.0,
        forward_distance + 10.0,
        forward_distance - 10.0
    ]

    z_options = [
        0.3,
        0.6,
        1.0
    ]

    for current_distance in distance_options:
        if current_distance <= 0:
            continue

        previous_waypoints = entry_waypoint.previous(current_distance)

        if not previous_waypoints:
            continue

        spawn_waypoint = previous_waypoints[0]
        base_transform = spawn_waypoint.transform

        for z_offset in z_options:
            spawn_location = carla.Location(
                x=base_transform.location.x,
                y=base_transform.location.y,
                z=base_transform.location.z + z_offset
            )

            # Rotation comes from the real lane heading rather than a
            # hardcoded +90 degrees, since the actual cross-street may not
            # meet the ego's road at a perfect right angle.
            spawn_transform = carla.Transform(
                spawn_location,
                base_transform.rotation
            )

            logging.info(
                f"Trying crash car spawn at road_id={spawn_waypoint.road_id}, "
                f"lane_id={spawn_waypoint.lane_id}, "
                f"backing_distance={current_distance}, "
                f"z_offset={z_offset}"
            )

            crash_car = world.try_spawn_actor(
                crash_car_bp,
                spawn_transform
            )

            if crash_car is not None:
                hold_crash_car_still(crash_car)
                world.wait_for_tick()

                logging.info(
                    f"Crash car spawned: ID {crash_car.id} at "
                    f"road_id={spawn_waypoint.road_id}, "
                    f"lane_id={spawn_waypoint.lane_id}, "
                    f"backing_distance={current_distance}"
                )

                return crash_car

    raise RuntimeError(
        "Crash car failed to spawn after trying multiple nearby locations. "
        "Try changing CRASH_CAR_FORWARD_DISTANCE or CRASH_CAR_SIDE_OFFSET."
    )


# -------------------------
# Crash car path
# -------------------------

def generate_left_turn_path(
    crash_car,
    vehicle,
    side_offset=-14.0,
    waypoint_step=2.0,
    lane_extension_distance=6.0
):
    """
    Precomputes a fixed, deterministic sequence of waypoints describing the
    crash car's unprotected left turn: from its spawn lane, through the
    junction, into the ego vehicle's lane, ending parallel to and heading
    the same direction as the ego vehicle (a merge into the ego's lane,
    not a perpendicular crossing).

    This follows CARLA's actual road/lane graph (rather than a manually
    computed geometric arc) so the crash car tracks real lane curvature
    through the junction instead of a hand-picked curve.

    side_offset must match the value passed to
    spawn_crash_car_relative_to_vehicle() so this finds the same junction
    connector the crash car was actually spawned on (see
    find_crash_car_turn_connector() for why matching is done by maneuver
    rather than by road/lane ID).

    Returns a list of carla.Waypoint objects. Raises RuntimeError if this
    junction has no left-turn connector from the crash car's entry lane
    into the ego vehicle's own lane/direction.
    """

    carla_map = crash_car.get_world().get_map()

    crash_car_waypoint = carla_map.get_waypoint(
        crash_car.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    # -------------------------
    # Walk forward from the crash car's spawn lane to the junction boundary.
    # -------------------------

    path_waypoints = [crash_car_waypoint]
    active_waypoint = crash_car_waypoint

    while not active_waypoint.is_junction:
        next_waypoints = active_waypoint.next(waypoint_step)

        if not next_waypoints:
            raise RuntimeError(
                "Crash car's spawn lane never reaches a junction. Check "
                "that the crash car was spawned facing into the "
                "intersection."
            )

        active_waypoint = next_waypoints[0]
        path_waypoints.append(active_waypoint)

    # -------------------------
    # Find the same left-turn junction connector the crash car was spawned
    # on (see find_crash_car_turn_connector() for why this can't be done
    # by matching road_id/lane_id directly).
    # -------------------------

    ego_junction_waypoint = find_ego_junction_waypoint(carla_map, vehicle)

    if ego_junction_waypoint is None:
        raise RuntimeError(
            "Could not find the ego vehicle's own junction entry point "
            "while generating the crash car's turn path."
        )

    connector = find_crash_car_turn_connector(
        vehicle,
        ego_junction_waypoint,
        side_offset
    )

    if connector is None:
        raise RuntimeError(
            "Could not find a left-turn connector from the crash car's "
            "entry lane into the ego vehicle's own lane/direction. This "
            "intersection may not support that maneuver, or side_offset "
            "doesn't match the value used at spawn time."
        )

    matching_entry_waypoint, matching_exit_waypoint = connector

    logging.info(
        f"Crash car connector: entry road_id={matching_entry_waypoint.road_id} "
        f"lane_id={matching_entry_waypoint.lane_id} -> "
        f"exit road_id={matching_exit_waypoint.road_id} "
        f"lane_id={matching_exit_waypoint.lane_id}, heading same direction "
        f"as ego at the ego's own junction exit point."
    )

    # -------------------------
    # Sample along the matched connector road, from its precise entry to
    # its precise exit.
    # -------------------------

    path_waypoints[-1] = matching_entry_waypoint
    active_waypoint = matching_entry_waypoint
    max_junction_steps = 200  # safety bound against unexpected loops

    for _ in range(max_junction_steps):
        remaining_distance = active_waypoint.transform.location.distance(
            matching_exit_waypoint.transform.location
        )

        if remaining_distance <= waypoint_step:
            break

        next_waypoints = active_waypoint.next(waypoint_step)

        if not next_waypoints:
            break

        # If the connector road branches, follow the branch that stays
        # closest to our chosen exit.
        active_waypoint = min(
            next_waypoints,
            key=lambda wp: wp.transform.location.distance(
                matching_exit_waypoint.transform.location
            )
        )

        path_waypoints.append(active_waypoint)

    path_waypoints.append(matching_exit_waypoint)

    # -------------------------
    # Extend a short distance past the junction exit so the path actually
    # terminates inside the ego's lane, not right at the junction boundary.
    # -------------------------

    active_waypoint = matching_exit_waypoint
    distance_extended = 0.0

    while distance_extended < lane_extension_distance:
        next_waypoints = active_waypoint.next(waypoint_step)

        if not next_waypoints:
            break

        active_waypoint = next_waypoints[0]
        path_waypoints.append(active_waypoint)
        distance_extended += waypoint_step

    logging.info(
        f"Crash car left-turn path generated with {len(path_waypoints)} "
        f"waypoints, ending at road_id={path_waypoints[-1].road_id}, "
        f"lane_id={path_waypoints[-1].lane_id}."
    )

    return path_waypoints


# -------------------------
# Crash car movement
# -------------------------

def start_crash_car_maneuver(crash_car, path, speed=14.0):
    """
    Begins the crash car's scripted left-turn maneuver at high speed.

    This only releases the handbrake and sets an initial target velocity
    toward the first waypoint on the path; update_crash_car_control() must
    be called every tick after this to steer the car along the rest of the
    path, since a single one-time command (like the pedestrian/bicyclist
    use) can't express a continuously curving turn.
    """

    if crash_car is None:
        raise ValueError("crash_car is None. Cannot start maneuver.")

    if not path:
        raise ValueError("path is empty. Cannot start maneuver.")

    crash_car.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=0.0,
            hand_brake=False
        )
    )

    first_waypoint_direction = path[0].transform.get_forward_vector()

    target_velocity = carla.Vector3D(
        x=first_waypoint_direction.x * speed,
        y=first_waypoint_direction.y * speed,
        z=0.0
    )

    crash_car.set_target_velocity(target_velocity)

    logging.info(f"Crash car maneuver started at {speed} m/s.")


def update_crash_car_control(
    crash_car,
    path,
    current_tick,
    target_speed=14.0,
    lookahead_distance=5.0,
    steer_gain=1.6
):
    """
    Per-tick steering/throttle update that drives the crash car along path.

    Meant to be called every tick from the scenario's monitor loop once
    start_crash_car_maneuver() has been called. Finds the nearest waypoint
    on the path to the crash car's current position, looks ahead a short
    distance along the path for a steering target, and applies throttle to
    hold target_speed.

    current_tick is accepted for logging/diagnostics; the actual steering
    decision is derived from the crash car's live position on path rather
    than tick count, so small physics variations between runs don't throw
    the car off its route.
    """

    if crash_car is None or not path:
        return

    crash_car_transform = crash_car.get_transform()
    crash_car_location = crash_car_transform.location

    nearest_index = min(
        range(len(path)),
        key=lambda i: crash_car_location.distance(path[i].transform.location)
    )

    target_index = nearest_index
    accumulated_distance = 0.0

    while target_index < len(path) - 1 and accumulated_distance < lookahead_distance:
        accumulated_distance += path[target_index].transform.location.distance(
            path[target_index + 1].transform.location
        )
        target_index += 1

    target_location = path[target_index].transform.location

    steer = compute_steer_toward_target(
        crash_car_transform,
        target_location,
        steer_gain=steer_gain
    )

    current_speed = get_speed_mps(crash_car)

    if current_speed < target_speed:
        throttle = 1.0
        brake = 0.0
    else:
        # Coast rather than brake mid-turn so the scripted maneuver timing
        # isn't disrupted.
        throttle = 0.0
        brake = 0.0

    crash_car.apply_control(
        carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False
        )
    )

    logging.debug(
        f"[tick {current_tick}] crash car control: steer={steer:.2f}, "
        f"speed={current_speed:.1f} m/s, "
        f"waypoint {target_index}/{len(path) - 1}"
    )


def stop_crash_car(crash_car):
    """
    Hard brakes and holds the crash car in place.
    """

    if crash_car is None:
        return

    crash_car.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True
        )
    )

    logging.info("Crash car stopped.")
