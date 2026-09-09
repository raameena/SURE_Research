# connect to carla
# get map 
# spawn
# run folder
# gps
# clean up actors

import carla
import random
import logging
import os
import json


# -------------------------
# CARLA connection / map setup
# -------------------------

def connect_to_carla(host="localhost", port=2000, timeout=10.0, reload_world=False):
    """
    Connects to the CARLA simulator.

    Args:
        host: CARLA host address
        port: CARLA server port
        timeout: connection timeout
        reload_world: if True, reloads the current CARLA world before returning it

    Returns:
        client: CARLA client object
        world: current CARLA world
        bp_lib: CARLA blueprint library
    """
    client = carla.Client(host, port)
    client.set_timeout(timeout)

    if reload_world:
        logging.info("Reloading CARLA world for clean scenario start...")
        world = client.reload_world()
    else:
        world = client.get_world()

    bp_lib = world.get_blueprint_library()

    logging.debug(f"Connected to CARLA world: {world.get_map().name}")

    return client, world, bp_lib


# -------------------------
# Synchronous mode
# -------------------------

def enable_synchronous_mode(world, fixed_delta_seconds=0.05):
    """
    Switches the world into synchronous mode with a fixed timestep.

    In asynchronous mode (CARLA's default) the server free-runs on its own
    real-time clock: whatever control was last applied to a vehicle keeps
    acting for however long the client takes to decide on the next one. For
    a slow per-tick decision (e.g. model inference), that turns a small
    per-step steer value into a much larger actual turn, and desyncs any
    tick-counted timers (like a fixed traffic light phase) from the
    vehicle's actual experienced simulation time.

    Some agents also bake in a fixed-cadence assumption -- e.g. PCLA's
    transfuserv4 agent hardcodes config.carla_frame_rate = 1/20 for its
    Kalman filter and sensor tick rate -- and only produce correct behavior
    if the world actually advances in fixed_delta_seconds steps, which only
    synchronous mode guarantees.

    Returns the previous carla.WorldSettings so they can be restored with
    restore_world_settings() once the scenario is done.
    """
    previous_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)

    world.tick()

    logging.debug(
        f"Synchronous mode enabled (fixed_delta_seconds={fixed_delta_seconds})."
    )

    return previous_settings


def restore_world_settings(world, previous_settings):
    """
    Restores world settings captured by enable_synchronous_mode().

    Important for scenarios that reuse a long-lived CARLA server process:
    leaving synchronous_mode on afterward would stall any later script that
    only calls world.wait_for_tick() (which never fires unless something
    calls world.tick()).
    """
    if previous_settings is None:
        return

    world.apply_settings(previous_settings)

    logging.debug("World settings restored.")


def advance_simulation(world, synchronous=None):
    """
    Advances the simulation by exactly one frame, working correctly whether
    the world is in synchronous or asynchronous mode.

    In synchronous mode, nothing moves until the client explicitly calls
    world.tick() -- world.wait_for_tick() alone blocks forever. In
    asynchronous mode the server free-runs on its own clock and
    world.wait_for_tick() just waits for (and returns) whichever tick
    happens next. Shared control code should call this instead of either
    one directly so it behaves correctly in both modes.

    Pass synchronous explicitly (checked once by the caller) to avoid an
    extra get_settings() round trip on every iteration of a hot loop.
    """
    if synchronous is None:
        synchronous = world.get_settings().synchronous_mode

    if synchronous:
        return world.tick()

    return world.wait_for_tick()


def get_current_map(world):
    """
    Gets the current CARLA map.
    """
    carla_map = world.get_map()

    logging.info(f"Current map: {carla_map.name}")

    return carla_map


def get_spawn_points(world):
    """
    Gets all available spawn points from the current map.
    """
    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points found on this map.")

    logging.debug(f"Found {len(spawn_points)} spawn points.")

    return spawn_points


# -------------------------
# Output folder setup
# -------------------------

def create_run_folder(
    base_folder="output",
    control_type=None,
    scenario_group=None,
    scenario_name=None,
    use_run_folder=False
):
    """
    Creates output folders for a scenario.

    Every scenario in this codebase calls this the same way now --
    control_type set to the track ("model" or "hard_controls", matching the
    scenarios/model/ vs scenarios/hard_controls/ split), scenario_group left
    unset, scenario_name matching the scenario's own file/function name, and
    use_run_folder=True -- so output mirrors the scenarios/ folder layout
    and every run gets its own numbered folder instead of overwriting the
    last one:

        control_type="model"
        scenario_name="simple_stop_go"
        use_run_folder=True

    Creates:
        output/
            model/
                simple_stop_go/
                    run_1/
                        images/
                        lidar_pointclouds/
                        gps_log.json
                    run_2/
                        ...

    scenario_group still exists for anything that needs a third path
    segment between control_type and scenario_name, but nothing in this
    codebase currently uses it -- the old output/<model_name>/
    <intersection_name>/<scenario_name>/... scheme (keying folders on
    per-run details like which model/intersection was tested) caused folder
    sprawl and was dropped in favor of this flatter, consistent layout;
    log which model/intersection a run used inside the run's own log file
    instead of encoding it in the path.

    Returns:
        run_folder
        images_folder
        lidar_folder
        gps_log_path
    """

    os.makedirs(base_folder, exist_ok=True)

    # -------------------------
    # Build scenario output path
    # -------------------------

    path_parts = [
        part for part in (control_type, scenario_group, scenario_name)
        if part is not None
    ]

    if path_parts:
        scenario_folder = os.path.join(base_folder, *path_parts)

    else:
        # Fallback for older scripts that call create_run_folder()
        scenario_folder = base_folder
        use_run_folder = True

    os.makedirs(scenario_folder, exist_ok=True)

    # -------------------------
    # Optional run_# folder
    # -------------------------

    if use_run_folder:
        existing_run_numbers = []

        for entry_name in os.listdir(scenario_folder):
            if not entry_name.startswith("run_"):
                continue

            if not os.path.isdir(os.path.join(scenario_folder, entry_name)):
                continue

            try:
                existing_run_numbers.append(int(entry_name[len("run_"):]))
            except ValueError:
                continue

        next_run_number = max(existing_run_numbers, default=0) + 1

        run_folder = os.path.join(scenario_folder, f"run_{next_run_number}")
        os.makedirs(run_folder)

    else:
        run_folder = scenario_folder
        os.makedirs(run_folder, exist_ok=True)

    # -------------------------
    # Sensor output folders/files
    # -------------------------

    images_folder = os.path.join(run_folder, "images")
    lidar_folder = os.path.join(run_folder, "lidar_pointclouds")
    gps_log_path = os.path.join(run_folder, "gps_log.json")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(lidar_folder, exist_ok=True)

    logging.debug(f"Scenario output folder: {scenario_folder}")

    return run_folder, images_folder, lidar_folder, gps_log_path

# -------------------------
# Vehicle spawning
# -------------------------

def spawn_ego_vehicle(
    world,
    bp_lib,
    vehicle_filter="vehicle.tesla.model3",
    spawn_point=None
):
    """
    Spawns the ego vehicle.

    If spawn_point is not provided, a random spawn point is used.

    Returns:
        vehicle actor
    """
    vehicle_bp = bp_lib.filter(vehicle_filter)[0]

    if spawn_point is None:
        spawn_points = get_spawn_points(world)
        spawn_point = random.choice(spawn_points)

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    if vehicle is None:
        raise RuntimeError("Vehicle failed to spawn. Try a different spawn point.")

    logging.debug(
        f"{vehicle.type_id} spawned on map {world.get_map().name} at {spawn_point.location}"
    )

    return vehicle


def spawn_ego_vehicle_at_index(
    world,
    bp_lib,
    spawn_index,
    vehicle_filter="vehicle.tesla.model3"
):
    """
    Spawns the ego vehicle at a specific spawn point index.

    This is useful for deterministic/repeatable scenarios.

    Returns:
        vehicle actor
    """
    spawn_points = get_spawn_points(world)

    if spawn_index >= len(spawn_points):
        raise RuntimeError(
            f"Spawn point index {spawn_index} does not exist. "
            f"This map only has {len(spawn_points)} spawn points."
        )

    spawn_point = spawn_points[spawn_index]

    logging.debug(f"Using fixed spawn point index: {spawn_index}")
    logging.debug(f"Spawn point location: {spawn_point.location}")

    vehicle = spawn_ego_vehicle(
        world,
        bp_lib,
        vehicle_filter=vehicle_filter,
        spawn_point=spawn_point
    )

    return vehicle

def spawn_ego_vehicle_right_lane_of_index(
    world,
    bp_lib,
    spawn_index,
    vehicle_filter="vehicle.tesla.model3"
):
    """
    Spawns the ego vehicle in the right lane of a given spawn point.

    This is useful for testing the same scenario from the lane beside
    the original ego vehicle lane.

    Returns:
        vehicle actor
    """

    spawn_points = get_spawn_points(world)

    if spawn_index >= len(spawn_points):
        raise RuntimeError(
            f"Spawn point index {spawn_index} does not exist. "
            f"This map only has {len(spawn_points)} spawn points."
        )

    original_spawn_point = spawn_points[spawn_index]

    carla_map = world.get_map()

    original_waypoint = carla_map.get_waypoint(
        original_spawn_point.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if original_waypoint is None:
        raise RuntimeError(
            f"Could not find a driving waypoint for spawn point index {spawn_index}."
        )

    right_lane_waypoint = original_waypoint.get_right_lane()

    if right_lane_waypoint is None:
        raise RuntimeError(
            f"No right lane found beside spawn point index {spawn_index}."
        )

    if right_lane_waypoint.lane_type != carla.LaneType.Driving:
        raise RuntimeError(
            f"Right lane beside spawn point index {spawn_index} is not a driving lane."
        )

    right_lane_spawn_point = right_lane_waypoint.transform

    # Keep the vehicle slightly above the road to avoid spawn collision issues.
    right_lane_spawn_point.location.z += 0.3

    logging.info(f"Using base spawn point index: {spawn_index}")
    logging.info(f"Original spawn location: {original_spawn_point.location}")
    logging.info(f"Right lane spawn location: {right_lane_spawn_point.location}")
    logging.info(
        f"Right lane waypoint road_id={right_lane_waypoint.road_id}, "
        f"lane_id={right_lane_waypoint.lane_id}, "
        f"s={right_lane_waypoint.s:.2f}"
    )

    vehicle = spawn_ego_vehicle(
        world,
        bp_lib,
        vehicle_filter=vehicle_filter,
        spawn_point=right_lane_spawn_point
    )

    return vehicle


def spawn_ego_vehicle_behind_index(
    world,
    bp_lib,
    spawn_index,
    distance_back,
    vehicle_filter="vehicle.tesla.model3"
):
    """
    Spawns the ego vehicle a fixed distance behind a given spawn point,
    walking backward along the same lane -- e.g. so the vehicle starts with
    more room before an intersection than the raw spawn point gives it.

    Returns:
        vehicle actor
    """

    spawn_points = get_spawn_points(world)

    if spawn_index >= len(spawn_points):
        raise RuntimeError(
            f"Spawn point index {spawn_index} does not exist. "
            f"This map only has {len(spawn_points)} spawn points."
        )

    original_spawn_point = spawn_points[spawn_index]

    carla_map = world.get_map()

    original_waypoint = carla_map.get_waypoint(
        original_spawn_point.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if original_waypoint is None:
        raise RuntimeError(
            f"Could not find a driving waypoint for spawn point index {spawn_index}."
        )

    behind_waypoints = original_waypoint.previous(distance_back)

    if not behind_waypoints:
        raise RuntimeError(
            f"No waypoint found {distance_back}m behind spawn point index {spawn_index}."
        )

    behind_spawn_point = behind_waypoints[0].transform

    # Keep the vehicle slightly above the road to avoid spawn collision issues.
    behind_spawn_point.location.z += 0.3

    logging.info(f"Using base spawn point index: {spawn_index}")
    logging.info(f"Original spawn location: {original_spawn_point.location}")
    logging.info(f"Spawning {distance_back:.2f}m behind at: {behind_spawn_point.location}")

    vehicle = spawn_ego_vehicle(
        world,
        bp_lib,
        vehicle_filter=vehicle_filter,
        spawn_point=behind_spawn_point
    )

    return vehicle

# -------------------------
# Data saving
# -------------------------

def save_gps_data(gps_data, gps_log_path):
    """
    Saves collected GPS/GNSS data to a JSON file.
    """
    try:
        with open(gps_log_path, "w") as gps_file:
            json.dump(gps_data, gps_file, indent=4)
        return True

    except Exception as gps_error:
        logging.warning(f"Could not save GPS log: {gps_error}")
        return False


def save_collision_data(collision_data, collision_log_path):
    """
    Saves collected collision events to a JSON file.
    """
    try:
        with open(collision_log_path, "w") as collision_file:
            json.dump(collision_data, collision_file, indent=4)
        return True

    except Exception as collision_error:
        logging.warning(f"Could not save collision log: {collision_error}")
        return False


def save_json_log(log_data, log_path):
    """
    Saves a list of log entries (e.g. model predictions, ground-truth
    traffic light state) to a JSON file.
    """
    try:
        with open(log_path, "w") as log_file:
            json.dump(log_data, log_file, indent=4)

    except Exception as log_error:
        logging.warning(f"Could not save log to {log_path}: {log_error}")

# -------------------------
# Cleanup
# -------------------------

def cleanup_actors(actors):
    """
    Stops sensors and destroys all spawned actors.
    """
    logging.debug("Cleaning up spawned actors...")

    for actor in reversed(actors):
        try:
            if actor is not None:
                if hasattr(actor, "stop"):
                    actor.stop()

                actor.destroy()

        except Exception as cleanup_error:
            logging.warning(f"Cleanup error: {cleanup_error}")

    logging.debug("Cleanup complete.")
