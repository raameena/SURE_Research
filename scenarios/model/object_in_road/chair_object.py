"""
chair_object: model-driven straight-road scenario testing perception on an
out-of-distribution static obstacle (a household/street prop -- plastic
chair, table, garden gnome, etc.), not a class the checkpoint was trained on
(vehicle/pedestrian/cyclist/cone). Object blueprint selection/spawning and
the error_detection resolvers live in config/ml_model/actors/object_in_road/
chair.py -- this file only wires them
together (connect, spawn, attach sensors, drive/monitor, clean up).

error_detection perception check -- different mechanism than
scenarios/model/intersection/simple_stop_go.py's LIGHT_TO_EXPECTED:

    scenarios/model/intersection/simple_stop_go.py string-matches specific
    BEV semantic messages ("Traffic light (red) detected", etc.) because the
    checkpoint's bev_semantic_decoder has real, dedicated classes for those.
    It has no "chair"/furniture class at all (confirmed against
    PCLA/pcla_agents/transfuserv4/model.py's forward() -- there's also no
    separate occupancy-grid output; pred_bev_semantic, a per-pixel argmaxed
    class map, is the only spatial BEV signal available). So instead of a
    class-name match, actual_perception here is a generic occupancy proxy:
    "obstacle_detected" if the model's own BEV output shows anything
    NOT classified as a drivable surface (road/lane_markers/
    lane_markers_broken) at the object's projected grid location, else
    "none". See chair.resolve_actual_perception().

v1 scope cut -- expected_action is "stop" only:
    The five action functions (stop/slow/go/go_fast/hard_brake) are
    speed-only -- there is no steering/lateral label in the current decision
    vocabulary to check a "go around if clear" expectation against, and
    whether the harness even exposes the model's lateral/steering
    prediction at all (PCLA/pcla_agents/transfuserv4/model.py's pred_wp) to
    this codebase needs confirming separately before that's buildable. Not
    a bug -- see chair.resolve_expected()'s docstring.
"""

import sys
import logging
from pathlib import Path

import carla

# chair_object.py is inside: scenarios/model/object_in_road/
# parents[3] goes back to: CARLA-Research/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from config.CARLA_actors.carla_config import (
    connect_to_carla,
    get_spawn_points,
    spawn_ego_vehicle,
    cleanup_actors,
    enable_synchronous_mode,
    restore_world_settings
)

from config.CARLA_actors.sensors import (
    attach_standard_sensors,
    attach_collision_sensor
)

from config.ml_model.output_config.object_in_road_output import create_object_in_road_output
from config.ml_model.setup.pcla_navigation import PCLAWorldRouteNavigation

from config.ml_model.setup.model_loader import (
    load_model_runtime,
    attach_model_sensors,
    DASHCAM_CAMERA_POS,
    DASHCAM_CAMERA_ROT,
    MODEL_NAME
)

from controls.ml_controls import (
    monitor_with_model_and_act_for_object
)

from config.ml_model.actors.object_in_road.chair import (
    resolve_expected,
    resolve_actual_perception,
    spawn_object_ahead
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------
# Scenario settings
# -------------------------

SPAWN_POINT_INDEX = 10

TICKS_PER_SECOND = 20

OBJECT_DISTANCE_AHEAD_M = 30.0

# Scenario times out after this many ticks (6.5s @ TICKS_PER_SECOND).
NUM_TICKS = 150

CHECK_INTERVAL_TICKS = 5

# The research camera remains at the requested dashcam viewpoint, while the
# model camera is restored to the checkpoint's original PCLA training mount.
RECORDING_CAMERA_MOUNT = "dashcam"
MODEL_CAMERA_MOUNT = "training"

# error_detection.evaluate_tick()'s mu_friction -- scenario-owned, not a
# shared default (see controls.ml_controls.monitor_with_model_and_act_for_object).
# TODO: confirm real value for this scenario (surface/weather/vehicle
# dependent) -- 0.7 is a placeholder dry-asphalt estimate.
MU_FRICTION = 0.7


# -------------------------
# Left-lane spawn helper
# -------------------------

def get_left_driving_lane_waypoint(world, spawn_index):
    """Resolve the adjacent same-direction left lane for a map spawn point."""
    spawn_points = get_spawn_points(world)

    if spawn_index >= len(spawn_points):
        raise RuntimeError(
            f"Spawn point index {spawn_index} does not exist. "
            f"This map only has {len(spawn_points)} spawn points."
        )

    base_waypoint = world.get_map().get_waypoint(
        spawn_points[spawn_index].location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if base_waypoint is None:
        raise RuntimeError(
            f"Could not find a driving waypoint for spawn point index {spawn_index}."
        )

    left_waypoint = base_waypoint.get_left_lane()

    if left_waypoint is None or left_waypoint.lane_type != carla.LaneType.Driving:
        raise RuntimeError(
            f"No adjacent left driving lane exists at spawn point index {spawn_index}."
        )

    # CARLA lane IDs with opposite signs travel in opposite directions.
    if base_waypoint.lane_id * left_waypoint.lane_id <= 0:
        raise RuntimeError(
            f"The left driving lane at spawn point index {spawn_index} travels "
            "in the opposite direction."
        )

    logging.info(
        "Chair scenario shifted left from lane_id=%s to lane_id=%s.",
        base_waypoint.lane_id,
        left_waypoint.lane_id,
    )
    return left_waypoint


# -------------------------
# Main scenario
# -------------------------

def main():
    spawned_actors = []

    gps_data = []
    collision_data = []
    action_log = []

    previous_world_settings = None
    run_output = None
    run_completed = False

    try:
        # -------------------------
        # Connect to CARLA
        # -------------------------

        client, world, bp_lib = connect_to_carla()

        # -------------------------
        # Strip decorative background traffic
        # -------------------------

        # Town10HD_Opt (an "_Opt", layered map) bakes parked cars/motorcycles
        # along the street into its ParkedVehicles map layer -- these aren't
        # CARLA actors (world.get_actors() never lists them, confirmed
        # against the live server before writing this), so cleanup_actors()
        # has nothing to destroy; unloading the layer is the only way to
        # remove them. This scenario is specifically about testing
        # perception on ONE isolated obstacle -- the only vehicle in the
        # scene should be ego, and the only other actor should be the object
        # this scenario spawns, not a street full of decorative parked
        # traffic. Restored in the finally block below.
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        # See scenarios/model/intersection/simple_stop_go.py's identical
        # comment -- model inference is far slower than one simulation
        # step, so synchronous mode is required for tick-count-based timing
        # (NUM_TICKS, check_interval_ticks) to mean what it says.
        previous_world_settings = enable_synchronous_mode(
            world,
            fixed_delta_seconds=1.0 / TICKS_PER_SECOND
        )

        # -------------------------
        # Create output folders
        # -------------------------

        run_output = create_object_in_road_output(
            scenario="Chair",
            scenario_description=(
                f"Stationary plastic chair placed {OBJECT_DISTANCE_AHEAD_M:.0f} m "
                "ahead of the ego vehicle in the adjacent left driving lane."
            ),
            world=world,
            model_name=MODEL_NAME,
            ego_spawn_point=f"{SPAWN_POINT_INDEX} (adjacent left lane)",
            initial_distance_m=OBJECT_DISTANCE_AHEAD_M,
            simulation_frequency_hz=TICKS_PER_SECOND,
            inference_interval_ticks=CHECK_INTERVAL_TICKS,
            num_ticks=NUM_TICKS,
        )
        images_folder = run_output.images_folder
        lidar_folder = run_output.lidar_folder

        # -------------------------
        # Spawn ego vehicle in the adjacent left driving lane
        # -------------------------

        ego_waypoint = get_left_driving_lane_waypoint(
            world,
            SPAWN_POINT_INDEX,
        )

        left_lane_transform = ego_waypoint.transform
        left_lane_spawn_transform = carla.Transform(
            carla.Location(
                x=left_lane_transform.location.x,
                y=left_lane_transform.location.y,
                z=left_lane_transform.location.z + 0.3,
            ),
            left_lane_transform.rotation,
        )

        vehicle = spawn_ego_vehicle(
            world,
            bp_lib,
            spawn_point=left_lane_spawn_transform,
        )

        spawned_actors.append(vehicle)

        # -------------------------
        # Spawn the out-of-distribution object, same lane, ahead of ego
        # -------------------------

        # Use the resolved left-lane waypoint directly rather than reading
        # vehicle.get_location() before the first world tick. A newly spawned
        # actor's transform may not be reflected by the server immediately.

        navigation_state = PCLAWorldRouteNavigation.from_spawn_waypoint(
            world.get_map(),
            ego_waypoint,
        )

        object_actor = spawn_object_ahead(world, bp_lib, ego_waypoint, OBJECT_DISTANCE_AHEAD_M)
        spawned_actors.append(object_actor)

        object_waypoint = ego_waypoint.next(OBJECT_DISTANCE_AHEAD_M)[0]
        run_output.update_metadata(
            object_type=f"Plastic Chair ({object_actor.type_id})",
            lane_id=object_waypoint.lane_id,
        )

        # -------------------------
        # Attach recording sensors
        # -------------------------

        recording_camera_transform = None

        if RECORDING_CAMERA_MOUNT == "dashcam":
            recording_camera_transform = carla.Transform(
                carla.Location(
                    x=DASHCAM_CAMERA_POS[0],
                    y=DASHCAM_CAMERA_POS[1],
                    z=DASHCAM_CAMERA_POS[2]
                ),
                carla.Rotation(
                    roll=DASHCAM_CAMERA_ROT[0],
                    pitch=DASHCAM_CAMERA_ROT[1],
                    yaw=DASHCAM_CAMERA_ROT[2]
                )
            )

        sensors = attach_standard_sensors(
            world,
            bp_lib,
            vehicle,
            images_folder,
            lidar_folder,
            gps_data,
            camera_transform=recording_camera_transform
        )

        spawned_actors.extend(sensors)

        collision_sensor = attach_collision_sensor(world, bp_lib, vehicle, collision_data)
        spawned_actors.append(collision_sensor)

        # -------------------------
        # Load model and attach its input sensors
        # -------------------------

        model = load_model_runtime(camera_mount=MODEL_CAMERA_MOUNT)

        model_sensors = attach_model_sensors(world, bp_lib, vehicle, model)
        spawned_actors.extend(model_sensors)

        # -------------------------
        # Drive and monitor
        # -------------------------

        run_output.log_startup()

        monitor_with_model_and_act_for_object(
            world,
            vehicle,
            model,
            object_actor,
            num_ticks=NUM_TICKS,
            action_log=action_log,
            resolve_expected=resolve_expected,
            resolve_actual_perception=resolve_actual_perception,
            mu_friction=MU_FRICTION,
            check_interval_ticks=CHECK_INTERVAL_TICKS,
            inference_callback=run_output.record_inference,
            collision_data=collision_data,
            use_pcla_control=True,
            navigation_state=navigation_state,
        )

        run_completed = True

    except KeyboardInterrupt:
        logging.warning("Scenario interrupted by user.")

    except Exception:
        logging.exception("Object in Road chair scenario failed.")
        raise

    finally:
        # -------------------------
        # Restore world settings (undo enable_synchronous_mode())
        # -------------------------

        try:
            if previous_world_settings is not None and "world" in locals():
                restore_world_settings(world, previous_world_settings)

        except Exception as settings_error:
            logging.warning(f"Could not restore world settings: {settings_error}")

        # -------------------------
        # Restore the ParkedVehicles map layer (undo unload_map_layer() above)
        # -------------------------

        try:
            if "world" in locals():
                world.load_map_layer(carla.MapLayer.ParkedVehicles)

        except Exception as layer_error:
            logging.warning(f"Could not restore ParkedVehicles map layer: {layer_error}")

        # -------------------------
        # Save GPS data
        # -------------------------

        if run_output is not None:
            run_output.finalize(
                run_completed,
                gps_data,
                collision_data,
                spawned_actors,
            )
        else:
            cleanup_actors(spawned_actors)


if __name__ == "__main__":
    main()
