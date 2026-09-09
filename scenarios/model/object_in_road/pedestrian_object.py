"""
pedestrian_object: model-driven straight-road scenario testing perception on
an idle pedestrian stopped in the middle of the road -- identical scenario
shape to chair_object.py (same spawn-ahead-of-ego, drive/monitor, cleanup
wiring), just with an idle pedestrian instead of an out-of-distribution
prop. Pedestrian blueprint selection/spawning and the error_detection
resolvers live in config/ml_model/actors/object_in_road/pedestrian.py -- this file only
wires them together (connect, spawn, attach sensors, drive/monitor, clean
up).

error_detection perception check -- different mechanism than chair_object.py's:

    chair_object.py has no real "chair"/furniture class in the checkpoint's
    bev_semantic_decoder, so it falls back to a generic occupancy proxy (see
    that file's module docstring). "walker" (pedestrian) IS a real class in
    both of the model's own semantic segmentation heads (see controls.
    ml_controls.SEMANTIC_CLASS_INFO/BEV_SEMANTIC_CLASS_INFO), so
    actual_perception here is just the model's own already-existing
    "walker" score off pred_bev_semantic -- same convention
    config/ml_model/actors/T_bone_car.py uses for its own "vehicle" class.
    See config/ml_model/actors/object_in_road/pedestrian.py's
    resolve_actual_perception().

v1 scope cut -- expected_action is "stop" only:
    Same cut as chair_object.py -- see that file's module docstring and
    config/ml_model/actors/object_in_road/pedestrian.py's
    resolve_expected() docstring.
"""

import sys
import logging
from pathlib import Path

import carla

# pedestrian_object.py is inside: scenarios/model/object_in_road/
# parents[3] goes back to: CARLA-Research/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from config.CARLA_actors.carla_config import (
    connect_to_carla,
    get_spawn_points,
    spawn_ego_vehicle_at_index,
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

from config.ml_model.actors.object_in_road.pedestrian import (
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
NUM_TICKS = 160

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
# Main scenario
# -------------------------

def main(
    scenario_name="Pedestrian",
    scenario_description=None,
    object_type_label="Pedestrian",
    actor_spawn_object_ahead=spawn_object_ahead,
    actor_resolve_expected=resolve_expected,
    actor_resolve_actual_perception=resolve_actual_perception,
):
    """Run the pedestrian scenario with injectable appearance-specific actors."""
    if scenario_description is None:
        scenario_description = (
            f"Stationary pedestrian placed {OBJECT_DISTANCE_AHEAD_M:.0f} m "
            "ahead of the ego vehicle."
        )

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
        # scene should be ego, and the only other actor should be the idle
        # pedestrian this scenario spawns, not a street full of decorative
        # parked traffic. Restored in the finally block below.
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
            scenario=scenario_name,
            scenario_description=scenario_description,
            world=world,
            model_name=MODEL_NAME,
            ego_spawn_point=SPAWN_POINT_INDEX,
            initial_distance_m=OBJECT_DISTANCE_AHEAD_M,
            simulation_frequency_hz=TICKS_PER_SECOND,
            inference_interval_ticks=CHECK_INTERVAL_TICKS,
            num_ticks=NUM_TICKS,
        )
        images_folder = run_output.images_folder
        lidar_folder = run_output.lidar_folder

        # -------------------------
        # Spawn ego vehicle
        # -------------------------

        vehicle = spawn_ego_vehicle_at_index(
            world,
            bp_lib,
            spawn_index=SPAWN_POINT_INDEX
        )

        spawned_actors.append(vehicle)

        # -------------------------
        # Spawn the idle pedestrian, same lane, ahead of ego
        # -------------------------

        # Uses the known spawn_points[SPAWN_POINT_INDEX] transform, not
        # vehicle.get_location() -- immediately after spawn_actor(), a
        # freshly spawned actor's transform isn't guaranteed to be reflected
        # yet (confirmed empirically: reading it back here before any
        # world.tick() returned a stale (0,0,0), which silently snapped
        # get_waypoint() onto a real but completely wrong waypoint near the
        # world origin instead of raising). spawn_ego_vehicle_behind_index()
        # in carla_config.py sidesteps the same hazard the same way, using
        # the spawn point's own known location rather than round-tripping
        # through the just-spawned actor.
        spawn_points = get_spawn_points(world)

        ego_waypoint = world.get_map().get_waypoint(
            spawn_points[SPAWN_POINT_INDEX].location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        navigation_state = PCLAWorldRouteNavigation.from_spawn_waypoint(
            world.get_map(),
            ego_waypoint,
        )

        object_actor = actor_spawn_object_ahead(
            world,
            bp_lib,
            ego_waypoint,
            OBJECT_DISTANCE_AHEAD_M,
        )
        spawned_actors.append(object_actor)

        object_waypoint = ego_waypoint.next(OBJECT_DISTANCE_AHEAD_M)[0]
        run_output.update_metadata(
            object_type=f"{object_type_label} ({object_actor.type_id})",
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
            resolve_expected=actor_resolve_expected,
            resolve_actual_perception=actor_resolve_actual_perception,
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
        logging.exception("Object in Road %s scenario failed.", scenario_name)
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
