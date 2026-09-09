"""Shared orchestration for Early, Medium, and Late Interception."""

import logging

import carla

from config.CARLA_actors.carla_config import (
    cleanup_actors,
    connect_to_carla,
    enable_synchronous_mode,
    restore_world_settings,
    spawn_ego_vehicle_at_index,
)
from config.CARLA_actors.sensors import attach_collision_sensor, attach_standard_sensors
from config.ml_model.actors.intersection_interception.common import (
    compute_distance_to_interception_vehicle_m,
    generate_interception_path,
    make_interception_controller,
    resolve_actual_perception,
    resolve_expected,
    spawn_interception_vehicle,
    stop_crash_car,
)
from config.ml_model.output_config.intersection_interception_output import (
    create_intersection_interception_output,
)
from config.ml_model.setup.model_loader import (
    DASHCAM_CAMERA_POS,
    DASHCAM_CAMERA_ROT,
    MODEL_NAME,
    attach_model_sensors,
    load_model_runtime,
)
from controls.light_controls import find_vehicle_traffic_light, force_traffic_light_state
from controls.ml_controls import monitor_with_model_and_act_for_object


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


SPAWN_POINT_INDEX = 11
TRAFFIC_LIGHT_SEARCH_TICKS = 350
TICKS_PER_SECOND = 20
CHECK_INTERVAL_TICKS = 5

# Shared 12-second observation horizon. The 8-second trigger fallback leaves
# four seconds to observe the crossing even if the model-driven ego never
# reaches the configured distance trigger.
NUM_TICKS = 240
CAMERA_MOUNT = "dashcam"

# Existing intersection-model target-point convention.
TARGET_POINT_FORWARD_DISTANCE_M = 15.0

# Existing model-scenario placeholder for nominal dry asphalt. This is an
# error-detection physics input, not scripted ego control; validate it against
# the actual CARLA weather/surface if safety labels are used analytically.
MU_FRICTION = 0.7


def _dashcam_transform():
    return carla.Transform(
        carla.Location(
            x=DASHCAM_CAMERA_POS[0],
            y=DASHCAM_CAMERA_POS[1],
            z=DASHCAM_CAMERA_POS[2],
        ),
        carla.Rotation(
            roll=DASHCAM_CAMERA_ROT[0],
            pitch=DASHCAM_CAMERA_ROT[1],
            yaw=DASHCAM_CAMERA_ROT[2],
        ),
    )


def run_interception_scenario(scenario_config):
    """Run one configured variant while keeping all common conditions fixed."""

    spawned_actors = []
    gps_data = []
    collision_data = []
    action_log = []

    world = None
    traffic_light = None
    previous_light_settings = []
    previous_world_settings = None
    interception_vehicle = None
    controller = None
    run_output = None
    run_completed = False

    try:
        _client, world, bp_lib = connect_to_carla()

        run_output = create_intersection_interception_output(
            scenario_config=scenario_config,
            world=world,
            model_name=MODEL_NAME,
            ego_spawn_point=SPAWN_POINT_INDEX,
            simulation_frequency_hz=TICKS_PER_SECOND,
            inference_interval_ticks=CHECK_INTERVAL_TICKS,
            num_ticks=NUM_TICKS,
        )

        vehicle = spawn_ego_vehicle_at_index(
            world,
            bp_lib,
            spawn_index=SPAWN_POINT_INDEX,
        )
        spawned_actors.append(vehicle)

        traffic_light = find_vehicle_traffic_light(
            world,
            vehicle,
            max_ticks=TRAFFIC_LIGHT_SEARCH_TICKS,
        )
        if traffic_light is None:
            raise RuntimeError(
                "No traffic light affecting the ego route was found from spawn point 11."
            )

        light_group = (
            list(traffic_light.get_group_traffic_lights())
            if hasattr(traffic_light, "get_group_traffic_lights")
            else [traffic_light]
        )
        if all(group_light.id != traffic_light.id for group_light in light_group):
            light_group.append(traffic_light)
        previous_light_settings = [
            (
                group_light,
                group_light.get_state(),
                bool(group_light.is_frozen())
                if hasattr(group_light, "is_frozen") else False,
            )
            for group_light in light_group
        ]
        force_traffic_light_state(
            world,
            traffic_light,
            carla.TrafficLightState.Green,
            "Green (fixed)",
        )
        for group_light in light_group:
            if group_light.id != traffic_light.id:
                group_light.set_state(carla.TrafficLightState.Red)
                group_light.freeze(True)
        logging.info(
            "Ego signal fixed GREEN; %d other signal(s) in its intersection "
            "group fixed RED.",
            max(0, len(light_group) - 1),
        )

        interception_vehicle = spawn_interception_vehicle(
            world,
            bp_lib,
            vehicle,
            forward_distance=scenario_config.interception_vehicle_forward_distance_m,
            side_offset=scenario_config.interception_vehicle_side_offset_m,
        )
        spawned_actors.append(interception_vehicle)
        interception_path = generate_interception_path(
            interception_vehicle,
            vehicle,
            side_offset=scenario_config.interception_vehicle_side_offset_m,
            waypoint_step=scenario_config.path_step_m,
            lane_extension_distance=scenario_config.path_extension_m,
        )
        run_output.update_metadata(
            crossing_vehicle_type=interception_vehicle.type_id,
            notes=(
                "interception_car.py cut-in/settling geometry and Early/Medium/Late trigger "
                "values are initial tuning parameters requiring a live CARLA run."
            ),
        )

        recording_sensors = attach_standard_sensors(
            world,
            bp_lib,
            vehicle,
            run_output.images_folder,
            run_output.lidar_folder,
            gps_data,
            camera_transform=_dashcam_transform(),
        )
        spawned_actors.extend(recording_sensors)
        collision_sensor = attach_collision_sensor(
            world, bp_lib, vehicle, collision_data
        )
        spawned_actors.append(collision_sensor)

        previous_world_settings = enable_synchronous_mode(
            world,
            fixed_delta_seconds=1.0 / TICKS_PER_SECOND,
        )

        model = load_model_runtime(camera_mount=CAMERA_MOUNT)
        spawned_actors.extend(attach_model_sensors(world, bp_lib, vehicle, model))

        controller = make_interception_controller(
            interception_vehicle,
            interception_path,
            vehicle,
            speed=scenario_config.interception_vehicle_speed_mps,
            trigger_distance_m=scenario_config.trigger_distance_m,
            trigger_max_ticks=int(scenario_config.trigger_max_seconds * TICKS_PER_SECOND),
            settle_distance_m=scenario_config.settle_distance_m,
        )

        run_output.log_startup()
        monitor_with_model_and_act_for_object(
            world,
            vehicle,
            model,
            interception_vehicle,
            num_ticks=NUM_TICKS,
            action_log=action_log,
            resolve_expected=resolve_expected,
            resolve_actual_perception=resolve_actual_perception,
            mu_friction=MU_FRICTION,
            check_interval_ticks=CHECK_INTERVAL_TICKS,
            forward_distance=TARGET_POINT_FORWARD_DISTANCE_M,
            on_each_tick=controller,
            compute_distance_to_object_m=compute_distance_to_interception_vehicle_m,
            inference_callback=run_output.record_inference,
            collision_data=collision_data,
            distance_log_name="distance_to_interception_vehicle",
            distance_display_name="Distance to Interception Vehicle",
            object_speed_log_name="interception_vehicle_speed",
            object_speed_display_name="Interception Vehicle Speed",
            collision_actor_display_name="Interception Vehicle",
        )

        run_output.update_metadata(
            actual_trigger_tick=controller.state["trigger_tick"],
            actual_trigger_distance_m=controller.state["actual_trigger_distance_m"],
        )
        run_completed = True

    except KeyboardInterrupt:
        logging.warning("Intersection Interception scenario interrupted by user.")

    except Exception:
        logging.exception("Intersection Interception scenario failed.")
        raise

    finally:
        try:
            stop_crash_car(interception_vehicle)
        except Exception as stop_error:
            logging.warning("Could not stop interception vehicle: %s", stop_error)

        try:
            if previous_light_settings:
                # Explicitly unfreeze first, then restore every intersection
                # signal's original state and freeze behavior.
                for group_light, previous_state, previous_frozen in previous_light_settings:
                    group_light.freeze(False)
                    group_light.set_state(previous_state)
                    group_light.freeze(previous_frozen)
                logging.info(
                    "Traffic light group state and freeze behavior restored."
                )
        except Exception as light_error:
            logging.warning("Could not restore traffic light: %s", light_error)

        try:
            if world is not None and previous_world_settings is not None:
                restore_world_settings(world, previous_world_settings)
        except Exception as settings_error:
            logging.warning("Could not restore world settings: %s", settings_error)

        if run_output is not None:
            if controller is not None:
                run_output.update_metadata(
                    actual_trigger_tick=controller.state["trigger_tick"],
                    actual_trigger_distance_m=controller.state["actual_trigger_distance_m"],
                )
            run_output.finalize(
                run_completed,
                gps_data,
                collision_data,
                spawned_actors,
            )
        else:
            cleanup_actors(spawned_actors)
