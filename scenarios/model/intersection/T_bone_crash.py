"""

T BONE

T_bone_crash: model-controlled version of scenarios/hard_controls/intersection/
car_crash_intersection.py's unprotected-left-turn T-bone. Ego is driven
entirely by the ML model (same pattern as scenarios/model/intersection/
simple_stop_go.py) -- it is NOT scripted to react to the crash car in any
special way; the point is watching what the model does on its own when
another vehicle runs its own red light into ego's path.

Two different error_detection ground truths apply depending on the light
phase, since the crash car (config/ml_model/actors/T_bone_car.py) stays
held still until ego's light turns green and only becomes a real hazard
partway through that phase:

    yellow/red phases: light-based ground truth (LIGHT_TO_EXPECTED), same
        mechanism as simple_stop_go.py -- the crash car isn't moving yet, so
        there's nothing else to check perception against.
    green phase: crash-car-based ground truth (T_bone_car.resolve_expected/
        resolve_actual_perception, via controls.ml_controls.
        monitor_with_model_and_act_for_object), since that's the actual
        hazard under test during this phase -- an in-distribution "vehicle"
        perception check, not an OOD workaround (T_bone_car.py's module
        docstring).

model_monitor() below branches between the two per phase by reading the
ground-truth light color at the top of each run_fixed_traffic_light_sequence
phase call (that call's own tick 0 is always the instant the light was just
force-set to that phase's color).

Crash car timing (spawn/path/speed/trigger) is ported unchanged from
car_crash_intersection.py -- see config/ml_model/actors/T_bone_car.py's
module docstring. That original timing was itself only ever a placeholder,
tuned against nothing (there was no live CARLA server available when it was
written -- see car_crash_intersection.py's CRASH_CAR_SPEED comment), and the
model-driven ego here can take a different line/speed through the
intersection than the original hard-controls scenario's deterministic one.
Confirm actual contact via collision_log.json (attach_collision_sensor,
below) after a run -- if it doesn't reliably connect, the fixed
distance/tick trigger in T_bone_car.make_crash_car_controller() needs to
become a live-position pursuit calc instead of a looser trajectory.
"""

import os
import sys
import logging
from pathlib import Path

import carla

# Add the main CARLA-Research folder to Python's import path
# T_bone_crash.py is inside: scenarios/model/intersection/
# parents[3] goes back to: CARLA-Research/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from config.CARLA_actors.carla_config import (
    connect_to_carla,
    create_run_folder,
    spawn_ego_vehicle_at_index,
    save_gps_data,
    save_collision_data,
    cleanup_actors,
    enable_synchronous_mode,
    restore_world_settings
)

from config.CARLA_actors.sensors import (
    attach_standard_sensors,
    attach_collision_sensor
)

from config.CARLA_actors.crash_car import (
    spawn_crash_car_relative_to_vehicle,
    generate_left_turn_path,
    stop_crash_car
)

from config.ml_model.setup.model_loader import (
    load_model_runtime,
    attach_model_sensors,
    DASHCAM_CAMERA_POS,
    DASHCAM_CAMERA_ROT,
    MODEL_NAME
)

from config.ml_model.actors.T_bone_car import (
    CRASH_CAR_FORWARD_DISTANCE,
    CRASH_CAR_SIDE_OFFSET,
    CRASH_CAR_SPEED,
    CRASH_CAR_TRIGGER_DISTANCE_METERS,
    CRASH_CAR_TRIGGER_MAX_SECONDS,
    make_crash_car_controller,
    resolve_expected,
    resolve_actual_perception
)

from controls.light_controls import (
    find_vehicle_traffic_light,
    run_fixed_traffic_light_sequence,
    force_traffic_light_state,
    unfreeze_traffic_light,
    get_light_color
)

from controls.ml_controls import (
    monitor_with_model_and_act,
    monitor_with_model_and_act_for_object,
    log_model_startup_prediction
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------
# Scenario settings
# -------------------------

# Same spawn point/no backward offset as car_crash_intersection.py -- the
# crash car's spawn/path logic (config.CARLA_actors.crash_car) locates the
# junction ahead of ego's actual spawned position, so this must match the
# original to reuse the same intersection geometry.
SPAWN_POINT_INDEX = 11

TRAFFIC_LIGHT_SEARCH_TICKS = 350

# How long the model drives/monitors during the green phase -- ported
# unchanged from car_crash_intersection.py's GREEN_LIGHT_DRIVE_TICKS.
GREEN_LIGHT_DRIVE_TICKS = 120

CHECK_INTERVAL_TICKS = 5

TICKS_PER_SECOND = 20

CRASH_CAR_TRIGGER_MAX_TICKS = int(CRASH_CAR_TRIGGER_MAX_SECONDS * TICKS_PER_SECOND)

# "training" = checkpoint's exact trained camera geometry. "dashcam" =
# windshield-height, forward-mounted driver's-POV mount -- see
# simple_stop_go.py's identical comment.
CAMERA_MOUNT = "dashcam"

# TODO: confirm real value for this scenario (surface/weather/vehicle
# dependent) -- 0.7 is a placeholder dry-asphalt estimate, same as the
# other model scenarios.
MU_FRICTION = 0.7

# Ground truth for the yellow/red phases, before the crash car is a factor
# -- same shape/values as simple_stop_go.py's LIGHT_TO_EXPECTED. Its
# "green" entry is unused (model_monitor() routes the green phase to the
# crash-car-based monitor instead), kept only so the dict lookup below
# never has to special-case a missing key.
LIGHT_TO_EXPECTED = {
    "red": {"expected_perception": "red_light", "expected_action": "stop"},
    "yellow": {"expected_perception": "yellow_light", "expected_action": "slow"},
    "green": {"expected_perception": "green_light", "expected_action": "go"},
}


# -------------------------
# Main scenario
# -------------------------

def main():
    spawned_actors = []

    gps_data = []
    collision_data = []
    light_log = []
    action_log = []

    traffic_light = None
    crash_car = None
    crash_car_path = None
    previous_world_settings = None

    try:
        # -------------------------
        # Connect to CARLA
        # -------------------------

        client, world, bp_lib = connect_to_carla()

        # -------------------------
        # Create output folders
        # -------------------------

        run_folder, images_folder, lidar_folder, gps_log_path = create_run_folder(
            control_type="model",
            scenario_name="T_bone_crash",
            use_run_folder=True
        )

        collision_log_path = os.path.join(run_folder, "collision_log.json")

        run_log_handler = logging.FileHandler(
            os.path.join(run_folder, "scenario_log.txt")
        )
        run_log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(run_log_handler)

        logging.info(f"model={MODEL_NAME} spawn_index={SPAWN_POINT_INDEX}")

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
        # Attach recording + collision sensors
        # -------------------------

        recording_camera_transform = None

        if CAMERA_MOUNT == "dashcam":
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

        # Confirms whether the crash car actually connects with ego (see
        # this module's docstring, point 3) -- without this there is no
        # ground truth for "did a T-bone actually happen this run" at all.
        collision_sensor = attach_collision_sensor(
            world,
            bp_lib,
            vehicle,
            collision_data
        )

        spawned_actors.append(collision_sensor)

        # -------------------------
        # Find correct traffic light
        # -------------------------

        traffic_light = find_vehicle_traffic_light(
            world,
            vehicle,
            max_ticks=TRAFFIC_LIGHT_SEARCH_TICKS,
            creep_throttle=0.2
        )

        if traffic_light is None:
            raise RuntimeError(
                "No traffic light was detected as affecting the ego vehicle."
            )

        # -------------------------
        # Spawn crash car and precompute its turn path -- BEFORE enabling
        # synchronous mode below. spawn_crash_car_relative_to_vehicle()
        # (config.CARLA_actors.crash_car) calls world.wait_for_tick()
        # internally, which blocks forever once synchronous mode is on and
        # nothing else is calling world.tick() yet (see carla_config.
        # advance_simulation()'s docstring) -- car_crash_intersection.py
        # never hits this because the hard-controls track never enables
        # synchronous mode at all.
        # -------------------------

        crash_car = spawn_crash_car_relative_to_vehicle(
            world,
            bp_lib,
            vehicle,
            forward_distance=CRASH_CAR_FORWARD_DISTANCE,
            side_offset=CRASH_CAR_SIDE_OFFSET
        )

        spawned_actors.append(crash_car)

        world.wait_for_tick()

        crash_car_path = generate_left_turn_path(
            crash_car,
            vehicle,
            side_offset=CRASH_CAR_SIDE_OFFSET
        )

        # -------------------------
        # Enable synchronous mode -- required from here on for model
        # inference-driven control (see simple_stop_go.py/chair_object.py's
        # identical rationale).
        # -------------------------

        previous_world_settings = enable_synchronous_mode(
            world,
            fixed_delta_seconds=1.0 / TICKS_PER_SECOND
        )

        # -------------------------
        # Load model and attach its input sensors
        # -------------------------

        model = load_model_runtime(camera_mount=CAMERA_MOUNT)

        model_sensors = attach_model_sensors(world, bp_lib, vehicle, model)
        spawned_actors.extend(model_sensors)

        # -------------------------
        # One-time startup diagnostic -- see simple_stop_go.py's identical
        # step for why (isolates "does the model detect a red light" from
        # any movement/state-machine logic).
        # -------------------------

        force_traffic_light_state(
            world,
            traffic_light,
            carla.TrafficLightState.Red,
            "Red"
        )

        log_model_startup_prediction(
            model,
            vehicle,
            world
        )

        # -------------------------
        # Model-driven monitor wrapper
        # -------------------------

        # Persists across every phase of run_fixed_traffic_light_sequence
        # (yellow, red, green) -- see monitor_with_model_and_act()'s
        # docstring for why this can't be a local inside that function.
        current_command_state = {"action": "go"}

        def model_monitor(
            world,
            vehicle,
            traffic_light,
            num_ticks,
            light_log,
            action_log,
            check_interval_ticks
        ):
            # Ground truth for which phase this call covers -- the light
            # was just force-set to this phase's color right before
            # run_fixed_traffic_light_sequence calls this function (see
            # this module's docstring).
            light_color = get_light_color(traffic_light)

            if light_color == "green":
                crash_car_on_tick = make_crash_car_controller(
                    crash_car,
                    crash_car_path,
                    vehicle,
                    speed=CRASH_CAR_SPEED,
                    trigger_distance_m=CRASH_CAR_TRIGGER_DISTANCE_METERS,
                    trigger_max_ticks=CRASH_CAR_TRIGGER_MAX_TICKS
                )

                monitor_with_model_and_act_for_object(
                    world,
                    vehicle,
                    model,
                    crash_car,
                    num_ticks=num_ticks,
                    action_log=action_log,
                    resolve_expected=resolve_expected,
                    resolve_actual_perception=resolve_actual_perception,
                    mu_friction=MU_FRICTION,
                    check_interval_ticks=check_interval_ticks,
                    current_command_state=current_command_state,
                    on_each_tick=crash_car_on_tick
                )
            else:
                monitor_with_model_and_act(
                    world,
                    vehicle,
                    model,
                    traffic_light,
                    num_ticks=num_ticks,
                    light_log=light_log,
                    action_log=action_log,
                    light_to_expected=LIGHT_TO_EXPECTED,
                    mu_friction=MU_FRICTION,
                    check_interval_ticks=check_interval_ticks,
                    current_command_state=current_command_state
                )

        # -------------------------
        # Run fixed yellow-red-green sequence
        # -------------------------

        run_fixed_traffic_light_sequence(
            world,
            vehicle,
            traffic_light,
            model_monitor,
            light_log,
            action_log,
            check_interval_ticks=CHECK_INTERVAL_TICKS,
            ticks_per_second=TICKS_PER_SECOND,
            yellow_seconds=2.5,
            red_seconds=3,
            green_ticks=GREEN_LIGHT_DRIVE_TICKS
        )

    except KeyboardInterrupt:
        logging.warning("Scenario interrupted by user.")

    except Exception as e:
        logging.error(f"Something went wrong: {e}")

    finally:
        # -------------------------
        # Stop crash car
        # -------------------------

        try:
            if crash_car is not None:
                stop_crash_car(crash_car)

        except Exception as crash_car_error:
            logging.warning(f"Could not stop crash car: {crash_car_error}")

        # -------------------------
        # Unfreeze traffic light
        # -------------------------

        try:
            if traffic_light is not None:
                unfreeze_traffic_light(traffic_light)

        except Exception as light_error:
            logging.warning(f"Could not unfreeze traffic light: {light_error}")

        # -------------------------
        # Restore world settings (undo enable_synchronous_mode())
        # -------------------------

        try:
            if previous_world_settings is not None and "world" in locals():
                restore_world_settings(world, previous_world_settings)

        except Exception as settings_error:
            logging.warning(f"Could not restore world settings: {settings_error}")

        # -------------------------
        # Save GPS and collision data
        # -------------------------

        if "gps_log_path" in locals():
            save_gps_data(
                gps_data,
                gps_log_path
            )

        if "collision_log_path" in locals():
            save_collision_data(
                collision_data,
                collision_log_path
            )

        # -------------------------
        # Cleanup actors
        # -------------------------

        cleanup_actors(spawned_actors)

        # -------------------------
        # Detach this run's file log handler
        # -------------------------

        if "run_log_handler" in locals():
            run_log_handler.close()
            logging.getLogger().removeHandler(run_log_handler)


if __name__ == "__main__":
    main()
