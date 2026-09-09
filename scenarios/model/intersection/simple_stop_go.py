import os
import sys
import logging
from pathlib import Path

import carla

# Add the main CARLA-Research folder to Python's import path
# simple_stop_go.py is inside: scenarios/model/intersection/
# parents[3] goes back to: CARLA-Research/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from config.CARLA_actors.carla_config import (
    connect_to_carla,
    create_run_folder,
    spawn_ego_vehicle_behind_index,
    save_gps_data,
    cleanup_actors,
    enable_synchronous_mode,
    restore_world_settings
)

from config.CARLA_actors.sensors import (
    attach_standard_sensors
)

from config.ml_model.setup.model_loader import (
    load_model_runtime,
    attach_model_sensors,
    DASHCAM_CAMERA_POS,
    DASHCAM_CAMERA_ROT,
    MODEL_NAME
)

from controls.light_controls import (
    find_vehicle_traffic_light,
    run_fixed_traffic_light_sequence,
    force_traffic_light_state,
    unfreeze_traffic_light
)

from controls.ml_controls import (
    monitor_with_model_and_act,
    log_model_startup_prediction,
    GO_TARGET_SPEED
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------
# Scenario settings
# -------------------------

SPAWN_POINT_INDEX = 11

TICKS_PER_SECOND = 20

# Moves the spawn point back along the lane by roughly the distance the
# vehicle would cover in SPAWN_BACKWARD_TICKS ticks at GO_TARGET_SPEED (one
# of the model's own canonical "go" bins, used here as a representative
# cruising-speed estimate -- the vehicle actually starts from a standstill,
# so this is an approximation, not a physics-accurate distance). Spawn point
# 11 put the ego essentially on top of the stop line already (the light was
# found at path index 0 from that spawn point), leaving almost no room to
# react before the intersection.
SPAWN_BACKWARD_TICKS = 60
SPAWN_BACKWARD_DISTANCE_METERS = GO_TARGET_SPEED * (SPAWN_BACKWARD_TICKS / TICKS_PER_SECOND)

TRAFFIC_LIGHT_SEARCH_TICKS = 350

# How long the model drives/monitors during the green phase (run_fixed_traffic_light_sequence's
# green_ticks below) -- the num_ticks a scenario actually controls, same
# convention as scenarios/model/object_in_road/chair_object.py's own
# NUM_TICKS. yellow/red phase lengths are set in seconds further down
# (yellow_seconds/red_seconds) since those are tied to real light-timing
# semantics, not a driving duration to tune per scenario.
NUM_TICKS = 130

# Same as the hardcoded scenarios' default -- not optimizing model-call
# timing yet.
CHECK_INTERVAL_TICKS = 5

# "training" = checkpoint's exact trained camera geometry (x=-1.5, z=2.0,
# roof-mounted). "dashcam" = windshield-height, forward-mounted driver's-POV
# mount -- see model_loader.DASHCAM_CAMERA_POS. One-line toggle to compare
# both; not a "fix" for the training mount, per professor's direction.
CAMERA_MOUNT = "dashcam"

# -------------------------
# error_detection tuning -- scenario-owned (see controls.ml_controls.
# monitor_with_model_and_act and config.ml_model.setup.error_detection), not a
# shared default, since a different scenario's surface/vehicle/weather could
# need a different MU_FRICTION or a different expected perception/action.
# -------------------------

# TODO: confirm real value for this scenario (surface/weather/vehicle
# dependent) -- 0.7 is a placeholder dry-asphalt estimate.
MU_FRICTION = 0.7

# error_detection.evaluate_tick()'s ground truth for this scenario, keyed by
# the light color read this tick (controls.light_controls.
# traffic_light_state_to_string()'s "red"/"yellow"/"green" -- must match that
# casing). The sequence cycles yellow -> red -> green (see
# run_fixed_traffic_light_sequence() below), and the correct
# perception/action changes with it, so this is resolved fresh every check
# rather than hardcoded to one phase. expected_perception values match
# controls.ml_controls' BEV-color-message -> "<color>_light" reduction.
# expected_action values are TARGET_SPEED_TO_ACTION's existing labels -- the
# bin that actually gives the safe behavior for that phase.
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
    light_log = []
    action_log = []

    traffic_light = None
    previous_world_settings = None

    try:
        # -------------------------
        # Connect to CARLA
        # -------------------------

        client, world, bp_lib = connect_to_carla()

        # Model inference (~1s+ per call) is far slower than one 0.05s
        # simulation step. Without synchronous mode, CARLA free-runs on its
        # own real-time clock while inference blocks the client -- the
        # vehicle keeps moving under whatever control was last applied for
        # that entire duration, so the frame a decision was based on is
        # already stale relative to the vehicle's position by the time the
        # decision is applied. Synchronous mode makes the simulation only
        # advance when this script calls world.tick() (see
        # controls.ml_controls.monitor_with_model_and_act and
        # config.ml_model.setup.model_loader.attach_model_sensors, both of which
        # now call world.tick() explicitly instead of world.wait_for_tick()),
        # so tick-count-based timing (red_seconds * ticks_per_second, etc.)
        # actually means what it says.
        previous_world_settings = enable_synchronous_mode(
            world,
            fixed_delta_seconds=1.0 / TICKS_PER_SECOND
        )

        # -------------------------
        # Create output folders
        # -------------------------

        # output/model/simple_stop_go/run_N/ -- see carla_config.create_run_folder()'s
        # docstring for why this doesn't also key on model/intersection.
        run_folder, images_folder, lidar_folder, gps_log_path = create_run_folder(
            control_type="model",
            scenario_name="simple_stop_go",
            use_run_folder=True
        )

        # Persist this run's logging.info()/warning()/error() output (tick
        # decisions, error_detection results, etc.) alongside its images/
        # LiDAR/GPS data -- the module-level basicConfig() above only wrote
        # to the console, so nothing was kept once the process exited.
        run_log_handler = logging.FileHandler(
            os.path.join(run_folder, "scenario_log.txt")
        )
        run_log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(run_log_handler)

        # Which intersection this run tested -- map name plus SPAWN_POINT_INDEX
        # (the scenario's own fixed starting point), logged (not folder-path-
        # encoded, see create_run_folder()'s docstring) so it's still
        # traceable per run without contributing to folder sprawl.
        intersection_name = f"{world.get_map().name}_spawn{SPAWN_POINT_INDEX}"
        logging.info(f"model={MODEL_NAME} intersection={intersection_name}")

        # -------------------------
        # Spawn ego vehicle
        # -------------------------

        vehicle = spawn_ego_vehicle_behind_index(
            world,
            bp_lib,
            spawn_index=SPAWN_POINT_INDEX,
            distance_back=SPAWN_BACKWARD_DISTANCE_METERS
        )

        spawned_actors.append(vehicle)

        # -------------------------
        # Attach recording sensors
        # -------------------------

        # Tied to CAMERA_MOUNT so the recording camera always matches the
        # model-input camera's angle -- None (attach_rgb_camera()'s own
        # default, x=1.5/z=2.4) when CAMERA_MOUNT="training".
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
        # Load model and attach its input sensors
        # -------------------------

        model = load_model_runtime(camera_mount=CAMERA_MOUNT)

        model_sensors = attach_model_sensors(world, bp_lib, vehicle, model)
        spawned_actors.extend(model_sensors)

        # -------------------------
        # One-time startup diagnostic (vehicle is still at a standstill --
        # isolates "does the model detect the red light" from any
        # movement/state-machine logic). Not part of the control loop.
        #
        # Force the light to red first -- run_fixed_traffic_light_sequence
        # hasn't run yet at this point, so without this the light would
        # still be in whatever state CARLA's own cycle left it in, and the
        # diagnostic wouldn't actually be testing "facing a red light".
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
        # (currently red, then green) -- see monitor_with_model_and_act's
        # docstring for why this can't be a local inside that function.
        # Starts at "go": tick 0 of the first phase applies this default
        # itself, so there's no separate one-shot "start moving" call.
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
            yellow_seconds=2,
            red_seconds=3,
            green_ticks=NUM_TICKS
        )

    except KeyboardInterrupt:
        logging.warning("Scenario interrupted by user.")

    except Exception as e:
        logging.error(f"Something went wrong: {e}")

    finally:
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
        # Save GPS data
        # -------------------------

        if "gps_log_path" in locals():
            save_gps_data(
                gps_data,
                gps_log_path
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
