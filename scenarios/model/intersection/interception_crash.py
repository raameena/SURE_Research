"""

INTERCEPTION CRASH (cut-in)

interception_crash: model-controlled cut-in scenario. Ego is driven
entirely by the ML model (same pattern as scenarios/model/intersection/
T_bone_crash.py/simple_stop_go.py) -- it is NOT scripted to react to the
crash car in any special way; the point is watching what the model does on
its own once another vehicle cuts into its lane ahead of it.

Different crash geometry than T_bone_crash.py's broadside T-bone (task
point 1): the crash car crosses into ego's lane and settles there, facing
ego's own direction, close ahead of ego -- a cut-in that becomes a
same-lane, in-path *obstacle*, not a car ego might still catch mid-turn on
a perpendicular collision course. All of the crash car's spawn/path/
trigger/settle/tiering logic lives in config/ml_model/actors/
interception_car.py (see that file's module docstring) -- this file only
wires it together (connect, spawn, attach sensors, run the light sequence,
clean up), same division of responsibility as T_bone_crash.py/T_bone_car.py.

Two different error_detection ground truths apply depending on the light
phase, same mechanism as T_bone_crash.py:

    yellow/red phases: light-based ground truth (LIGHT_TO_EXPECTED) -- the
        crash car isn't moving yet (held still since spawn), so there's
        nothing else to check perception against.
    green phase: crash-car-based ground truth (interception_car.
        resolve_expected/resolve_actual_perception, via controls.ml_controls.
        monitor_with_model_and_act_for_object), tiered by distance into
        go -> stop -> hard_brake (task point 3) rather than T_bone_car.py's
        flat relevant/not-relevant threshold, since this scenario exists
        specifically to test whether the model ever issues hard_brake at
        all (see controls.ml_controls.decide_action()'s docstring for how
        "hard_brake" became a reachable decision -- previously the model's
        decoded action vocabulary only ever produced stop/slow/go/go_fast,
        which made an expected_action of "hard_brake" structurally
        unsatisfiable no matter how the model actually behaved).

model_monitor() below branches between the two per phase the same way
T_bone_crash.py's does, by reading the ground-truth light color at the top
of each run_fixed_traffic_light_sequence phase call.

Ground-truth distance (task point 2): interception_car.
compute_gap_to_crash_car_m() -- longitudinal distance from ego's front
bumper to the crash car's rear bumper along ego's own forward direction,
recomputed every check -- not monitor_with_model_and_act_for_object's
default straight-line center-to-near-surface distance (T_bone_crash.py's
choice, correct for a car ego is heading straight at), since the crash car
here spends most of the merge beside/behind ego, where a straight-line
distance would misrepresent how much stopping room ego actually has.

Tuning status -- unlike T_bone_crash.py's crash car timing (ported
unchanged from the original hard-controls car_crash_intersection.py), every
distance/speed constant this scenario introduces (config/ml_model/actors/
interception_car.py's CRASH_CAR_LANE_EXTENSION_M/RELEVANT_RANGE_M/
HARD_BRAKE_RANGE_M/REFERENCE_EGO_SPEED_MPS, and this file's SPAWN_POINT_INDEX/
MU_FRICTION) is a documented-but-unverified placeholder (task point 4): no
live CARLA/Colab session was reachable while this was written, so the
"tight but survivable" geometry -- a correct, timely hard_brake just barely
sufficient to avoid contact -- has not been checked against an actual
model-driven run yet. Because ego is ML-driven (not deterministic like
car_crash_intersection.py's hard-controls track), its speed/position at the
merge point will vary run to run; confirm the geometry actually lands in
that tight-but-survivable zone (via collision_log.json, and the
action_log's distance_to_object_m/ego speed at the point action_correct
starts requiring "hard_brake") before trusting results from this scenario,
and retune interception_car.py's constants if it doesn't.
"""

import os
import sys
import logging
from pathlib import Path

import carla

# Add the main CARLA-Research folder to Python's import path
# interception_crash.py is inside: scenarios/model/intersection/
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

from config.ml_model.actors.interception_car import (
    CRASH_CAR_FORWARD_DISTANCE,
    CRASH_CAR_SIDE_OFFSET,
    CRASH_CAR_SPEED,
    CRASH_CAR_TRIGGER_DISTANCE_METERS,
    CRASH_CAR_TRIGGER_MAX_SECONDS,
    CRASH_CAR_SETTLE_DISTANCE_M,
    CRASH_CAR_LANE_EXTENSION_M,
    make_cutin_controller,
    compute_gap_to_crash_car_m,
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

# Same spawn point/intersection as T_bone_crash.py -- the crash car's spawn/
# path logic (config.CARLA_actors.crash_car) locates the junction ahead of
# ego's actual spawned position, so this must match a spawn point already
# validated to support that junction's left-turn connector.
SPAWN_POINT_INDEX = 11

TRAFFIC_LIGHT_SEARCH_TICKS = 350

# How long the model drives/monitors during the green phase. Longer than
# T_bone_crash.py's GREEN_LIGHT_DRIVE_TICKS (120) -- this scenario needs ego
# to actually close the gap on a settled obstacle roughly
# CRASH_CAR_LANE_EXTENSION_M ahead (task point 4), not just survive a single
# live-crossing moment, so it needs more ticks of approach.
GREEN_LIGHT_DRIVE_TICKS = 200

CHECK_INTERVAL_TICKS = 5

TICKS_PER_SECOND = 20

CRASH_CAR_TRIGGER_MAX_TICKS = int(CRASH_CAR_TRIGGER_MAX_SECONDS * TICKS_PER_SECOND)

# "training" = checkpoint's exact trained camera geometry. "dashcam" =
# windshield-height, forward-mounted driver's-POV mount -- see
# simple_stop_go.py's identical comment.
CAMERA_MOUNT = "dashcam"

# TODO: confirm real value for this scenario (surface/weather/vehicle
# dependent) -- 0.7 is a placeholder dry-asphalt estimate, same as the other
# model scenarios. Also read directly by config.ml_model.actors.
# interception_car's HARD_BRAKE_RANGE_M derivation (MU_FRICTION_ASSUMED,
# duplicated there rather than imported -- see that module's comment) --
# keep the two in sync if this is retuned.
MU_FRICTION = 0.7

# Ground truth for the yellow/red phases, before the crash car is a factor
# -- same shape/values as simple_stop_go.py's LIGHT_TO_EXPECTED. Its
# "green" entry is unused (model_monitor() routes the green phase to the
# crash-car-based monitor instead), kept only so the dict lookup below never
# has to special-case a missing key.
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
            scenario_name="interception_crash",
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

        # Confirms whether the cut-in actually connects with ego (task
        # point 4's "not physically unavoidable" boundary needs real
        # contact evidence to check against, same as T_bone_crash.py).
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
        # Spawn crash car and precompute its cut-in path -- BEFORE enabling
        # synchronous mode below. spawn_crash_car_relative_to_vehicle()
        # calls world.wait_for_tick() internally, which blocks forever once
        # synchronous mode is on and nothing else is calling world.tick()
        # yet -- see T_bone_crash.py's identical comment.
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

        # lane_extension_distance=CRASH_CAR_LANE_EXTENSION_M (not
        # generate_left_turn_path's own 6.0 default) -- see that constant's
        # docstring in interception_car.py for why this scenario needs the
        # crash car to settle much further past the junction than
        # T_bone_crash.py's live-crossing crash car ever needs to travel.
        crash_car_path = generate_left_turn_path(
            crash_car,
            vehicle,
            side_offset=CRASH_CAR_SIDE_OFFSET,
            lane_extension_distance=CRASH_CAR_LANE_EXTENSION_M
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
                crash_car_on_tick = make_cutin_controller(
                    crash_car,
                    crash_car_path,
                    vehicle,
                    speed=CRASH_CAR_SPEED,
                    trigger_distance_m=CRASH_CAR_TRIGGER_DISTANCE_METERS,
                    trigger_max_ticks=CRASH_CAR_TRIGGER_MAX_TICKS,
                    settle_distance_m=CRASH_CAR_SETTLE_DISTANCE_M
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
                    on_each_tick=crash_car_on_tick,
                    compute_distance_to_object_m=compute_gap_to_crash_car_m
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
