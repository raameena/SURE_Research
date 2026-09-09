import sys
import os
import logging
from pathlib import Path

# Add the main CARLA-Research folder to Python's import path
# car_crash_intersection.py is inside: scenarios/hard_controls/intersection/
# parents[3] goes back to: CARLA-Research/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from config.CARLA_actors.carla_config import (
    connect_to_carla,
    create_run_folder,
    spawn_ego_vehicle_at_index,
    save_gps_data,
    save_collision_data,
    cleanup_actors
)

from config.CARLA_actors.sensors import (
    attach_standard_sensors,
    attach_collision_sensor
)

from config.CARLA_actors.crash_car import (
    spawn_crash_car_relative_to_vehicle,
    generate_left_turn_path,
    start_crash_car_maneuver,
    update_crash_car_control,
    stop_crash_car
)

from controls.light_controls import (
    find_vehicle_traffic_light,
    run_fixed_traffic_light_sequence,
    unfreeze_traffic_light
)

from controls.hazard_controls import (
    monitor_light_and_hazards_and_act
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------
# Scenario settings
# -------------------------

SPAWN_POINT_INDEX = 11

TRAFFIC_LIGHT_SEARCH_TICKS = 350

GREEN_LIGHT_DRIVE_TICKS = 120

CHECK_INTERVAL_TICKS = 5

TICKS_PER_SECOND = 20

CRASH_CAR_FORWARD_DISTANCE = 20.0
CRASH_CAR_SIDE_OFFSET = -14.0

# Placeholder speed for the crash car's scripted turn. There is no live
# CARLA server available while writing this scenario, so this has not been
# tuned against the actual intersection geometry/ego speed at SPAWN_POINT_INDEX.
# Watch a run and adjust this (and/or the trigger settings below) so the
# crash car reaches the ego's lane while the ego is crossing.
CRASH_CAR_SPEED = 14.0

# Rather than starting the crash car's maneuver the instant the light turns
# green, give the ego a head start so it's already moving at speed when the
# crash car pulls out: wait until the ego has traveled
# CRASH_CAR_TRIGGER_DISTANCE_METERS past its position when the light turned
# green, or CRASH_CAR_TRIGGER_MAX_SECONDS has passed since then, whichever
# comes first.
CRASH_CAR_TRIGGER_DISTANCE_METERS = 15.0 * 0.3048  # 15 feet
CRASH_CAR_TRIGGER_MAX_SECONDS = 2.0
CRASH_CAR_TRIGGER_MAX_TICKS = int(CRASH_CAR_TRIGGER_MAX_SECONDS * TICKS_PER_SECOND)

# Distance-box thresholds used for pedestrian/bicyclist hazards don't apply
# to a car-speed hazard, so the crash car uses time-to-collision detection
# instead (see hazard_controls.compute_time_to_collision). This threshold
# is also an untuned starting point.
CRASH_CAR_TTC_LIMIT_SECONDS = 3.5

# Outer bounding gate: hazard detection only evaluates the crash car at all
# once it's within this straight-line distance of the ego (see
# is_actor_in_ego_safety_zone's use_time_to_collision branch in
# hazard_controls.py). Halved from the original 60.0 so ego notices the
# crash car later/closer.
CRASH_CAR_MAX_CHECK_DISTANCE = 21.0


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

    try:
        # -------------------------
        # Connect to CARLA
        # -------------------------

        client, world, bp_lib = connect_to_carla()

        # -------------------------
        # Create output folders
        # -------------------------

        run_folder, images_folder, lidar_folder, gps_log_path = create_run_folder(
            control_type="hard_controls",
            scenario_name="car_crash_intersection",
            use_run_folder=True
        )

        collision_log_path = os.path.join(run_folder, "collision_log.json")

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
        # Attach sensors
        # -------------------------

        sensors = attach_standard_sensors(
            world,
            bp_lib,
            vehicle,
            images_folder,
            lidar_folder,
            gps_data
        )

        spawned_actors.extend(sensors)

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
        # Spawn crash car and precompute its turn path
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
        # Start crash car maneuver after the ego has had a head start on
        # green: wait until it has traveled CRASH_CAR_TRIGGER_DISTANCE_METERS
        # or CRASH_CAR_TRIGGER_MAX_TICKS have passed since green started,
        # whichever comes first, rather than pulling out the instant the
        # light changes.
        # -------------------------

        maneuver_started = [False]
        green_start_location = [None]

        def record_green_start():
            green_start_location[0] = vehicle.get_location()

        def tick_crash_car(tick):
            # Only steer once the maneuver has actually been released;
            # otherwise this would fight the held-still brake applied at
            # spawn during the yellow/red phases.
            if maneuver_started[0]:
                update_crash_car_control(
                    crash_car,
                    crash_car_path,
                    tick,
                    target_speed=CRASH_CAR_SPEED
                )
                return

            if green_start_location[0] is None:
                return  # light hasn't turned green yet

            distance_traveled = vehicle.get_location().distance(
                green_start_location[0]
            )

            if (
                distance_traveled >= CRASH_CAR_TRIGGER_DISTANCE_METERS or
                tick >= CRASH_CAR_TRIGGER_MAX_TICKS
            ):
                start_crash_car_maneuver(
                    crash_car,
                    crash_car_path,
                    speed=CRASH_CAR_SPEED
                )
                maneuver_started[0] = True

        # -------------------------
        # Hazard-aware monitor wrapper
        # -------------------------

        def hazard_aware_monitor(
            world,
            vehicle,
            traffic_light,
            num_ticks,
            light_log,
            action_log,
            check_interval_ticks
        ):
            monitor_light_and_hazards_and_act(
                world,
                vehicle,
                traffic_light,
                num_ticks=num_ticks,
                light_log=light_log,
                action_log=action_log,
                check_interval_ticks=check_interval_ticks,
                hazard_actors=[crash_car],
                hazard_types=["vehicle"],
                use_time_to_collision=True,
                time_to_collision_limit=CRASH_CAR_TTC_LIMIT_SECONDS,
                max_check_distance=CRASH_CAR_MAX_CHECK_DISTANCE,
                on_green_start=record_green_start,
                on_each_tick=tick_crash_car
            )

        # -------------------------
        # Run fixed yellow-red-green sequence
        # -------------------------

        run_fixed_traffic_light_sequence(
            world,
            vehicle,
            traffic_light,
            hazard_aware_monitor,
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


if __name__ == "__main__":
    main()
