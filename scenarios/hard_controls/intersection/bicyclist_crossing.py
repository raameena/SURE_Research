import sys
import logging
from pathlib import Path

# Add the main CARLA-Research folder to Python's import path
# bicyclist_crossing.py is inside: scenarios/hard_controls/intersection/
# parents[3] goes back to: CARLA-Research/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from config.CARLA_actors.carla_config import (
    connect_to_carla,
    create_run_folder,
    spawn_ego_vehicle_at_index,
    save_gps_data,
    cleanup_actors
)

from config.CARLA_actors.sensors import (
    attach_standard_sensors
)

from config.CARLA_actors.bicyclist import (
    spawn_bicyclist_relative_to_vehicle,
    start_bicyclist_crossing,
    stop_bicyclist
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

BICYCLIST_FORWARD_DISTANCE = 12.0
BICYCLIST_SIDE_OFFSET = -6.0
BICYCLIST_SPEED = 6.0


# -------------------------
# Main scenario
# -------------------------

def main():
    spawned_actors = []

    gps_data = []
    light_log = []
    action_log = []

    traffic_light = None
    bicyclist = None

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
            scenario_name="bicyclist_crossing",
            use_run_folder=True
        )

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
        # Spawn bicyclist
        # -------------------------

        bicyclist = spawn_bicyclist_relative_to_vehicle(
            world,
            bp_lib,
            vehicle,
            forward_distance=BICYCLIST_FORWARD_DISTANCE,
            side_offset=BICYCLIST_SIDE_OFFSET,
            crossing_direction="left_to_right"
        )

        spawned_actors.append(bicyclist)

        world.wait_for_tick()

        # -------------------------
        # Start bicyclist crossing when light turns green
        # -------------------------

        def start_crossing_when_green():
            start_bicyclist_crossing(
                bicyclist,
                vehicle,
                speed=BICYCLIST_SPEED,
                crossing_direction="left_to_right"
            )

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
                hazard_actors=[bicyclist],
                hazard_types=["bicyclist"],
                forward_distance_limit=20.0,
                side_distance_limit=4.0,
                on_green_start=start_crossing_when_green
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
            ticks_per_second=20,
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
        # Stop bicyclist
        # -------------------------

        try:
            if bicyclist is not None:
                stop_bicyclist(bicyclist)

        except Exception as bicyclist_error:
            logging.warning(f"Could not stop bicyclist: {bicyclist_error}")

        # -------------------------
        # Unfreeze traffic light
        # -------------------------

        try:
            if traffic_light is not None:
                unfreeze_traffic_light(traffic_light)

        except Exception as light_error:
            logging.warning(f"Could not unfreeze traffic light: {light_error}")

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


if __name__ == "__main__":
    main()