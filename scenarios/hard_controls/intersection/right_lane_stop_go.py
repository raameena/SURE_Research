import sys
import logging
from pathlib import Path

# Add the main CARLA-Research folder to Python's import path
# right_lane_stop_go.py is inside: scenarios/hard_controls/intersection/
# parents[3] goes back to: CARLA-Research/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from config.CARLA_actors.carla_config import (
    connect_to_carla,
    create_run_folder,
    spawn_ego_vehicle_right_lane_of_index,
    save_gps_data,
    cleanup_actors
)

from config.CARLA_actors.sensors import (
    attach_standard_sensors
)

from controls.light_controls import (
    find_vehicle_traffic_light,
    run_fixed_traffic_light_sequence,
    unfreeze_traffic_light
)

from controls.hard_controls import (
    monitor_light_and_act
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------
# Scenario settings
# -------------------------

BASE_SPAWN_POINT_INDEX = 11

TRAFFIC_LIGHT_SEARCH_TICKS = 350

GREEN_LIGHT_DRIVE_TICKS = 100

CHECK_INTERVAL_TICKS = 5


# -------------------------
# Main scenario
# -------------------------

def main():
    spawned_actors = []

    gps_data = []
    light_log = []
    action_log = []

    traffic_light = None

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
            scenario_name="right_lane_stop_go",
            use_run_folder=True
        )

        # -------------------------
        # Spawn ego vehicle in right lane
        # -------------------------

        vehicle = spawn_ego_vehicle_right_lane_of_index(
            world,
            bp_lib,
            spawn_index=BASE_SPAWN_POINT_INDEX
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
                "No traffic light was detected as affecting the ego vehicle. "
                "Try using a different spawn point or increasing max_ticks."
            )

        # -------------------------
        # Run fixed yellow-red-green sequence
        # -------------------------

        run_fixed_traffic_light_sequence(
            world,
            vehicle,
            traffic_light,
            monitor_light_and_act,
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