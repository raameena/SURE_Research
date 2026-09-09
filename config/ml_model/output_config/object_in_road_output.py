"""Reusable structured output support for CARLA research experiments."""

import csv
import logging
from pathlib import Path

from config.CARLA_actors.carla_config import (
    cleanup_actors,
    create_run_folder,
    save_collision_data,
    save_gps_data,
)


INFERENCE_COLUMNS = [
    "experiment", "scenario", "run_id", "tick", "carla_frame",
    "simulation_time_s", "ego_speed_mps", "ego_acceleration_mps2",
    "ground_truth_object_type", "distance_to_object_m",
    "camera_vehicle_pixels", "camera_pedestrian_pixels",
    "camera_traffic_light_pixels", "bev_vehicle_pixels",
    "bev_walker_pixels", "bev_stop_sign_pixels", "bev_light_green_pixels",
    "bev_light_yellow_pixels", "bev_light_red_pixels", "prob_speed_0",
    "prob_speed_2", "prob_speed_3", "prob_speed_6",
    "pred_target_speed_mps", "decided_action", "previous_action",
    "decision_changed", "throttle", "brake", "steer",
    "target_point_x", "target_point_y", "road_command",
    "expected_perception", "actual_perception", "perception_correct",
    "expected_safe_action", "action_correct", "timely_action", "collision",
    "time_to_collision_s",
]

RUN_COLUMNS = [
    "experiment", "scenario", "run_id", "initial_distance_m",
    "minimum_distance_m", "first_slowdown_tick",
    "first_slowdown_distance_m", "first_stop_tick", "first_stop_distance_m",
    "minimum_ego_speed_mps", "final_action", "decision_change_count",
    "collision_occurred",
]

SCENARIO_OUTPUT_FOLDERS = {
    "Chair": "chair_object",
    "Pedestrian": "pedestrian_object",
    "Pedestrian in Red": "pedestrian_in_red",
    "Vehicle": "vehicle_object",
}

EXPERIMENT_README = """# Object in Road

## Purpose

This experiment evaluates how the driving model perceives and responds to a stationary object placed directly in the ego vehicle's lane.

## Research question

How does object identity affect the model's perception, target-speed prediction, and stopping behavior when the road geometry and starting conditions are held constant?

## Scenario variants

- **Chair:** an out-of-distribution plastic chair.
- **Pedestrian:** an in-distribution stationary pedestrian.
- **Pedestrian in Red:** the same stationary-pedestrian setup with a red clothing color.
- **Vehicle:** an in-distribution stationary vehicle.

## Controlled conditions

The scenarios use the same CARLA map, ego spawn point, initial object distance, simulation frequency, inference interval, camera mount, model configuration, and straight-road approach. The obstacle remains stationary in the ego lane.

## Variable being tested

The principal independent variable is object identity or pedestrian appearance: chair, pedestrian, pedestrian in red, or vehicle.

## Data collected

Run folders are grouped by scenario variant as `chair_object/run_N`, `pedestrian_object/run_N`, `pedestrian_in_red/run_N`, and `vehicle_object/run_N`. Each run records one structured row per inference in `inference_log.csv`, human-readable runtime diagnostics in `scenario_log.txt`, GPS readings, frame-aligned RGB images and LiDAR point clouds, and collision events. Aggregate inference and completed-run tables remain stored at the Object in Road family level in `all_inferences.csv` and `all_runs.csv`.

## Comparing runs

Compare variants using matching scenario settings. Use `all_inferences.csv` for tick-level model, perception, control, and safety measurements. Use `all_runs.csv` for run-level approach and stopping outcomes. Consult each run's README for its exact map, blueprint, weather, and settings.
"""

DATA_DICTIONARY = """# Object in Road Data Dictionary

Blank CSV fields mean that a value was unavailable or not applicable. Boolean values are written as `True` or `False`.

## Inference tables

`inference_log.csv` contains one row for every inference attempt in one run. `all_inferences.csv` contains the same rows appended across all Object in Road runs.

| Column | Datatype | Units | Description |
| --- | --- | --- | --- |
| experiment | string | — | Experiment family; `Object in Road`. |
| scenario | string | — | Scenario variant: Chair, Pedestrian, Pedestrian in Red, or Vehicle. |
| run_id | string | — | Shared experiment run identifier, such as `run_8`. |
| tick | integer | tick | Scenario-local control-loop tick, beginning at zero. |
| carla_frame | integer | frame | CARLA frame used by the model camera and LiDAR inputs. |
| simulation_time_s | float | s | Scenario-local simulated time, computed from tick and fixed simulation step. |
| ego_speed_mps | float | m/s | Actual CARLA ego vehicle speed, separate from model target speed. |
| ego_acceleration_mps2 | float | m/s² | Magnitude of CARLA's ego acceleration vector at inference time. |
| ground_truth_object_type | string | — | Actual obstacle variant placed in the road. |
| distance_to_object_m | float | m | Ground-truth distance from ego to the obstacle's near surface. |
| camera_vehicle_pixels | integer | pixels | Semantic-image pixels assigned to vehicle; this is not an object count. |
| camera_pedestrian_pixels | integer | pixels | Semantic-image pixels assigned to pedestrian; this is not an object count. |
| camera_traffic_light_pixels | integer | pixels | Semantic-image pixels assigned to traffic light; this is not an object count. |
| bev_vehicle_pixels | integer | BEV cells/pixels | Semantic BEV cells assigned to vehicle; this is not an object count. |
| bev_walker_pixels | integer | BEV cells/pixels | Semantic BEV cells assigned to walker/pedestrian. |
| bev_stop_sign_pixels | integer | BEV cells/pixels | Semantic BEV cells assigned to stop sign. |
| bev_light_green_pixels | integer | BEV cells/pixels | Semantic BEV cells assigned to green traffic light. |
| bev_light_yellow_pixels | integer | BEV cells/pixels | Semantic BEV cells assigned to yellow traffic light. |
| bev_light_red_pixels | integer | BEV cells/pixels | Semantic BEV cells assigned to red traffic light. |
| prob_speed_0 | float | probability [0,1] | Probability assigned to the 0 m/s target-speed bin. |
| prob_speed_2 | float | probability [0,1] | Probability assigned to the 2 m/s target-speed bin. |
| prob_speed_3 | float | probability [0,1] | Probability assigned to the 3 m/s target-speed bin. |
| prob_speed_6 | float | probability [0,1] | Probability assigned to the 6 m/s target-speed bin. |
| pred_target_speed_mps | float | m/s | PCLA-processed target speed: brake-bin threshold or probability-weighted expected speed. |
| decided_action | string | — | High-level action decoded from the model target speed. |
| previous_action | string | — | Action held before this inference result. |
| decision_changed | boolean | — | Whether `decided_action` differs from `previous_action`. |
| throttle | float | [0,1] | Newly issued PCLA-controller throttle command for this inference row. |
| brake | float | [0,1] | Newly issued PCLA-controller service-brake command for this inference row. |
| steer | float | [-1,1] | Newly issued PCLA-controller steering command for this inference row. |
| target_point_x | float | m | Active world-route target in ego-local coordinates; positive is forward. |
| target_point_y | float | m | Active world-route target in ego-local coordinates; positive is right. |
| road_command | string | command | Lagged PCLA RoadOption command supplied to the model. |
| expected_perception | string | — | Ground-truth perception label expected at the measured distance. |
| actual_perception | string | — | Perception label resolved from the model's own output. |
| perception_correct | boolean | — | Whether actual and expected perception labels match. |
| expected_safe_action | string | — | Safe action expected from scenario ground truth. |
| action_correct | boolean | — | Whether the decoded action matches the expected safe action. |
| timely_action | boolean | — | Whether a correct action was issued with sufficient physical stopping distance; blank when not applicable. |
| collision | boolean | — | Whether a collision event occurred since the preceding inference row. |
| time_to_collision_s | float | s | Estimated center-to-center time to collision from current relative motion; blank when actors are not closing. |

The `camera_*_pixels` fields count semantic-image pixels assigned to a class. They are **not object counts**. The `bev_*_pixels` fields count semantic BEV cells/pixels assigned to a class. Raw counts are preserved even when below the model's detection threshold.

## Completed-run table

`all_runs.csv` contains one row per successfully completed run.

| Column | Datatype | Units | Description |
| --- | --- | --- | --- |
| experiment | string | — | Experiment family. |
| scenario | string | — | Scenario variant. |
| run_id | string | — | Run folder identifier. |
| initial_distance_m | float | m | Configured starting object distance. |
| minimum_distance_m | float | m | Smallest object distance observed at an inference check. |
| first_slowdown_tick | integer | tick | First tick whose decided action was `slow`; blank if absent. |
| first_slowdown_distance_m | float | m | Object distance at the first `slow` decision. |
| first_stop_tick | integer | tick | First tick whose action was `stop` or `hard_brake`; blank if absent. |
| first_stop_distance_m | float | m | Object distance at the first stop/hard-brake decision. |
| minimum_ego_speed_mps | float | m/s | Smallest actual ego speed observed at inference checks. |
| final_action | string | — | Last action recorded by an inference row. |
| decision_change_count | integer | changes | Number of inference rows that changed the held action. |
| collision_occurred | boolean | — | Whether the collision sensor recorded any event during the run. |
"""


def _write_if_missing(path, contents):
    path = Path(path)
    if not path.exists():
        path.write_text(contents, encoding="utf-8")


def _append_csv(path, fieldnames, row):
    path = Path(path)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow({name: row.get(name) for name in fieldnames})


def _ensure_csv_header(path, fieldnames):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        csv.DictWriter(csv_file, fieldnames=fieldnames).writeheader()


def _format_number(value, decimals=2, suffix=""):
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"


class ExperimentRunOutput:
    """Owns files and readable logging for one Object in Road run."""

    def __init__(self, scenario, scenario_description, metadata, base_folder="output"):
        self.experiment = "Object in Road"
        self.scenario = scenario
        self.scenario_description = scenario_description
        self.metadata = dict(metadata)
        self.rows = []
        self.completed = False

        try:
            scenario_folder_name = SCENARIO_OUTPUT_FOLDERS[scenario]
        except KeyError as error:
            raise ValueError(
                f"Unknown Object in Road scenario {scenario!r}; expected one of "
                f"{sorted(SCENARIO_OUTPUT_FOLDERS)}."
            ) from error

        paths = create_run_folder(
            base_folder=base_folder,
            control_type="model",
            scenario_group="object_in_road",
            scenario_name=scenario_folder_name,
            use_run_folder=True,
        )
        self.run_folder, self.images_folder, self.lidar_folder, self.gps_log_path = paths
        self.run_folder = Path(self.run_folder)
        self.images_folder = Path(self.images_folder)
        self.lidar_folder = Path(self.lidar_folder)
        self.gps_log_path = Path(self.gps_log_path)
        self.scenario_folder = self.run_folder.parent
        self.experiment_folder = self.scenario_folder.parent
        self.run_id = self.run_folder.name
        self.run_number = int(self.run_id.removeprefix("run_"))
        self.inference_log_path = self.run_folder / "inference_log.csv"
        self.all_inferences_path = self.experiment_folder / "all_inferences.csv"
        self.all_runs_path = self.experiment_folder / "all_runs.csv"
        self.scenario_log_path = self.run_folder / "scenario_log.txt"
        self.run_readme_path = self.run_folder / "README.md"
        self.collision_log_path = self.run_folder / "collision_log.json"

        _write_if_missing(self.experiment_folder / "README.md", EXPERIMENT_README)
        _write_if_missing(self.experiment_folder / "DATA_DICTIONARY.md", DATA_DICTIONARY)

        # Create CSVs immediately so a failed/empty run still has an inspectable schema.
        _ensure_csv_header(self.inference_log_path, INFERENCE_COLUMNS)
        _ensure_csv_header(self.all_inferences_path, INFERENCE_COLUMNS)
        _ensure_csv_header(self.all_runs_path, RUN_COLUMNS)

        self.file_handler = logging.FileHandler(self.scenario_log_path, encoding="utf-8")
        self.file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(self.file_handler)
        self.write_run_readme()

    def update_metadata(self, **values):
        self.metadata.update(values)
        self.write_run_readme()

    def write_run_readme(self):
        comparison = self.metadata.get(
            "comparison",
            ["Chair", "Pedestrian", "Pedestrian in Red", "Vehicle"],
        )
        notes = self.metadata.get("notes") or "No additional run-specific notes."
        object_type = self.metadata.get("object_type", "Pending actor spawn")
        contents = f"""# Run {self.run_number} — {self.scenario}

## Experiment

{self.experiment}

## Scenario

{self.scenario_description}

## Setup

- CARLA Map: {self.metadata.get('map_name', 'Pending CARLA connection')}
- Model: {self.metadata.get('model_name', 'N/A')}
- Ego Spawn Point: {self.metadata.get('ego_spawn_point', 'N/A')}
- Object Type: {object_type}
- Starting Object Distance: {_format_number(self.metadata.get('initial_distance_m'), suffix=' m')}
- Weather: {self.metadata.get('weather', 'N/A')}
- Simulation Frequency: {self.metadata.get('simulation_frequency_hz', 'N/A')} Hz
- Model Inference Interval: Every {self.metadata.get('inference_interval_ticks', 'N/A')} ticks

## Variable Being Tested

{self.metadata.get('variable_being_tested', 'Object identity')}

## Comparison

""" + "\n".join(f"- {item}" for item in comparison) + f"""

## Notes

{notes}
"""
        self.run_readme_path.write_text(contents, encoding="utf-8")

    def log_startup(self):
        separator = "=" * 61
        logging.info(
            "\n%s\n                    CARLA EXPERIMENT\n%s\n"
            "Experiment:  %s\nScenario:    %s\nRun:         %d\n\n"
            "Map:         %s\nModel:       %s\nEgo Spawn:   %s\n\n"
            "Object:      %s\nDistance:    %.1f m\nLane:        %s\n\n"
            "Sensors\n  Recording: RGB, LiDAR, GPS, Collision\n"
            "  Model:     Camera, LiDAR\n\nSimulation\n"
            "  Frequency:       %s Hz\n  Model Inference: Every %s ticks\n%s\n"
            "RUN STARTED\n%s",
            separator, separator, self.experiment, self.scenario, self.run_number,
            self.metadata.get("map_name", "N/A"), self.metadata.get("model_name", "N/A"),
            self.metadata.get("ego_spawn_point", "N/A"),
            self.metadata.get("object_type", "N/A"), self.metadata.get("initial_distance_m", 0.0),
            self.metadata.get("lane_id", "N/A"), self.metadata.get("simulation_frequency_hz", "N/A"),
            self.metadata.get("inference_interval_ticks", "N/A"), separator, separator,
        )
        logging.warning(
            "PCLA fidelity note: model inference currently runs every 5 ticks; "
            "original PCLA uses every tick."
        )

    def record_inference(self, raw_row):
        probabilities = raw_row.get("target_speed_probabilities") or {}
        class_scores = raw_row.get("class_scores") or {}
        camera_scores = class_scores.get("pred_semantic", {})
        bev_scores = class_scores.get("pred_bev_semantic", {})

        def probability_for(speed):
            return next(
                (value for key, value in probabilities.items() if abs(float(key) - speed) < 1e-6),
                None,
            )

        row = {
            "experiment": self.experiment,
            "scenario": self.scenario,
            "run_id": self.run_id,
            "ground_truth_object_type": self.scenario.lower(),
            "tick": raw_row.get("tick"),
            "carla_frame": raw_row.get("carla_frame"),
            "simulation_time_s": raw_row.get("simulation_time_s"),
            "ego_speed_mps": raw_row.get("ego_speed_mps"),
            "ego_acceleration_mps2": raw_row.get("ego_acceleration_mps2"),
            "distance_to_object_m": raw_row.get("distance_to_object_m"),
            "camera_vehicle_pixels": camera_scores.get("vehicle"),
            "camera_pedestrian_pixels": camera_scores.get("pedestrian"),
            "camera_traffic_light_pixels": camera_scores.get("traffic_light"),
            "bev_vehicle_pixels": bev_scores.get("vehicle"),
            "bev_walker_pixels": bev_scores.get("walker"),
            "bev_stop_sign_pixels": bev_scores.get("stop_sign"),
            "bev_light_green_pixels": bev_scores.get("traffic_light_green"),
            "bev_light_yellow_pixels": bev_scores.get("traffic_light_yellow"),
            "bev_light_red_pixels": bev_scores.get("traffic_light_red"),
            "prob_speed_0": probability_for(0.0),
            "prob_speed_2": probability_for(2.0),
            "prob_speed_3": probability_for(3.0),
            "prob_speed_6": probability_for(6.0),
            "pred_target_speed_mps": raw_row.get("pred_target_speed"),
            "decided_action": raw_row.get("decided_action"),
            "previous_action": raw_row.get("previous_command"),
            "decision_changed": raw_row.get("changed"),
            "throttle": raw_row.get("throttle"),
            "brake": raw_row.get("brake"),
            "steer": raw_row.get("steer"),
            "target_point_x": raw_row.get("target_point_x"),
            "target_point_y": raw_row.get("target_point_y"),
            "road_command": raw_row.get("road_command"),
            "expected_perception": raw_row.get("expected_perception"),
            "actual_perception": raw_row.get("actual_perception"),
            "perception_correct": raw_row.get("perceieved_right"),
            "expected_safe_action": raw_row.get("expected_safe_action"),
            "action_correct": raw_row.get("action_correct"),
            "timely_action": raw_row.get("timely_action"),
            "collision": raw_row.get("collision"),
            "time_to_collision_s": raw_row.get("time_to_collision_s"),
        }
        self.rows.append(row)
        _append_csv(self.inference_log_path, INFERENCE_COLUMNS, row)
        _append_csv(self.all_inferences_path, INFERENCE_COLUMNS, row)

    def complete(self, collision_occurred, saved_items):
        if self.completed:
            return

        usable_distances = [r.get("distance_to_object_m") for r in self.rows if r.get("distance_to_object_m") is not None]
        usable_speeds = [r.get("ego_speed_mps") for r in self.rows if r.get("ego_speed_mps") is not None]
        first_slow = next((r for r in self.rows if r.get("decided_action") == "slow"), None)
        first_stop = next((r for r in self.rows if r.get("decided_action") in {"stop", "hard_brake"}), None)
        final_action = self.rows[-1].get("decided_action") if self.rows else None

        summary = {
            "experiment": self.experiment,
            "scenario": self.scenario,
            "run_id": self.run_id,
            "initial_distance_m": self.metadata.get("initial_distance_m"),
            "minimum_distance_m": min(usable_distances) if usable_distances else None,
            "first_slowdown_tick": first_slow.get("tick") if first_slow else None,
            "first_slowdown_distance_m": first_slow.get("distance_to_object_m") if first_slow else None,
            "first_stop_tick": first_stop.get("tick") if first_stop else None,
            "first_stop_distance_m": first_stop.get("distance_to_object_m") if first_stop else None,
            "minimum_ego_speed_mps": min(usable_speeds) if usable_speeds else None,
            "final_action": final_action,
            "decision_change_count": sum(bool(r.get("decision_changed")) for r in self.rows),
            "collision_occurred": bool(collision_occurred),
        }
        _append_csv(self.all_runs_path, RUN_COLUMNS, summary)
        self.completed = True

        duration_s = self.metadata.get("num_ticks", 0) / self.metadata.get("simulation_frequency_hz", 1)
        separator = "=" * 61
        saved_text = "\n".join(f"  {item}" for item in saved_items)
        slow_text = "N/A" if first_slow is None else f"Tick {first_slow['tick']} @ {first_slow['distance_to_object_m']:.2f} m"
        stop_text = "N/A" if first_stop is None else f"Tick {first_stop['tick']} @ {first_stop['distance_to_object_m']:.2f} m"
        logging.info(
            "\n%s\n                       RUN COMPLETE\n%s\n"
            "Experiment:           %s\nScenario:             %s\nRun:                  %d\n\n"
            "Duration:             %.2f s\nInference Checks:     %d\nDecision Changes:     %d\n\n"
            "Initial Distance:     %s\nMinimum Distance:     %s\n\n"
            "First Slowdown:       %s\nFirst Stop:           %s\n\n"
            "Final Action:         %s\nCollision:            %s\n\nSaved\n%s\n%s",
            separator, separator, self.experiment, self.scenario, self.run_number,
            duration_s, len(self.rows), summary["decision_change_count"],
            _format_number(summary["initial_distance_m"], suffix=" m"),
            _format_number(summary["minimum_distance_m"], suffix=" m"),
            slow_text, stop_text, (final_action or "N/A").upper(),
            "Yes" if collision_occurred else "No", saved_text, separator,
        )

    def finalize(self, run_completed, gps_data, collision_data, spawned_actors):
        """Persist final sensor logs, clean up actors, summarize, and close."""
        gps_saved = save_gps_data(gps_data, self.gps_log_path)
        collision_saved = save_collision_data(collision_data, self.collision_log_path)
        cleanup_actors(spawned_actors)

        try:
            if run_completed:
                saved_items = [
                    "README.md", "inference_log.csv", "all_inferences.csv",
                    "all_runs.csv", "scenario_log.txt",
                ]
                if gps_saved:
                    saved_items.append("gps_log.json")
                if collision_saved:
                    saved_items.append("collision_log.json")
                if any(self.images_folder.glob("*.png")):
                    saved_items.append("RGB images")
                if any(self.lidar_folder.glob("*.ply")):
                    saved_items.append("LiDAR point clouds")
                self.complete(bool(collision_data), saved_items)
        finally:
            self.close()

    def close(self):
        if self.file_handler is not None:
            self.file_handler.close()
            logging.getLogger().removeHandler(self.file_handler)
            self.file_handler = None


def create_object_in_road_output(
    scenario,
    scenario_description,
    world,
    model_name,
    ego_spawn_point,
    initial_distance_m,
    simulation_frequency_hz,
    inference_interval_ticks,
    num_ticks,
):
    """Build one run output object from the shared scenario settings."""
    return ExperimentRunOutput(
        scenario=scenario,
        scenario_description=scenario_description,
        metadata={
            "map_name": world.get_map().name.rsplit("/", 1)[-1],
            "model_name": model_name,
            "ego_spawn_point": ego_spawn_point,
            "initial_distance_m": initial_distance_m,
            "weather": str(world.get_weather()),
            "simulation_frequency_hz": simulation_frequency_hz,
            "inference_interval_ticks": inference_interval_ticks,
            "num_ticks": num_ticks,
            "variable_being_tested": "Object identity or pedestrian appearance",
        },
    )
