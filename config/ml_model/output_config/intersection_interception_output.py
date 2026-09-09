"""Structured output for the Intersection Interception experiment family."""

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
    "camera_vehicle_pixels", "camera_pedestrian_pixels",
    "camera_traffic_light_pixels", "bev_vehicle_pixels", "bev_walker_pixels",
    "bev_stop_sign_pixels", "bev_light_green_pixels", "bev_light_yellow_pixels",
    "bev_light_red_pixels", "prob_speed_0", "prob_speed_2", "prob_speed_3",
    "prob_speed_6", "pred_target_speed_mps", "decided_action",
    "previous_action", "decision_changed", "throttle", "brake", "steer",
    "expected_perception", "actual_perception", "perception_correct",
    "expected_safe_action", "action_correct", "timely_action", "collision",
    "distance_to_crossing_vehicle_m", "crossing_vehicle_speed_mps",
    "time_to_collision_s",
]

RUN_COLUMNS = [
    "experiment", "scenario", "run_id", "interception_severity",
    "trigger_distance_m", "trigger_max_seconds", "actual_trigger_tick",
    "actual_trigger_distance_m", "initial_ego_speed_mps",
    "first_vehicle_detection_tick", "first_vehicle_detection_distance_m",
    "first_slowdown_tick", "first_slowdown_distance_m", "first_stop_tick",
    "first_stop_distance_m", "first_hard_brake_tick",
    "first_hard_brake_distance_m", "minimum_distance_to_crossing_vehicle_m",
    "minimum_time_to_collision_s", "minimum_ego_speed_mps", "final_action",
    "decision_change_count", "collision_occurred",
]

EXPERIMENT_README = """# Intersection Interception Experiment

## Research question

How does model behavior change as an unexpected cut-in vehicle turns into the ego lane with decreasing reaction margin?

## Scenarios

- Early Interception
- Medium Interception
- Late Interception

## Controlled variables

The ego spawn, map, model, sensors, weather, simulation rate, model inference interval, interception vehicle type, interception vehicle speed, left-turn merge path, fixed green ego traffic light, and fixed red non-ego intersection signals are held constant.

## Primary changed variable

The cut-in vehicle's release trigger changes the interception timing. Early provides the largest nominal reaction margin, Medium a moderate margin, and Late the smallest. The vehicle uses `interception_car.py` logic: it turns into the ego lane and stops after reaching the end of the merge path. Actual gap, speed, and TTC must be used when comparing runs because the model-controlled ego can vary.

## Data collected

The experiment records model target speed and probabilities, semantic outputs, actual ego speed and controls, longitudinal bumper gap to the interception vehicle, interception-vehicle speed, TTC when physically meaningful, and collision outcome. RGB and LiDAR filenames use the CARLA frame number recorded in each inference row.

Collision is an observed outcome, not a scripted goal. The ego remains controlled by the ML model; no emergency response is injected by the scenario.
"""


_INFERENCE_DESCRIPTIONS = {
    "experiment": ("string", "—", "Experiment family; Intersection Interception."),
    "scenario": ("string", "—", "Early, Medium, or Late Interception variant."),
    "run_id": ("string", "—", "Run folder identifier, such as run_3."),
    "tick": ("integer", "tick", "Scenario-local control-loop tick."),
    "carla_frame": ("integer", "frame", "CARLA frame shared with RGB/LiDAR filenames."),
    "simulation_time_s": ("float", "s", "Scenario-local simulated time."),
    "ego_speed_mps": ("float", "m/s", "Actual ego speed at inference."),
    "ego_acceleration_mps2": ("float", "m/s²", "Magnitude of ego acceleration."),
    "camera_vehicle_pixels": ("integer", "pixels", "Camera semantic pixels classified as vehicle."),
    "camera_pedestrian_pixels": ("integer", "pixels", "Camera semantic pixels classified as pedestrian."),
    "camera_traffic_light_pixels": ("integer", "pixels", "Camera semantic pixels classified as traffic light."),
    "bev_vehicle_pixels": ("integer", "BEV cells", "BEV cells classified as vehicle."),
    "bev_walker_pixels": ("integer", "BEV cells", "BEV cells classified as walker."),
    "bev_stop_sign_pixels": ("integer", "BEV cells", "BEV cells classified as stop sign."),
    "bev_light_green_pixels": ("integer", "BEV cells", "BEV cells classified as green traffic light."),
    "bev_light_yellow_pixels": ("integer", "BEV cells", "BEV cells classified as yellow traffic light."),
    "bev_light_red_pixels": ("integer", "BEV cells", "BEV cells classified as red traffic light."),
    "prob_speed_0": ("float", "probability", "Probability of target speed 0 m/s."),
    "prob_speed_2": ("float", "probability", "Probability of target speed 2 m/s."),
    "prob_speed_3": ("float", "probability", "Probability of target speed 3 m/s."),
    "prob_speed_6": ("float", "probability", "Probability of target speed 6 m/s."),
    "pred_target_speed_mps": ("float", "m/s", "Model-selected target speed."),
    "decided_action": ("string", "—", "Decoded go, slow, stop, or hard_brake action."),
    "previous_action": ("string", "—", "Action held before this inference."),
    "decision_changed": ("boolean", "—", "Whether the decoded action changed."),
    "throttle": ("float", "[0,1]", "Applied CARLA throttle."),
    "brake": ("float", "[0,1]", "Applied CARLA brake."),
    "steer": ("float", "[-1,1]", "Applied CARLA steering."),
    "expected_perception": ("string", "—", "Ground-truth perception label."),
    "actual_perception": ("string", "—", "Perception resolved from model BEV output."),
    "perception_correct": ("boolean", "—", "Whether actual matches expected perception."),
    "expected_safe_action": ("string", "—", "Scenario ground-truth safe action."),
    "action_correct": ("boolean", "—", "Whether model action matches expected action."),
    "timely_action": ("boolean", "—", "Stopping-distance timing result when applicable."),
    "collision": ("boolean", "—", "Collision sensor event since the previous inference row."),
    "distance_to_crossing_vehicle_m": ("float", "m", "Signed longitudinal gap from ego's front bumper to the interception vehicle's rear bumper, using interception_car.py logic; it may be negative while the vehicle is beside or behind ego."),
    "crossing_vehicle_speed_mps": ("float", "m/s", "Actual interception-vehicle speed."),
    "time_to_collision_s": ("float", "s", "Relative-motion TTC; blank when actors are not closing."),
}

_RUN_DESCRIPTIONS = {
    "experiment": ("string", "—", "Experiment family."),
    "scenario": ("string", "—", "Scenario variant."),
    "run_id": ("string", "—", "Run folder identifier."),
    "interception_severity": ("string", "—", "EARLY, MEDIUM, or LATE."),
    "trigger_distance_m": ("float", "m", "Configured ego travel distance before release."),
    "trigger_max_seconds": ("float", "s", "Shared timeout fallback for crossing-car release."),
    "actual_trigger_tick": ("integer", "tick", "Tick on which the crossing car was released."),
    "actual_trigger_distance_m": ("float", "m", "Actual ego travel at crossing-car release."),
    "initial_ego_speed_mps": ("float", "m/s", "Ego speed at first inference."),
    "first_vehicle_detection_tick": ("integer", "tick", "First inference resolving vehicle presence."),
    "first_vehicle_detection_distance_m": ("float", "m", "Vehicle distance at first detection."),
    "first_slowdown_tick": ("integer", "tick", "First slow action."),
    "first_slowdown_distance_m": ("float", "m", "Vehicle distance at first slow action."),
    "first_stop_tick": ("integer", "tick", "First stop action, excluding hard_brake."),
    "first_stop_distance_m": ("float", "m", "Vehicle distance at first stop action."),
    "first_hard_brake_tick": ("integer", "tick", "First hard_brake action."),
    "first_hard_brake_distance_m": ("float", "m", "Vehicle distance at first hard brake."),
    "minimum_distance_to_crossing_vehicle_m": ("float", "m", "Smallest recorded signed longitudinal bumper gap."),
    "minimum_time_to_collision_s": ("float", "s", "Smallest valid relative-motion TTC."),
    "minimum_ego_speed_mps": ("float", "m/s", "Smallest recorded actual ego speed."),
    "final_action": ("string", "—", "Last recorded decoded action."),
    "decision_change_count": ("integer", "changes", "Number of action changes."),
    "collision_occurred": ("boolean", "—", "Whether any collision event was recorded."),
}


def _data_dictionary():
    def table(columns, descriptions):
        lines = ["| Column | Datatype | Units | Meaning |", "| --- | --- | --- | --- |"]
        for column in columns:
            datatype, units, meaning = descriptions[column]
            lines.append(f"| {column} | {datatype} | {units} | {meaning} |")
        return "\n".join(lines)

    return (
        "# Intersection Interception Data Dictionary\n\n"
        "Blank fields mean a value was unavailable or not physically applicable. "
        "Pixel fields are class-assigned pixel/cell counts, not object counts.\n\n"
        "## Inference tables\n\n`inference_log.csv` is one run; "
        "`all_inferences.csv` contains the same rows across runs.\n\n"
        + table(INFERENCE_COLUMNS, _INFERENCE_DESCRIPTIONS)
        + "\n\n## Completed-run table\n\n`all_runs.csv` contains one row per completed run.\n\n"
        + table(RUN_COLUMNS, _RUN_DESCRIPTIONS)
        + "\n"
    )


def _write_if_missing(path, contents):
    path = Path(path)
    if not path.exists():
        path.write_text(contents, encoding="utf-8")


def _ensure_csv(path, columns):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=columns).writeheader()


def _append_csv(path, columns, row):
    with Path(path).open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore").writerow(
            {column: row.get(column) for column in columns}
        )


def _number(value, suffix=""):
    return "N/A" if value is None else f"{value:.2f}{suffix}"


def _event_text(row):
    if row is None:
        return "N/A"
    return f"Tick {row['tick']} @ {row['distance_to_crossing_vehicle_m']:.2f} m"


class IntersectionInterceptionRunOutput:
    """Own run files, streaming CSV rows, summaries, and readable logs."""

    def __init__(self, scenario, description, metadata, base_folder="output"):
        self.experiment = "Intersection Interception"
        self.scenario = scenario
        self.description = description
        self.metadata = dict(metadata)
        self.rows = []
        self.completed = False

        paths = create_run_folder(
            base_folder=base_folder,
            control_type="model",
            scenario_name="intersection_interception",
            use_run_folder=True,
        )
        self.run_folder, self.images_folder, self.lidar_folder, self.gps_log_path = map(Path, paths)
        self.experiment_folder = self.run_folder.parent
        self.run_id = self.run_folder.name
        self.run_number = int(self.run_id.removeprefix("run_"))
        self.inference_log_path = self.run_folder / "inference_log.csv"
        self.all_inferences_path = self.experiment_folder / "all_inferences.csv"
        self.all_runs_path = self.experiment_folder / "all_runs.csv"
        self.scenario_log_path = self.run_folder / "scenario_log.txt"
        self.run_readme_path = self.run_folder / "README.md"
        self.collision_log_path = self.run_folder / "collision_log.json"

        _write_if_missing(self.experiment_folder / "README.md", EXPERIMENT_README)
        _write_if_missing(self.experiment_folder / "DATA_DICTIONARY.md", _data_dictionary())
        _ensure_csv(self.inference_log_path, INFERENCE_COLUMNS)
        _ensure_csv(self.all_inferences_path, INFERENCE_COLUMNS)
        _ensure_csv(self.all_runs_path, RUN_COLUMNS)

        self.file_handler = logging.FileHandler(self.scenario_log_path, encoding="utf-8")
        self.file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(self.file_handler)
        self.write_run_readme()

    def update_metadata(self, **values):
        self.metadata.update(values)
        self.write_run_readme()

    def write_run_readme(self):
        trigger = (
            f"ego travel >= {self.metadata['trigger_distance_m']:.3f} m "
            f"or {self.metadata['trigger_max_seconds']:.1f} s timeout"
        )
        contents = f"""# Run {self.run_number} — {self.scenario}

## Experiment

{self.experiment}

## Scenario

{self.description}

## Setup

- Run ID: {self.run_id}
- CARLA Map: {self.metadata.get('map_name', 'N/A')}
- Model: {self.metadata.get('model_name', 'N/A')}
- Ego Spawn Point: {self.metadata.get('ego_spawn_point', 'N/A')}
- Ego Traffic Light State: GREEN (fixed during experiment)
- Other Intersection Signals: RED (fixed during experiment)
- Interception Severity: {self.metadata.get('interception_severity', 'N/A')}
- Trigger Configuration: {trigger}
- Interception Vehicle: {self.metadata.get('crossing_vehicle_type', 'Pending actor spawn')}
- Interception Vehicle Target Speed: {self.metadata.get('crossing_vehicle_speed_mps', 'N/A')} m/s
- Simulation Frequency: {self.metadata.get('simulation_frequency_hz', 'N/A')} Hz
- Model Inference Interval: Every {self.metadata.get('inference_interval_ticks', 'N/A')} ticks
- Weather: {self.metadata.get('weather', 'N/A')}

## Relevant notes

{self.metadata.get('notes', 'Trigger values are initial tuning parameters requiring live CARLA validation. Collision is observed, not forced.')}
"""
        self.run_readme_path.write_text(contents, encoding="utf-8")

    def log_startup(self):
        separator = "=" * 61
        logging.info(
            "\n%s\n                    CARLA EXPERIMENT\n%s\n"
            "Experiment:  %s\nScenario:    %s\nRun:         %d\n\n"
            "Map:         %s\nModel:       %s\nEgo Spawn:   %s\n\n"
            "Traffic Light: GREEN (fixed)\n\nInterception Vehicle\n"
            "  Type:       %s\n  Speed:      %.2f m/s\n  Severity:   %s\n"
            "  Trigger:    ego travel >= %.3f m (timeout %.1f s)\n\n"
            "Sensors\n  Recording: RGB, LiDAR, GPS, Collision\n"
            "  Model:     Camera, LiDAR\n\nSimulation\n"
            "  Frequency:       %s Hz\n  Model Inference: Every %s ticks\n%s\n"
            "RUN STARTED\n%s",
            separator, separator, self.experiment, self.scenario, self.run_number,
            self.metadata.get("map_name", "N/A"), self.metadata.get("model_name", "N/A"),
            self.metadata.get("ego_spawn_point", "N/A"),
            self.metadata.get("crossing_vehicle_type", "N/A"),
            self.metadata.get("crossing_vehicle_speed_mps", 0.0),
            self.metadata.get("interception_severity", "N/A"),
            self.metadata.get("trigger_distance_m", 0.0),
            self.metadata.get("trigger_max_seconds", 0.0),
            self.metadata.get("simulation_frequency_hz", "N/A"),
            self.metadata.get("inference_interval_ticks", "N/A"), separator, separator,
        )

    def record_inference(self, raw_row):
        probabilities = raw_row.get("target_speed_probabilities") or {}
        scores = raw_row.get("class_scores") or {}
        camera = scores.get("pred_semantic", {})
        bev = scores.get("pred_bev_semantic", {})

        def probability(speed):
            return next(
                (value for key, value in probabilities.items() if abs(float(key) - speed) < 1e-6),
                None,
            )

        row = {
            "experiment": self.experiment,
            "scenario": self.scenario,
            "run_id": self.run_id,
            "tick": raw_row.get("tick"),
            "carla_frame": raw_row.get("carla_frame"),
            "simulation_time_s": raw_row.get("simulation_time_s"),
            "ego_speed_mps": raw_row.get("ego_speed_mps"),
            "ego_acceleration_mps2": raw_row.get("ego_acceleration_mps2"),
            "camera_vehicle_pixels": camera.get("vehicle"),
            "camera_pedestrian_pixels": camera.get("pedestrian"),
            "camera_traffic_light_pixels": camera.get("traffic_light"),
            "bev_vehicle_pixels": bev.get("vehicle"),
            "bev_walker_pixels": bev.get("walker"),
            "bev_stop_sign_pixels": bev.get("stop_sign"),
            "bev_light_green_pixels": bev.get("traffic_light_green"),
            "bev_light_yellow_pixels": bev.get("traffic_light_yellow"),
            "bev_light_red_pixels": bev.get("traffic_light_red"),
            "prob_speed_0": probability(0.0),
            "prob_speed_2": probability(2.0),
            "prob_speed_3": probability(3.0),
            "prob_speed_6": probability(6.0),
            "pred_target_speed_mps": raw_row.get("pred_target_speed"),
            "decided_action": raw_row.get("decided_action"),
            "previous_action": raw_row.get("previous_command"),
            "decision_changed": raw_row.get("changed"),
            "throttle": raw_row.get("throttle"),
            "brake": raw_row.get("brake"),
            "steer": raw_row.get("steer"),
            "expected_perception": raw_row.get("expected_perception"),
            "actual_perception": raw_row.get("actual_perception"),
            "perception_correct": raw_row.get("perceieved_right"),
            "expected_safe_action": raw_row.get("expected_safe_action"),
            "action_correct": raw_row.get("action_correct"),
            "timely_action": raw_row.get("timely_action"),
            "collision": raw_row.get("collision"),
            "distance_to_crossing_vehicle_m": raw_row.get("distance_to_object_m"),
            "crossing_vehicle_speed_mps": raw_row.get("object_speed_mps"),
            "time_to_collision_s": raw_row.get("time_to_collision_s"),
        }
        self.rows.append(row)
        _append_csv(self.inference_log_path, INFERENCE_COLUMNS, row)
        _append_csv(self.all_inferences_path, INFERENCE_COLUMNS, row)

    def _first(self, predicate):
        return next((row for row in self.rows if predicate(row)), None)

    def complete(self, collision_occurred, saved_items):
        if self.completed:
            return

        distances = [row["distance_to_crossing_vehicle_m"] for row in self.rows if row["distance_to_crossing_vehicle_m"] is not None]
        ttc_values = [row["time_to_collision_s"] for row in self.rows if row["time_to_collision_s"] is not None]
        speeds = [row["ego_speed_mps"] for row in self.rows if row["ego_speed_mps"] is not None]
        detected = self._first(lambda row: row["actual_perception"] == "vehicle")
        slow = self._first(lambda row: row["decided_action"] == "slow")
        stop = self._first(lambda row: row["decided_action"] == "stop")
        hard_brake = self._first(lambda row: row["decided_action"] == "hard_brake")
        final_action = self.rows[-1]["decided_action"] if self.rows else None

        summary = {
            "experiment": self.experiment,
            "scenario": self.scenario,
            "run_id": self.run_id,
            "interception_severity": self.metadata.get("interception_severity"),
            "trigger_distance_m": self.metadata.get("trigger_distance_m"),
            "trigger_max_seconds": self.metadata.get("trigger_max_seconds"),
            "actual_trigger_tick": self.metadata.get("actual_trigger_tick"),
            "actual_trigger_distance_m": self.metadata.get("actual_trigger_distance_m"),
            "initial_ego_speed_mps": speeds[0] if speeds else None,
            "first_vehicle_detection_tick": detected["tick"] if detected else None,
            "first_vehicle_detection_distance_m": detected["distance_to_crossing_vehicle_m"] if detected else None,
            "first_slowdown_tick": slow["tick"] if slow else None,
            "first_slowdown_distance_m": slow["distance_to_crossing_vehicle_m"] if slow else None,
            "first_stop_tick": stop["tick"] if stop else None,
            "first_stop_distance_m": stop["distance_to_crossing_vehicle_m"] if stop else None,
            "first_hard_brake_tick": hard_brake["tick"] if hard_brake else None,
            "first_hard_brake_distance_m": hard_brake["distance_to_crossing_vehicle_m"] if hard_brake else None,
            "minimum_distance_to_crossing_vehicle_m": min(distances) if distances else None,
            "minimum_time_to_collision_s": min(ttc_values) if ttc_values else None,
            "minimum_ego_speed_mps": min(speeds) if speeds else None,
            "final_action": final_action,
            "decision_change_count": sum(bool(row["decision_changed"]) for row in self.rows),
            "collision_occurred": bool(collision_occurred),
        }
        _append_csv(self.all_runs_path, RUN_COLUMNS, summary)
        self.completed = True

        duration = self.metadata.get("num_ticks", 0) / self.metadata.get("simulation_frequency_hz", 1)
        saved_text = "\n".join(f"  {item}" for item in saved_items)
        separator = "=" * 61
        logging.info(
            "\n%s\n                       RUN COMPLETE\n%s\n"
            "Experiment:           %s\nScenario:             %s\nRun:                  %d\n\n"
            "Duration:             %.2f s\nInference Checks:     %d\nDecision Changes:     %d\n\n"
            "Minimum Distance:     %s\nMinimum TTC:          %s\n\n"
            "First Vehicle Detect: %s\nFirst Slowdown:       %s\nFirst Stop:           %s\n"
            "First Hard Brake:     %s\n\nFinal Action:         %s\nCollision:            %s\n\n"
            "Saved\n%s\n%s",
            separator, separator, self.experiment, self.scenario, self.run_number,
            duration, len(self.rows), summary["decision_change_count"],
            _number(summary["minimum_distance_to_crossing_vehicle_m"], " m"),
            _number(summary["minimum_time_to_collision_s"], " s"),
            _event_text(detected), _event_text(slow), _event_text(stop),
            _event_text(hard_brake), (final_action or "N/A").upper(),
            "Yes" if collision_occurred else "No", saved_text, separator,
        )

    def finalize(self, run_completed, gps_data, collision_data, spawned_actors):
        gps_saved = save_gps_data(gps_data, self.gps_log_path)
        collision_saved = save_collision_data(collision_data, self.collision_log_path)
        cleanup_actors(spawned_actors)
        try:
            if run_completed:
                saved = [
                    "README.md", "inference_log.csv", "all_inferences.csv",
                    "all_runs.csv", "scenario_log.txt",
                ]
                if gps_saved:
                    saved.append("gps_log.json")
                if collision_saved:
                    saved.append("collision_log.json")
                if any(self.images_folder.glob("*.png")):
                    saved.append("RGB images")
                if any(self.lidar_folder.glob("*.ply")):
                    saved.append("LiDAR point clouds")
                self.complete(bool(collision_data), saved)
        finally:
            self.close()

    def close(self):
        if self.file_handler is not None:
            self.file_handler.close()
            logging.getLogger().removeHandler(self.file_handler)
            self.file_handler = None


def create_intersection_interception_output(
    scenario_config,
    world,
    model_name,
    ego_spawn_point,
    simulation_frequency_hz,
    inference_interval_ticks,
    num_ticks,
    base_folder="output",
):
    """Create one run under the shared intersection_interception folder."""

    return IntersectionInterceptionRunOutput(
        scenario=scenario_config.scenario_name,
        description=scenario_config.description,
        metadata={
            "map_name": world.get_map().name.rsplit("/", 1)[-1],
            "model_name": model_name,
            "ego_spawn_point": ego_spawn_point,
            "weather": str(world.get_weather()),
            "interception_severity": scenario_config.severity,
            "trigger_distance_m": scenario_config.trigger_distance_m,
            "trigger_max_seconds": scenario_config.trigger_max_seconds,
            "crossing_vehicle_speed_mps": scenario_config.interception_vehicle_speed_mps,
            "simulation_frequency_hz": simulation_frequency_hz,
            "inference_interval_ticks": inference_interval_ticks,
            "num_ticks": num_ticks,
        },
        base_folder=base_folder,
    )
