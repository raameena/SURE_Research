# Object in Road Experiment

This directory contains generated data for model-driven scenarios with a stationary object placed in the ego vehicle's lane. The experiment compares model perception and driving behavior across different object types and pedestrian appearances.

## Research Question

How does object identity affect model perception, target-speed prediction, and stopping behavior when the road geometry and starting conditions are held constant?

## Scenario Variants

| Directory | Scenario |
| --- | --- |
| `chair_object/` | A stationary plastic chair placed ahead of the ego vehicle. |
| `pedestrian_object/` | A stationary pedestrian placed ahead of the ego vehicle. |
| `pedestrian_in_red/` | The pedestrian scenario using the configured red-shirt pedestrian blueprint. |
| `vehicle_object/` | A stationary vehicle placed ahead of the ego vehicle. |

The primary changed condition is object identity or pedestrian appearance.

## Controlled Conditions

Comparable runs use the same CARLA map, ego spawn point, starting object distance, simulation frequency, inference interval, camera configuration, model configuration, and straight-road approach. The object remains stationary in the ego lane.

Check each run's `README.md` before combining data because settings can differ during development.

## Directory Structure

```text
object_in_road/
├── all_inferences.csv
├── all_runs.csv
├── DATA_DICTIONARY.md
├── chair_object/run_N/
├── pedestrian_object/run_N/
├── pedestrian_in_red/run_N/
└── vehicle_object/run_N/
```

Each `run_N/` directory can contain an inference CSV, scenario log, GPS and collision logs, RGB images, LiDAR point clouds, and a run-specific README.

## Recorded Data

| Category | Examples |
| --- | --- |
| Ground truth | Object type, object distance, expected perception, and expected safe action |
| Model output | Target-speed probabilities, semantic counts, selected target speed, and decoded action |
| Vehicle behavior | Speed, acceleration, throttle, brake, and steering |
| Safety measures | Perception/action checks, stopping timeliness, time to collision, and collision events |
| Navigation and sensors | Route target, road command, RGB images, LiDAR point clouds, and GNSS data |

Semantic image and BEV values are pixel or cell counts, not object counts. See [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) for complete field definitions.

## Using the Data

- Use a run's `inference_log.csv` to follow model behavior over time.
- Use `all_inferences.csv` to compare compatible rows across variants and runs.
- Use `all_runs.csv` for run-level approach, stopping, and collision summaries.
- Use run README files to match map, object, weather, and inference settings.

