# Intersection Interception Experiment

This directory contains generated data for the model-driven Intersection Interception experiment. The experiment varies when a cut-in vehicle is released into the ego vehicle's lane.

## Research Question

How does model behavior change as an unexpected cut-in vehicle enters the ego lane with less reaction margin?

## Scenario Variants

| Variant | Release Timing | Nominal Reaction Margin |
| --- | --- | --- |
| Early Interception | Earliest release | Largest |
| Medium Interception | Intermediate release | Moderate |
| Late Interception | Latest release | Smallest |

Exact trigger distances are defined in `config/ml_model/actors/intersection_interception/`. Use recorded bumper gap, speed, and time to collision when comparing runs because the model-driven ego vehicle can vary between executions.

## Controlled Conditions

The scenarios share the ego spawn, CARLA map, autonomous-driving model, sensors, weather, simulation frequency, inference interval, cut-in vehicle, merge path, and fixed intersection signal states.

The cut-in vehicle follows `interception_car.py`, turns into the ego lane, and stops after reaching the end of the merge path.

## Directory Structure

```text
intersection_interception/
├── all_inferences.csv
├── all_runs.csv
├── DATA_DICTIONARY.md
└── run_N/
    ├── README.md
    ├── inference_log.csv
    ├── scenario_log.txt
    ├── gps_log.json
    ├── collision_log.json
    ├── images/
    └── lidar_pointclouds/
```

## Recorded Data

| Category | Examples |
| --- | --- |
| Model output | Target-speed probabilities, selected target speed, semantic output, and decoded action |
| Vehicle behavior | Ego speed, acceleration, throttle, brake, and steering |
| Interaction geometry | Longitudinal bumper gap, cut-in vehicle speed, and time to collision |
| Events | Decision changes, perception/action checks, and collision events |
| Sensors | Frame-aligned RGB images, LiDAR point clouds, and GNSS data |

RGB and LiDAR filenames use the CARLA frame number recorded in each inference row. See [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) for field definitions and units.

## Using the Data

- Use `inference_log.csv` to reconstruct one run over time.
- Use `all_inferences.csv` to compare inference rows across runs.
- Use `all_runs.csv` for run-level detection, braking, distance, and collision summaries.
- Use each run's `README.md` to confirm its configuration.

The ego vehicle remains under model control. Collision is recorded as an observed event rather than forced by the scenario.

