# Experiment Outputs

The `output/` directory contains generated experiment data rather than executable program logic. Each run connects model behavior, vehicle state, simulator ground truth, and sensor data to one scenario execution.

## Directory Structure

```text
output/model/
├── intersection_interception/
│   └── run_N/
└── object_in_road/
    └── <scenario_variant>/run_N/
```

The Intersection Interception experiment stores runs directly under its experiment directory. Object-in-Road runs are grouped by variant, such as `chair_object/`, `pedestrian_object/`, `pedestrian_in_red/`, or `vehicle_object/`.

## Experiment Folders

| Directory | Contents |
| --- | --- |
| `model/intersection_interception/` | Early, medium, and late cut-in experiment data. |
| `model/object_in_road/` | Stationary chair, pedestrian, red-shirt pedestrian, and vehicle experiment data. |

## Run Files

| File / Directory | Purpose |
| --- | --- |
| `inference_log.csv` | Stores one row for each inference attempt in a run. |
| `scenario_log.txt` | Stores human-readable runtime events and diagnostics. |
| `gps_log.json` | Stores GNSS readings. |
| `collision_log.json` | Stores collision-sensor events. |
| `images/` | Stores RGB frames as `frame_<CARLA frame>.png`. |
| `lidar_pointclouds/` | Stores LiDAR frames as `frame_<CARLA frame>.ply`. |
| `README.md` | Records the run configuration and notes. |

## Experiment-Level Files

| File | Purpose |
| --- | --- |
| `all_inferences.csv` | Combines compatible inference rows across runs in one experiment. |
| `all_runs.csv` | Stores one summary row for each completed run. |
| `DATA_DICTIONARY.md` | Defines CSV columns, units, datatypes, and blank-value behavior. |
| `README.md` | Explains the experiment and run organization. |

## Using the Data

Recorded fields can include ego-vehicle motion, model probabilities and actions, applied control, semantic counts, object distance, time to collision, and collision events. Pixel and BEV-cell values are class-assigned counts, not object counts.

Use each run README to confirm its configuration and the corresponding data dictionary to interpret CSV fields.

## Experiment Documentation

- [Intersection Interception](model/intersection_interception/README.md)
- [Intersection Interception data dictionary](model/intersection_interception/DATA_DICTIONARY.md)
- [Object in Road](model/object_in_road/README.md)
- [Object in Road data dictionary](model/object_in_road/DATA_DICTIONARY.md)

