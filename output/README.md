# Generated Experiment Outputs

This directory primarily contains generated run data, not executable source code. The code that creates and populates the structured model-output hierarchy is in `config/ml_model/output_config/`; scenario entry points are in `scenarios/model/`.

## Organization

```text
output/model/
|-- intersection_interception/
|   |-- all_inferences.csv
|   |-- all_runs.csv
|   |-- DATA_DICTIONARY.md
|   |-- README.md
|   `-- run_N/
`-- object_in_road/
    |-- all_inferences.csv
    |-- all_runs.csv
    |-- DATA_DICTIONARY.md
    |-- README.md
    `-- <scenario_variant>/run_N/
```

The Intersection Interception family groups runs directly under the experiment directory. Object-in-Road runs are first grouped by variant, such as `chair_object`, `pedestrian_object`, `pedestrian_in_red`, or `vehicle_object`.

## Run Contents

A structured run can contain:

- `inference_log.csv`: one row per inference attempt for that run;
- `scenario_log.txt`: human-readable runtime diagnostics;
- `gps_log.json`: collected GNSS readings;
- `collision_log.json`: collision-sensor events;
- `images/frame_<CARLA frame>.png`: RGB recording frames;
- `lidar_pointclouds/frame_<CARLA frame>.ply`: LiDAR recordings; and
- `README.md`: run-specific configuration and notes.

The family-level `all_inferences.csv` appends compatible inference rows across runs. `all_runs.csv` contains summaries for runs finalized as completed. Consult each family's `DATA_DICTIONARY.md` for authoritative columns, datatypes, units, and blank-value semantics.

## Interpretation

The data includes simulator state, model target-speed probabilities and decoded actions, applied controls, semantic class pixel/cell counts, scenario ground truth, distance or time-to-collision measures, and collision observations where supported by the experiment. Pixel/cell counts are not object counts. Scenario/run README files document configuration details needed when deciding whether runs are comparable.

These files are records of executions, not validated findings. No performance, safety, or statistical conclusion should be inferred from their presence alone.

## Experiment Documentation

- [Intersection Interception](model/intersection_interception/README.md)
- [Intersection Interception data dictionary](model/intersection_interception/DATA_DICTIONARY.md)
- [Object in Road](model/object_in_road/README.md)
- [Object in Road data dictionary](model/object_in_road/DATA_DICTIONARY.md)

