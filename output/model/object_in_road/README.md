# Object in Road

## Purpose

This experiment evaluates how the driving model perceives and responds to a stationary object placed directly in the ego vehicle's lane.

## Research question

How does object identity affect the model's perception, target-speed prediction, and stopping behavior when the road geometry and starting conditions are held constant?

## Scenario variants

- **Chair:** an out-of-distribution plastic chair.
- **Pedestrian:** an in-distribution stationary pedestrian.
- **Pedestrian in Red:** the stationary-pedestrian setup using the configured red-shirt pedestrian blueprint.
- **Vehicle:** an in-distribution stationary vehicle.

## Controlled conditions

The scenarios use the same CARLA map, ego spawn point, initial object distance, simulation frequency, inference interval, camera mount, model configuration, and straight-road approach. The obstacle remains stationary in the ego lane.

## Variable being tested

The principal independent variable is object identity or pedestrian appearance: chair, pedestrian, pedestrian in red, or vehicle.

## Data collected

Run folders are grouped by scenario variant as `chair_object/run_N`, `pedestrian_object/run_N`, `pedestrian_in_red/run_N`, and `vehicle_object/run_N`. Each run records one structured row per inference in `inference_log.csv`, human-readable runtime diagnostics in `scenario_log.txt`, GPS readings, frame-aligned RGB images and LiDAR point clouds, and collision events. Aggregate inference and completed-run tables remain stored at the Object in Road family level in `all_inferences.csv` and `all_runs.csv`.

## Comparing runs

Compare variants using matching scenario settings. Use `all_inferences.csv` for tick-level model, perception, control, and safety measurements. Use `all_runs.csv` for run-level approach and stopping outcomes. Consult each run's README for its exact map, blueprint, weather, and settings.
