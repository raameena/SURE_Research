# Scenario Suite

This directory contains executable CARLA scenario entry points. Each script combines reusable configuration, actor, sensor, control, and logging modules with scenario-specific constants. The hierarchy separates scripted controls from model-driven evaluation.

## Scripted Scenarios: `hard_controls/`

These scenarios use CARLA ground truth and predefined rules from `controls/hard_controls.py` or `controls/hazard_controls.py`. They provide controlled baselines and reusable hazard setups; they are not evaluations of learned decision-making.

All copied scripted scenarios are under `hard_controls/intersection/`:

| Scenario | Controlled interaction |
| --- | --- |
| `simple_stop_go.py` | Runs red and green traffic-light phases and applies the corresponding stop/forward actions. |
| `right_lane_stop_go.py` | Exercises the light-based stop/go behavior from the driving lane to the right of the configured base spawn. |
| `pedestrian_crossing.py` | Releases a pedestrian across the ego path while monitoring both the light and the pedestrian hazard. |
| `bicyclist_crossing.py` | Releases a bicycle actor across the ego path and uses the same light-plus-hazard monitoring structure. |
| `car_crash_intersection.py` | Scripts a second vehicle along a generated left-turn path and evaluates it through time-to-collision-aware hazard logic. Source comments identify speed/timing values as requiring live tuning. |

These scripts create run folders through `carla_config.create_run_folder()`, attach recording sensors, execute the controlled sequence, save available GPS/collision data, restore traffic-light state where applicable, and destroy actors.

## Model-Driven Scenarios: `model/`

These scenarios attach model-input camera/LiDAR sensors, request PCLA TransFuser++ predictions, and apply PCLA-controller throttle, brake, and steering to the ego vehicle. Scenario code supplies ground truth for later comparison but does not script the ego vehicle to produce the expected response.

- `intersection/` contains model-controlled traffic-light and vehicle-conflict scenarios.
- `intersection_interception/` contains early, medium, and late cut-in entry points backed by one shared runner.
- `object_in_road/` contains straight-road stationary-obstacle scenarios for a chair, pedestrian, red-shirt pedestrian, and vehicle.

See [Model-Driven Scenarios](model/README.md) for file-level detail.

## Common Execution Flow

1. Connect to CARLA and select the configured map spawn point or lane.
2. Configure synchronous stepping where required by model sensor alignment.
3. Create a run directory and spawn the ego/scenario actors.
4. Attach independent recording sensors and, for model scenarios, model-input sensors.
5. Execute fixed signal phases or an object/hazard monitoring loop.
6. Record outputs and restore traffic-light, world, or map-layer state.
7. Stop sensors and destroy spawned actors.

The precise constants, run duration, actor geometry, and cleanup behavior remain defined in each scenario file. Some model and conflict scenarios explicitly document parameters that have not yet been validated in a live end-to-end session.

