# Model-Driven Scenarios

Model-driven scenarios use autonomous-driving model predictions to control the ego vehicle. Scenario code defines the environment and expected conditions, while the PCLA TransFuser++ integration supplies throttle, brake, and steering.

## How Model Control Works

1. Spawn the ego vehicle and any traffic lights, obstacles, or hazard actors.
2. Attach recording sensors and separate model-input camera and LiDAR sensors.
3. Request frame-matched model inference.
4. Convert the predicted checkpoint and target speed into CARLA control.
5. Record the decision, applied control, ground truth, and safety measurements.

Expected perception and action labels support logging and later comparison. They do not replace the model's control command.

## Experiment Families

| Directory | Purpose |
| --- | --- |
| `intersection/` | Tests model behavior around controlled traffic lights and intersection vehicle conflicts. |
| `intersection_interception/` | Runs early, medium, and late cut-in conditions through a shared experiment runner. |
| `object_in_road/` | Tests model behavior when a stationary object or road user is placed ahead of the ego vehicle. |

## Intersection Scenarios

| File | Purpose |
| --- | --- |
| `intersection/simple_stop_go.py` | Runs controlled red and green signal phases and records model decisions and controls. |
| `intersection/T_bone_crash.py` | Adds a turning vehicle on a broadside conflict path and switches from light-based to vehicle-based evaluation when relevant. |
| `intersection/interception_crash.py` | Creates a cut-in vehicle that turns into and settles in the ego lane. It uses longitudinal bumper gap and distance-based expected actions. |

Some conflict geometry and friction values are marked for live validation in the source comments.

## Intersection Interception

`early_interception.py`, `medium_interception.py`, and `late_interception.py` pass different release configurations to `run_interception_scenario.py`. The shared runner handles CARLA setup, fixed signal states, cut-in control, model inference, and structured output.

See the [Intersection Interception output guide](../../output/model/intersection_interception/README.md).

## Object in Road

| File | Scenario Object | Perception Check |
| --- | --- | --- |
| `object_in_road/chair_object.py` | Stationary chair | Checks for non-drivable BEV occupancy near the projected chair location because the configured classes do not include a chair class. |
| `object_in_road/pedestrian_object.py` | Stationary pedestrian | Uses the model's walker-class BEV output. |
| `object_in_road/pedestrian_in_red.py` | Stationary red-shirt pedestrian | Reuses the pedestrian scenario with a specific blueprint and output label. |
| `object_in_road/vehicle_object.py` | Stationary vehicle | Uses the model's vehicle-class BEV output. |

These scenarios use world-route navigation and structured output writers. They unload CARLA's parked-vehicle map layer before a run and restore it during cleanup.

See the [Object in Road output guide](../../output/model/object_in_road/README.md).

## Related Components

- Model setup and evaluation: [`../../config/ml_model/README.md`](../../config/ml_model/README.md)
- Model control: [`../../controls/README.md`](../../controls/README.md)
- Generated data: [`../../output/README.md`](../../output/README.md)

