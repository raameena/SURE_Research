# Model-Driven Scenarios

The scenarios in this subtree leave longitudinal and lateral ego control to the integrated PCLA TransFuser++ prediction/control path. They observe model outputs against scenario-defined conditions; expected labels are used for evaluation and logging, not to replace the model's control with a scripted answer.

## `intersection/`

- `simple_stop_go.py` runs controlled red/green traffic-light phases, interprets model semantic output, and records model decisions and applied control.
- `T_bone_crash.py` combines signal phases with a scripted turning vehicle on a broadside conflict path. During the vehicle-relevant phase, object-specific perception/action expectations replace light-only expectations.
- `interception_crash.py` creates a cut-in vehicle that turns into and settles in the ego lane. It uses longitudinal bumper gap and tiered expected actions; its source explicitly marks introduced geometry and friction values as requiring live validation.

These older intersection scripts use the general run-folder utility and scenario logs rather than the newer family-level structured-output classes.

## `intersection_interception/`

`early_interception.py`, `medium_interception.py`, and `late_interception.py` are thin entry points. Each passes a different `InterceptionScenarioConfig` to `run_interception_scenario.py`. The shared runner controls setup, fixed signal states, cut-in execution, model monitoring, and structured output. The variants change the ego travel-distance trigger before release, providing different nominal reaction margins; recorded gap, speed, and time-to-collision values are needed for actual run comparisons.

Experiment documentation: [Intersection Interception](../../output/model/intersection_interception/README.md).

## `object_in_road/`

- `chair_object.py` places a stationary chair ahead of the ego vehicle. Because the documented model class map lacks a chair class, its resolver checks for non-drivable BEV occupancy at the projected object location.
- `pedestrian_object.py` places a stationary pedestrian and evaluates the model's walker-class BEV output.
- `pedestrian_in_red.py` reuses the pedestrian runner with a specified red-shirt pedestrian blueprint and corresponding output label.
- `vehicle_object.py` places a stationary vehicle and evaluates the vehicle-class BEV output.

These scripts use `PCLAWorldRouteNavigation`, structured Object-in-Road output writers, collision sensing, and independent recording/model sensor sets. They temporarily unload CARLA's parked-vehicle map layer and restore it during cleanup.

Experiment documentation: [Object in Road](../../output/model/object_in_road/README.md).

## External Model Boundary

The PCLA package, TransFuser++ implementation, checkpoint, and remote inference service are external to this copied tree. The repository code supplies CARLA scenario orchestration, sensor synchronization, preprocessing/inference transport, controller adaptation, ground-truth comparison, and output recording around those dependencies.

