# CARLA Scenario Suite

The `scenarios/` directory contains the executable simulation setups used by the project. Each scenario combines shared actor, sensor, control, model, and output components with a specific traffic condition or hazard.

## Scenario Categories

| Category | Description |
| --- | --- |
| `hard_controls/` | Scripted-control scenarios that use predefined vehicle actions and hazard-response rules. |
| `model/` | Model-driven scenarios in which autonomous-driving model predictions control the ego vehicle. |

Scripted-control scenarios support scenario development and controlled baseline behavior without model inference. Model-driven scenarios use the same types of conditions to study autonomous-driving model behavior.

## Scripted-Control Scenarios

All scripted-control scenarios are currently under `hard_controls/intersection/`.

| Scenario | Purpose |
| --- | --- |
| `simple_stop_go.py` | Runs red and green traffic-light phases with predefined stop and forward actions. |
| `right_lane_stop_go.py` | Repeats the stop/go sequence from the driving lane to the right of the configured base spawn. |
| `pedestrian_crossing.py` | Releases a pedestrian across the ego vehicle's path and combines light-based control with pedestrian hazard checks. |
| `bicyclist_crossing.py` | Uses the same hazard-aware structure for a crossing bicyclist. |
| `car_crash_intersection.py` | Scripts a second vehicle along a left-turn path and uses time-to-collision-aware hazard logic. |

The vehicle-conflict scenario includes speed and timing values that still require live CARLA tuning, as noted in the source file.

## Model-Driven Scenario Groups

| Scenario Group | Description |
| --- | --- |
| `model/intersection/` | Traffic-light and vehicle-conflict scenarios using model-derived ego-vehicle controls. |
| `model/intersection_interception/` | Early, medium, and late cut-in conditions that vary the nominal reaction margin. |
| `model/object_in_road/` | Stationary chair, pedestrian, red-shirt pedestrian, and vehicle conditions on a straight-road approach. |

See [`model/README.md`](model/README.md) for file-level descriptions.

## Common Scenario Workflow

```text
Connect to CARLA
       ↓
Configure the world and spawn actors
       ↓
Attach recording and model sensors
       ↓
Run traffic-light or hazard conditions
       ↓
Apply scripted or model-driven controls
       ↓
Record outputs and clean up actors
```

Each scenario defines its own spawn point, actor placement, duration, and monitoring interval. Model-driven scenarios also enable synchronous stepping so camera and LiDAR frames align with each inference.

## Related Components

- [`../config/README.md`](../config/README.md) describes CARLA actors, sensors, and model setup.
- [`../controls/README.md`](../controls/README.md) describes the control paths used during a run.
- [`../output/README.md`](../output/README.md) describes the generated data.

