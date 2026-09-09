# Vehicle Controls

The `controls/` directory contains the vehicle actions and decision logic shared across the scenario suite. It supports scripted-control scenarios, hazard-aware behavior, traffic-light sequences, and model-driven control.

## Main Components

| File | Purpose |
| --- | --- |
| `actions.py` | Defines reusable actions such as drive, slow, coast, stop, hard brake, turn, and reverse. It also provides vehicle-speed helpers. |
| `hard_controls.py` | Maps observed traffic-light colors to predefined actions and monitors light-based scripted-control scenarios. |
| `hazard_controls.py` | Detects nearby actors, evaluates safety zones, estimates time to collision, and combines hazard response with traffic-light behavior. |
| `light_controls.py` | Finds the ego vehicle's traffic light, logs its state, controls fixed signal sequences, measures stop-line distance, and restores signal behavior. |
| `ml_controls.py` | Interprets autonomous-driving model output, applies model-derived controls, evaluates expected behavior, and records inference-level information. |

## Control Types

### Reusable Action Primitives

`actions.py` converts named actions into CARLA `VehicleControl` values. Higher-level control modules use these functions to apply consistent commands.

### Scripted Control

`hard_controls.py` selects predefined actions from the current traffic-light state:

- red: stop;
- yellow: slow;
- green: drive forward; and
- unknown: stop conservatively.

### Hazard-Aware Control

`hazard_controls.py` adds pedestrian, bicyclist, or vehicle checks to the scripted-control path. Depending on the scenario, it uses relative position, a safety zone, or time to collision to decide when the ego vehicle should slow or stop.

### Traffic-Light Control

`light_controls.py` manages the environmental side of intersection scenarios. It can force signal states, freeze them for a controlled phase, run a fixed sequence, and restore normal behavior afterward.

### Model-Driven Control

`ml_controls.py` requests synchronized predictions through `config/ml_model/setup/model_loader.py`. It then:

1. decodes the model's target-speed prediction;
2. runs the PCLA controller adapter;
3. applies throttle, brake, and steering;
4. interprets camera and bird's-eye-view semantic output; and
5. records decisions, controls, perception checks, distances, collision state, and timing information.

`monitor_with_model_and_act()` handles traffic-light phases. `monitor_with_model_and_act_for_object()` applies the same model-control structure to a specific scenario actor.

## Relationship to Scenarios

Scripted-control scenarios use `hard_controls.py` or `hazard_controls.py`. Model-driven scenarios use `ml_controls.py` while retaining `light_controls.py` for controlled signal conditions.

See [`../scenarios/README.md`](../scenarios/README.md) for the scenario hierarchy.

