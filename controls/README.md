# Vehicle Controls

The modules in this directory turn scenario state or model output into CARLA vehicle commands. They are shared by both scripted scenarios and model-evaluation scenarios.

## Modules

### `actions.py`

Defines the `VehicleAction` vocabulary, default `VehicleControl` mappings, and `apply_action()`. Convenience functions expose forward driving, slow driving, coasting, stopping, hard/emergency braking, turning, reversing, and speed measurement. These are direct command primitives; they do not detect hazards or interpret model output.

### `hard_controls.py`

Implements predefined traffic-light behavior. Red stops the vehicle, green drives forward, yellow slows, and an unknown state uses a conservative stop. `monitor_light_and_act()` periodically observes the active signal, records light/action information, applies the corresponding scripted action, and advances the simulation.

### `hazard_controls.py`

Extends scripted light handling with nearby-actor checks. It finds selected CARLA actor types, computes actor position relative to the ego vehicle, estimates time to collision from relative motion, evaluates safety zones, and monitors traffic lights and hazards together. Pedestrian and bicyclist scenarios use spatial safety-zone logic; the vehicle-conflict path can use time-to-collision gating.

### `light_controls.py`

Normalizes CARLA traffic-light states, logs signal phases, estimates distance to a stop line, finds the light affecting the ego vehicle, changes or freezes signal state, runs fixed light sequences, monitors state, and restores an unfrozen light. Scenario files use these helpers to construct controlled signal phases.

### `ml_controls.py`

Connects model predictions to vehicle control and research logging. It:

- maps target-speed bins to high-level actions;
- interprets camera and bird's-eye-view semantic class maps;
- projects world locations into the model's BEV grid;
- selects actions, including an urgency-dependent hard-brake path;
- requests synchronized model/control predictions;
- applies the PCLA-derived `VehicleControl` on each tick; and
- records decisions, class scores, expected/actual perception, action correctness, stopping timeliness, distance, time to collision, and collision state through callbacks.

`monitor_with_model_and_act()` handles traffic-light-oriented phases. `monitor_with_model_and_act_for_object()` generalizes monitoring to a scenario actor and accepts scenario-specific expectation, perception, and distance functions.

## Scripted Versus Model Control

Scripted scenarios call `hard_controls.py` or `hazard_controls.py`, which select known action primitives from CARLA ground truth. Model scenarios obtain a prediction through `config/ml_model/setup/model_loader.py`; the TransFuser++ controller adapter supplies throttle, brake, and steering, while the high-level decoded action is used for interpretation and evaluation. The model scenarios do not inject a separate emergency response solely to force a desired experimental outcome.

