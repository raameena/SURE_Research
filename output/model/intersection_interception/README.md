# Intersection Interception Experiment

## Research question

How does model behavior change as an unexpected cut-in vehicle turns into the ego lane with decreasing reaction margin?

## Scenarios

- Early Interception
- Medium Interception
- Late Interception

## Controlled variables

The ego spawn, map, model, sensors, weather, simulation rate, model inference interval, interception vehicle type, interception vehicle speed, left-turn merge path, fixed green ego traffic light, and fixed red non-ego intersection signals are held constant.

## Primary changed variable

The cut-in vehicle's release trigger changes the interception timing. Early provides the largest nominal reaction margin, Medium a moderate margin, and Late the smallest. The vehicle uses `interception_car.py` logic: it turns into the ego lane and stops after reaching the end of the merge path. Actual gap, speed, and TTC must be used when comparing runs because the model-controlled ego can vary.

## Data collected

The experiment records model target speed and probabilities, semantic outputs, actual ego speed and controls, longitudinal bumper gap to the interception vehicle, interception-vehicle speed, TTC when physically meaningful, and collision outcome. RGB and LiDAR filenames use the CARLA frame number recorded in each inference row.

Collision is an observed outcome, not a scripted goal. The ego remains controlled by the ML model; no emergency response is injected by the scenario.
