# Configuration and CARLA Components

This directory contains reusable CARLA actor/sensor utilities and the integration layer used by model-driven experiments. Despite the directory name, these modules include executable setup and behavior code as well as constants.

## `CARLA_actors/`

| Module | Responsibility |
| --- | --- |
| `carla_config.py` | Connects to CARLA, manages synchronous settings and ticks, creates run folders, spawns ego vehicles at selected map locations or lanes, writes JSON logs, and destroys spawned actors. |
| `sensors.py` | Attaches recording RGB, LiDAR, GNSS, and collision sensors and provides the spectator-follow view. These recording sensors are separate from model-input sensors. |
| `spawn_locations.py` | Interactive utility for browsing CARLA spawn points using a spawned vehicle and spectator camera. |
| `pedestrian.py` | Spawns, holds, starts, and stops crossing pedestrians. |
| `bicyclist.py` | Spawns, holds, starts, and stops crossing bicycle actors. |
| `crash_car.py` | Finds junction geometry, spawns a second vehicle, constructs a left-turn path, and updates or stops its scripted maneuver. |

Scenario modules import these functions to keep connection, actor, sensor, and cleanup behavior consistent. The scenario remains responsible for choosing spawn indices, distances, speeds, durations, and which actors to combine.

## `ml_model/`

The model subtree contains research integration code, not the external model implementation itself:

- `setup/` loads PCLA configuration/checkpoints, prepares model sensors and inputs, provides world-route navigation, adapts TransFuser++ controller output, performs expectation checks, and includes checkpoint-fetch/smoke-test helpers.
- `actors/` defines model-experiment actors, ground-truth expectations, perception resolvers, and early/medium/late interception parameters.
- `output_config/` defines structured CSV schemas, run metadata, experiment documentation, sensor/log paths, and finalization behavior.

See [Model Integration and Experiment Support](ml_model/README.md) for details.

## Relationship to Other Directories

`scenarios/` selects and composes these components. `controls/` applies the resulting rule-based or model-derived vehicle behavior. Output helpers create data under `output/`; recording callbacks then populate the run directories while a scenario executes.

