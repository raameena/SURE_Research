# Configuration and CARLA Components

The `config/` directory contains reusable CARLA setup, actor, sensor, and model-integration components. Scenario scripts use these modules to assemble each simulation without duplicating common behavior.

## Directory Structure

```text
config/
├── CARLA_actors/
└── ml_model/
    ├── actors/
    ├── output_config/
    └── setup/
```

## CARLA Components

| File | Purpose |
| --- | --- |
| `CARLA_actors/carla_config.py` | Connects to CARLA, manages synchronous settings, creates run folders, spawns the ego vehicle, writes JSON logs, and cleans up actors. |
| `CARLA_actors/sensors.py` | Attaches RGB, LiDAR, GNSS, and collision sensors used for research data collection. It also provides the spectator-follow view. |
| `CARLA_actors/spawn_locations.py` | Provides an interactive utility for browsing CARLA spawn points. |
| `CARLA_actors/pedestrian.py` | Spawns, holds, starts, and stops crossing pedestrians. |
| `CARLA_actors/bicyclist.py` | Spawns, holds, starts, and stops crossing bicyclists. |
| `CARLA_actors/crash_car.py` | Finds junction geometry and controls a second vehicle along a generated left-turn path. |

## Model-Related Components

| Directory | Purpose |
| --- | --- |
| `ml_model/setup/` | Loads PCLA configuration, prepares model inputs, provides route navigation, adapts controller output, and evaluates expected behavior. |
| `ml_model/actors/` | Defines experiment-specific actor placement, hazard geometry, perception checks, and expected actions. |
| `ml_model/output_config/` | Creates structured run folders, CSV files, metadata, and data dictionaries. |

See [`ml_model/README.md`](ml_model/README.md) for a detailed component map.

## How Scenarios Use This Directory

1. Connect to CARLA through `carla_config.py`.
2. Select a spawn location and create the ego vehicle.
3. Add pedestrians, bicyclists, vehicles, or stationary objects.
4. Attach the recording and collision sensors needed for the run.
5. Load components from `ml_model/` for model-driven scenarios.
6. Stop sensors, destroy actors, and restore simulation settings.

Control decisions are handled in [`../controls/`](../controls/README.md), while executable scenarios are organized in [`../scenarios/`](../scenarios/README.md).

