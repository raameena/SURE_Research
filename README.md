# CARLA Autonomous Driving Safety Research

## Project Overview

This repository contains research software for constructing controlled driving scenarios in the CARLA simulator and observing how an autonomous-driving system responds to traffic signals, road users, and obstacles. The model-driven scenarios integrate a PCLA TransFuser++ checkpoint through repository-specific adapters; the scripted scenarios provide explicit rule-based control paths for comparison and scenario development.

The broader research concerns potentially unsafe model decisions and the relationship between model perception, selected driving actions, available reaction distance, and collision events. This is an ongoing project. The included data records prior runs, but this repository does not present those records as validated research findings.

## Research Motivation

Controlled simulation makes it possible to vary a hazard or traffic condition while keeping much of the surrounding environment fixed. In this codebase, scenarios place known actors, control traffic-light phases, advance CARLA at defined intervals where required, and record both simulator ground truth and model-derived information. These mechanisms support repeatable investigation without assuming that a recorded run generalizes beyond its documented configuration.

## Research Objectives

The implementation supports the following objectives:

- construct repeatable intersection and straight-road CARLA scenarios;
- introduce controlled traffic-light phases, crossing actors, cut-in vehicles, and stationary road obstacles;
- compare scripted vehicle behavior with model-driven behavior;
- record sensor data, model predictions, applied controls, ground-truth distances, and collision events;
- evaluate whether model perception and selected actions agree with scenario-defined expectations; and
- organize run-level and inference-level data for later comparison.

## My Contributions

The research-developed work represented in this deliverable includes:

- modular CARLA connection, spawning, actor-cleanup, sensor-recording, and run-folder utilities;
- reusable control primitives and traffic-light/hazard monitoring logic;
- scripted intersection scenarios for stop/go behavior and interactions with pedestrians, bicyclists, and another vehicle;
- model-evaluation scenarios for traffic lights, intersection conflicts, cut-in timing, and stationary objects in the road;
- adapters that load PCLA configuration, prepare synchronized camera/LiDAR input, send inference requests, translate model outputs into CARLA controls, and provide route inputs;
- scenario-specific perception/action expectations and timing checks; and
- structured output support for per-inference records, completed-run summaries, sensor files, metadata, and data dictionaries.

These contributions are integration and experimental-harness code. CARLA, PCLA, TransFuser++, PyTorch, and the other imported third-party libraries are external projects and are not presented as original contributions here. The copied source does not include the PCLA implementation or its checkpoint.

## Repository Structure

```text
contained_source_code/
|-- config/
|   |-- CARLA_actors/          CARLA connection, spawning, sensors, and actor behavior
|   `-- ml_model/              Model adapters, scenario-specific actor logic, and output writers
|-- controls/                  Scripted, hazard-aware, light-aware, and model-driven controls
|-- scenarios/
|   |-- hard_controls/         Scenarios using predefined control rules
|   `-- model/                 Scenarios driven by model predictions and PCLA-derived controls
`-- output/                    Generated experiment/run data; not executable source code
```

See [Configuration](config/README.md), [Controls](controls/README.md), [Scenarios](scenarios/README.md), and [Outputs](output/README.md) for subsystem details.

## Architecture

```text
CARLA and model configuration
             |
             v
scenario entry point and actor placement
             |
             v
CARLA ego vehicle, environment, and sensors
             |
             v
scripted controls OR PCLA model inference/control adapter
             |
             v
simulation ticks and scenario-specific monitoring
             |
             v
run folders, logs, sensor captures, and aggregate CSV tables
```

Scenario scripts assemble reusable components rather than defining every behavior locally. `config/CARLA_actors/` manages CARLA actors and sensors. `controls/` interprets traffic lights or hazards, applies action primitives, and handles model-derived decisions. Model scenarios additionally use `config/ml_model/` for PCLA configuration, synchronized model sensors, navigation inputs, object-specific perception checks, and structured output generation.

## Scenario Categories

- **Scripted intersection scenarios:** exercise fixed traffic-light sequences and rule-based stop/go or hazard responses. Variants include ordinary and right-lane stop/go runs, pedestrian and bicyclist crossings, and a turning-vehicle conflict.
- **Model intersection scenarios:** let model predictions control the ego vehicle during traffic-light and vehicle-conflict conditions. The copied scenarios include simple stop/go, a broadside conflict, and a cut-in/interception conflict.
- **Intersection-interception experiment:** factors early, medium, and late release configurations through one shared runner to vary nominal reaction margin.
- **Object-in-Road experiment:** places a stationary chair, pedestrian, red-shirt pedestrian, or vehicle ahead of the ego vehicle and records model perception, target-speed, control, distance, and collision information.

The scenario definitions establish experimental conditions; they do not establish experimental outcomes.

## Technologies Visible in the Source

- Python
- CARLA Simulator Python API
- PyTorch
- NumPy
- OpenCV
- `timm`
- `requests`
- Hugging Face Hub (checkpoint download helper)
- PCLA and its TransFuser++ agent interfaces

No complete dependency manifest or authoritative version list is included in this self-contained copy.

## Setup and Requirements

The source establishes the following prerequisites:

1. A CARLA server reachable at `localhost:2000` unless `connect_to_carla()` is called with different values.
2. A Python environment containing the imported CARLA and machine-learning libraries.
3. For model-driven scenarios, a `PCLA/` directory at this repository root with the package layout referenced by `config/ml_model/setup/model_loader.py` and `pcla_navigation.py`.
4. The `tfv4_l6_0` configuration/checkpoint files under PCLA's `pcla_agents/transfuserv4_pretrained/longest6/tfpp_all_0/` path. `config/ml_model/setup/fetch_transfuser_model.py` is the repository's download helper.
5. For the live model-scenario path, a reachable remote `/predict` service whose base URL is stored in `config/ml_model/setup/endpoint.txt`.

PCLA, its inference service, CARLA installation files, and environment specifications are not included here. Consequently, this copy documents the integration but is not, by itself, a complete reproducible installation bundle.

## Running the Code

The scenario files expose `main()` entry points and `if __name__ == "__main__"` guards. Run them from the repository root so imports such as `config`, `controls`, and `scenarios` resolve correctly. Representative entry points are:

```text
scenarios/hard_controls/intersection/simple_stop_go.py
scenarios/hard_controls/intersection/pedestrian_crossing.py
scenarios/model/intersection/simple_stop_go.py
scenarios/model/intersection_interception/early_interception.py
scenarios/model/object_in_road/chair_object.py
```

The source does not provide a single launcher, command-line interface, or verified end-to-end setup command. Review scenario constants and prerequisites before executing a run; several source comments explicitly identify tuning values or assumptions that still require live validation.

## Outputs

Depending on the scenario, runs can produce RGB PNG frames, LiDAR PLY point clouds, GNSS/GPS JSON, collision JSON, human-readable scenario logs, per-inference CSV files, and aggregate CSV summaries. The structured model experiments also generate run metadata and data dictionaries. See [output/README.md](output/README.md) and the experiment-specific documentation linked there.

## Project Status

This is ongoing research software. Some scenario parameters are documented in the source as placeholders or values requiring confirmation in a live CARLA/model session. Existing output demonstrates that runs were recorded; no claim of validated performance, safety, statistical significance, or completed experimental analysis is made here.

## Directory Documentation

- [Configuration and actor utilities](config/README.md)
- [Model integration and experiment support](config/ml_model/README.md)
- [Control modules](controls/README.md)
- [Scenario hierarchy](scenarios/README.md)
- [Model-driven scenarios](scenarios/model/README.md)
- [Generated outputs](output/README.md)
- [Intersection Interception experiment](output/model/intersection_interception/README.md)
- [Object in Road experiment](output/model/object_in_road/README.md)

