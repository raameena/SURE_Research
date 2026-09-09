# Can We Trust AI Behind the Wheel?

## Autonomous Driving Safety Research Using CARLA

**Raameen Ahmed · Dr. Jian Hu · Ming Gao**  
**Sponsored by Ford Motor Company**

This repository contains the source code developed for a research project investigating how autonomous-driving models behave in controlled, safety-critical driving scenarios.

Using the **CARLA simulator**, the project creates repeatable driving situations, runs an autonomous-driving model through them, and records simulation, perception, vehicle-behavior, and model-decision data for later analysis.

---

## Background & Motivation

Machine-learning errors in autonomous vehicles can have serious safety consequences. Even small perception or decision errors may contribute to:

- collisions;
- unsafe lane changes;
- inappropriate speed changes; or
- delayed responses to hazards.

Simulation provides a controlled environment where these situations can be recreated and modified systematically without relying entirely on real-world testing.

Using CARLA, this project can:

- test autonomous-driving models in controlled simulated environments;
- create repeatable safety-critical situations;
- observe how models respond to specific conditions and stimuli;
- collect simulation, perception, and vehicle-behavior data; and
- analyze model behavior across different scenarios.

These analyses can help identify situations in which a model behaves unexpectedly or unsafely and support future work on improving autonomous-driving model behavior and error detection.

---

## Research Question

> **How do autonomous-driving models behave when they encounter unfamiliar or safety-critical situations in simulation, and how can the resulting data help identify unsafe model behavior?**

---

## Methodology & Experimental Design

The research follows four primary stages:

**Build Scenarios → Run Model → Record Data → Analyze Behavior**

### 1. Build Controlled Scenarios

Controlled driving scenarios are developed in CARLA around specific traffic conditions and potential hazards.

Current scenario types include:

- traffic-light controlled intersections;
- pedestrian and bicyclist crossings;
- vehicle conflicts at intersections;
- vehicle interception and cut-in situations; and
- stationary objects in the roadway.

### 2. Run the Autonomous-Driving Model

The autonomous-driving model is integrated into selected scenarios and used to control the ego vehicle.

Scripted-control scenarios are also maintained to develop and test scenario behavior independently of the autonomous-driving model.

### 3. Record Simulation Data

During experimental runs, the framework can record information such as:

- ego-vehicle speed and acceleration;
- model decisions and vehicle controls;
- object and hazard distances;
- model perception information;
- RGB camera data;
- LiDAR data;
- GPS/GNSS data; and
- collision events.

### 4. Analyze Model Behavior

The collected data can be used to investigate:

- when the model detects or responds to a hazard;
- what driving action the model selects;
- how the vehicle behaves following that decision;
- how much reaction distance is available; and
- whether unexpected or unsafe behavior occurs.

---

## My Contributions

My work on this research focuses on developing the **CARLA simulation and experimental framework** used to test and analyze autonomous-driving behavior.

Key contributions include:

- developing controlled CARLA driving scenarios for traffic signals, road users, vehicles, and roadway obstacles;
- creating reusable vehicle-control, traffic-light, hazard, actor, and sensor components;
- developing both scripted-control and model-driven scenarios;
- integrating an autonomous-driving model into the CARLA scenario framework;
- building infrastructure to record sensor, perception, vehicle-behavior, model-decision, and collision data;
- organizing experimental outputs for comparison across scenarios; and
- troubleshooting simulation, model-integration, trajectory, and data-collection behavior.

The autonomous-driving model and external frameworks used by this project are third-party technologies. My contributions focus on their **integration, scenario development, experimental design, data collection, and analysis infrastructure**.

---

## Repository Structure

```text
contained_source_code/
│
├── config/
│   ├── CARLA_actors/
│   └── ml_model/
│
├── controls/
│
├── scenarios/
│   ├── hard_controls/
│   └── model/
│
└── output/
    └── model/
```

### Directory Overview

| Directory | Purpose |
| --- | --- |
| `config/` | Reusable configuration and setup components for CARLA actors, sensors, and model integration. |
| `controls/` | Vehicle actions and control logic used by scripted and model-driven scenarios. |
| `scenarios/` | Controlled CARLA experiments used to test different driving conditions and hazards. |
| `output/` | Recorded simulation and model data generated during experimental runs. |

More detailed documentation is available inside the major directories.

---

## System Architecture

The project uses a modular structure so scenarios can share common actor, sensor, control, and model-integration components.

```text
Scenario Configuration
        │
        ▼
CARLA Environment & Actors
        │
        ▼
Sensors & Model Inputs
        │
        ▼
Autonomous Model / Scripted Controls
        │
        ▼
Vehicle Behavior
        │
        ▼
Data Collection
        │
        ▼
Behavior Analysis
```

Rather than defining every component separately inside each scenario, reusable modules provide common functionality for CARLA actors, sensors, controls, model integration, and data collection.

---

## Scenario Organization

The scenarios are divided primarily into **scripted-control scenarios** and **model-driven scenarios**.

### Scripted-Control Scenarios

**Location:** `scenarios/hard_controls/`

These scenarios use predefined vehicle actions and hazard-response logic. They allow scenario behavior and environmental interactions to be developed and tested independently of the autonomous-driving model.

Examples include:

- traffic-light stop/go behavior;
- pedestrian crossings;
- bicyclist crossings; and
- vehicle interactions at intersections.

### Model-Driven Scenarios

**Location:** `scenarios/model/`

These scenarios integrate the autonomous-driving model so that model predictions influence the behavior of the ego vehicle.

Current model-driven scenario families include:

| Scenario Family | Purpose |
| --- | --- |
| `intersection/` | Tests model behavior around traffic lights and intersection conditions. |
| `intersection_interception/` | Creates controlled vehicle conflicts with different reaction margins. |
| `object_in_road/` | Places stationary objects or road users in the ego vehicle's path. |

---

## Data Collection

Model-driven experiments record information across multiple parts of the simulation.

The collected data can connect:

| Category | Example Information |
| --- | --- |
| Simulation | Time, scenario state, run information |
| Ground Truth | Object type, object distance, collision state |
| Perception | Detected objects and model perception information |
| Model Decision | Selected action and target behavior |
| Vehicle Behavior | Speed, acceleration, throttle, brake, steering |
| Sensors | RGB camera, LiDAR, GPS/GNSS |
| Outcome | Collision events and final run state |

This structure allows a model's behavior to be reconstructed over time rather than evaluating an experiment only by its final outcome.

---

## Tools & Technologies

The project uses:

- **Python**
- **CARLA 0.9.16**
- **Bench2Drive / Bench2DriveZoo**
- **PCLA / TransFuser++**
- **PyTorch**
- **NumPy**
- **Git / GitHub**

Third-party autonomous-driving models, frameworks, and libraries are integrated into the research environment but are not original components of this project.

---

## Current Progress

This research is ongoing.

### Completed

- Developed the modular CARLA scenario framework.
- Built controlled traffic-light and hazard scenarios.
- Developed scripted interactions involving pedestrians, bicyclists, vehicles, and other roadway conditions.
- Built the intersection-interception experiment.
- Integrated the autonomous-driving model into the intersection-interception scenario.
- Implemented structured simulation and model data collection.
- Collected simulation data including vehicle behavior, perception, model decisions, and collision information.

### In Progress

- Integrating and testing the autonomous-driving model with the chair object-in-road scenario.
- Expanding model-driven testing to additional controlled scenarios.
- Organizing collected data for comparison across experimental conditions.

---

## Future Work

Future development and research will focus on:

- completing model integration for the chair scenario;
- running additional controlled scenarios;
- comparing model behavior across different safety-critical situations;
- analyzing collected data for patterns in unsafe or unexpected behavior; and
- using these findings to support future autonomous-driving error-detection work.

---

## Documentation

Additional technical documentation is available for the major components of the project:

- [`config/`](config/README.md) — CARLA and model configuration.
- [`config/ml_model/`](config/ml_model/README.md) — autonomous-driving model integration.
- [`controls/`](controls/README.md) — vehicle and scenario control modules.
- [`scenarios/`](scenarios/README.md) — scenario organization and experimental framework.
- [`scenarios/model/`](scenarios/model/README.md) — model-driven experiments.
- [`output/`](output/README.md) — collected experimental data.
- [`output/model/intersection_interception/`](output/model/intersection_interception/README.md) — intersection-interception experiment outputs.
- [`output/model/object_in_road/`](output/model/object_in_road/README.md) — object-in-road experiment outputs.

---

## Project Status

This repository represents an **ongoing undergraduate research project**.

The current software provides the simulation, model-integration, and data-collection infrastructure used to conduct controlled autonomous-driving experiments. Additional scenarios and analysis methods continue to be developed as the research progresses.

---

## Acknowledgment

This research is conducted at the **University of Michigan-Dearborn** and is **sponsored by Ford Motor Company**.
