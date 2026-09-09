# Model Integration and Experiment Support

The `config/ml_model/` directory connects CARLA scenarios to the PCLA TransFuser++ autonomous-driving model. It contains model-loading adapters, route inputs, experiment-specific evaluation logic, and structured output tools.

PCLA, TransFuser++, and the pretrained checkpoint are third-party components. This directory contains the integration code used by the research project.

## Directory Structure

```text
ml_model/
├── actors/
│   ├── intersection_interception/
│   └── object_in_road/
├── output_config/
└── setup/
```

## Setup Components

| File | Purpose |
| --- | --- |
| `setup/model_loader.py` | Loads the `tfv4_l6_0` configuration, prepares synchronized camera and LiDAR input, sends live inference requests, and returns model predictions. It also supports local checkpoint loading for the smoke test. |
| `setup/pcla_navigation.py` | Builds a CARLA world route with PCLA tools and supplies ego-relative target points and road commands. |
| `setup/tfpp_controller.py` | Converts predicted checkpoints and target speed into CARLA throttle, brake, and steering through PCLA's direct controller. |
| `setup/error_detection.py` | Compares expected and observed perception/actions and checks stopping-distance timing when applicable. |
| `setup/fetch_transfuser_model.py` | Downloads the configured PCLA checkpoint files from Hugging Face. |
| `setup/test_transfuser.py` | Runs a CPU smoke test with dummy model inputs. |
| `setup/endpoint.txt` | Stores the session-specific base URL used by the remote `/predict` service. |

## Experiment Components

| File / Directory | Purpose |
| --- | --- |
| `actors/object_in_road/` | Spawns chair, pedestrian, red-shirt pedestrian, and vehicle obstacles and evaluates their model perception. |
| `actors/intersection_interception/` | Defines the shared cut-in configuration and early, medium, and late release distances. |
| `actors/interception_car.py` | Controls cut-in behavior, measures longitudinal bumper gap, and defines distance-based expected actions. |
| `actors/T_bone_car.py` | Defines perception and action expectations for the broadside vehicle-conflict scenario. |
| `output_config/object_in_road_output.py` | Records Object-in-Road inference rows, run summaries, metadata, and sensor/log paths. |
| `output_config/intersection_interception_output.py` | Provides the equivalent structured output support for the Intersection Interception experiment. |

## Model-Control Workflow

```text
CARLA camera and LiDAR
          ↓
PCLA-compatible preprocessing
          ↓
Remote model inference
          ↓
Predicted checkpoint and target speed
          ↓
PCLA direct controller
          ↓
CARLA vehicle control
```

Model-driven scenarios also compare semantic predictions and selected actions with scenario-defined expectations. These checks are written to the experiment outputs for later analysis.

## External Requirements

Model-driven scenarios expect:

- a `PCLA/` directory at the project root;
- the configured TransFuser++ checkpoint under PCLA's pretrained-model directory;
- CARLA, PyTorch, NumPy, OpenCV, `timm`, and `requests`; and
- a reachable remote inference service configured in `endpoint.txt`.

The project does not include a dependency lockfile or the remote inference service setup.

