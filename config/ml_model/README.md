# Model Integration and Experiment Support

This directory implements the repository-specific bridge between CARLA scenarios and an external PCLA TransFuser++ model. PCLA and the pretrained model are dependencies; their implementations are not included here.

## Organization

### `setup/`

- `model_loader.py` loads the `tfv4_l6_0` configuration and, for the local smoke-test path, its checkpoint. Live scenarios use `load_model_runtime()`, which keeps preprocessing and control support locally while sending forward passes to the configured remote `/predict` endpoint. It also creates frame-matched camera/LiDAR queues and converts returned predictions into a common result structure.
- `pcla_navigation.py` uses PCLA route-planning utilities to construct a CARLA world route and provide ego-local target points with lagged road commands.
- `tfpp_controller.py` adapts predicted checkpoints and target speed to PCLA's direct PID controller, returning a CARLA `VehicleControl`.
- `error_detection.py` compares scenario-defined expected perception/actions with observed model decisions and evaluates stopping-distance timeliness when applicable.
- `fetch_transfuser_model.py` downloads the explicitly listed PCLA checkpoint members from the configured Hugging Face repository into the PCLA directory structure.
- `test_transfuser.py` is a CPU smoke test using dummy inputs; it is not an experiment or a performance evaluation.
- `endpoint.txt` stores the remote inference base URL consumed by `model_loader.py`. Source comments describe this value as session-specific.

### `actors/`

The actor modules separate scenario-specific ground truth and hazard geometry from scenario orchestration:

- `object_in_road/` supplies chair, pedestrian, red-shirt pedestrian, and vehicle spawning/perception logic. Pedestrian and vehicle variants query corresponding BEV semantic classes; the chair variant uses non-drivable occupancy near the projected object position because the documented class map has no chair class.
- `intersection_interception/` defines a shared interception configuration plus early, medium, and late release distances.
- `interception_car.py` implements cut-in control, longitudinal bumper-gap measurement, expected-action tiers, and vehicle-perception resolution.
- `T_bone_car.py` provides corresponding expectation and perception logic for the broadside vehicle-conflict scenario.

### `output_config/`

`object_in_road_output.py` and `intersection_interception_output.py` create experiment/run folders, write per-inference and aggregate CSV rows, maintain metadata, emit data dictionaries, and finalize sensor and collision/GPS records. Their column definitions are the authoritative description of the structured tables.

## Local and External Boundaries

The code in this directory is integration, orchestration, evaluation, and output support developed for this research project. It imports PCLA modules such as `pcla_agents`, `leaderboard_codes`, and `pcla_functions`; uses a PCLA TransFuser++ checkpoint; and depends on CARLA, PyTorch, NumPy, OpenCV, `timm`, `requests`, and Hugging Face Hub. Those third-party systems are not part of this copied source.

No dependency lockfile is present, and the source does not establish a complete compatible version matrix. Model runs also depend on the external remote inference service expected by `model_loader.py`.

