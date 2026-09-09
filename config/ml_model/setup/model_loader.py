import os
import io
import sys
import math
import types
import pickle
import queue
import logging
import contextlib
from dataclasses import dataclass
from pathlib import Path

import requests

# PCLA is a separately-rooted package (git submodule) that expects its own
# directory (and its transfuserv4 agent's own directory) on sys.path -- add
# both before importing anything from them.
PCLA_DIR = str(Path(__file__).resolve().parents[3] / "PCLA")
AGENT_DIR = str(Path(PCLA_DIR) / "pcla_agents" / "transfuserv4")

if PCLA_DIR not in sys.path:
    sys.path.insert(0, PCLA_DIR)

if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

import numpy as np

# imgaug 0.4.0 (pulled in transitively by the transfuserv4 agent's data.py)
# predates numpy 2.0's removal of these aliases.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }
if not hasattr(np, "string_"):
    np.string_ = np.bytes_

import timm

# TransfuserBackbone.__init__ always calls timm.create_model(..., pretrained=True)
# to seed ImageNet weights, which the checkpoint's own load_state_dict() fully
# overwrites anyway. Skip the download so this doesn't depend on network
# access / a populated Hugging Face cache.
_timm_create_model = timm.create_model


def _create_model_no_pretrained(*args, **kwargs):
    kwargs["pretrained"] = False
    return _timm_create_model(*args, **kwargs)


timm.create_model = _create_model_no_pretrained

import cv2
import carla
import torch
import torch.nn.functional as F

from pcla_functions.setup_sensor_attributes import setup_sensor_attributes
from config.ml_model.setup.tfpp_controller import TFPPDirectController

WEIGHTS_DIR = os.path.join(
    PCLA_DIR, "pcla_agents", "transfuserv4_pretrained", "longest6", "tfpp_all_0"
)

# TransFuser++ v4, longest6 benchmark, seed 0 -- reused for both log messages
# below and for output path construction (see scenarios/model/intersection/
# simple_stop_go.py's create_run_folder() call), so the scenario doesn't need
# to load the model just to know its name.
MODEL_NAME = "tfv4_l6_0"

# Hand-updated with the current Colab ngrok URL each session -- see
# remote_forward(). Gitignored: the URL changes every session, so it isn't
# meant to be committed.
ENDPOINT_FILE = os.path.join(os.path.dirname(__file__), "endpoint.txt")

# Colab's forward pass + network round trip is far slower than a local call
# would be; 3s (the reference client's default) was too tight in practice.
REMOTE_INFERENCE_TIMEOUT_SECONDS = 15.0

# Matches sensor_agent.py's SensorAgent.setup() SLOWER=1 default: classification
# networks are known to be overconfident, which leads them to brake a bit too
# late, so junction/outside-junction target speeds are nudged down slightly.
SLOWER_SPEED_REDUCTION_MPS = 2.0

# Matches leaderboard_codes.local_planner.RoadOption's integer values.
ROAD_OPTION_VALUES = {
    "LEFT": 1,
    "RIGHT": 2,
    "STRAIGHT": 3,
    "LANEFOLLOW": 4,
    "CHANGELANELEFT": 5,
    "CHANGELANERIGHT": 6,
}

# Backward-compatible navigation fallback for experiments that have not opted
# into explicit route-derived target_point/command inputs. Object in Road now
# passes both explicitly; intersection callers continue to use this fallback.
FIXED_COMMAND = "STRAIGHT"

# -------------------------
# Camera mount presets
# -------------------------
#
# Driver's-POV dash-cam mount, per professor's direction -- an intentional
# deviation from the checkpoint's trained camera geometry (x=-1.5, z=2.0,
# roof-mounted, per config.pickle), not a mismatch to "fix" back to it.
# Roughly windshield height and centered left/right, mounted toward the
# windshield (forward of the vehicle's origin) rather than behind it.
#
# Rotation is intentionally left at [0, 0, 0] (dead level, facing straight
# forward) -- transfuser_utils.create_projection_grid(), the model's
# internal camera->BEV geometric fusion (baked into the network once at
# construction time below, as a fixed nn.Parameter), hard-asserts
# camera_rot_0 == [0, 0, 0] and has no support for a rotated/tilted camera
# at all. A pitched-down dash-cam mount would crash model construction, not
# just look different -- so only camera_pos (position) is adjustable here.
DASHCAM_CAMERA_POS = [0.8, 0.0, 1.3]
DASHCAM_CAMERA_ROT = [0.0, 0.0, 0.0]


@contextlib.contextmanager
def _pcla_transfuser_namespace():
    """
    PCLA/pcla_agents/transfuserv4's config.py/model.py/data.py/transfuser_utils.py
    use bare imports (`import config`, `from data import ...`, ...), assuming
    their own directory is the only thing on sys.path providing those names.

    Our own top-level `config` package (CARLA-Research/config/) shares the
    name `config` with PCLA's config.py. Every scenario does
    `from config.CARLA_actors... import ...` before calling load_model(),
    which caches our package under sys.modules['config'] -- so by the time
    PCLA's own code says `from config import GlobalConfig`, Python would
    silently hand back OUR package (which has no GlobalConfig) instead of
    PCLA's config.py, and the import would fail (or worse, resolve to
    whichever happened to be cached first).

    PCLA.py's own setup_agent() hits this exact problem when swapping
    between agents and works around it by clearing these names out of
    sys.modules first. We do the same here -- but keep the substitution in
    place for the *entire* `with` block, not just one import statement:
    pickle.load() on the checkpoint's config.pickle re-resolves the
    GlobalConfig class by looking up sys.modules['config'].GlobalConfig at
    unpickling time (not by any reference grabbed earlier), so restoring too
    early breaks it. Everything that needs 'config'/'model'/'data'/
    'transfuser_utils' to mean PCLA's versions -- the imports below, and the
    caller's pickle.load()/LidarCenterNet(...) -- must happen inside this
    `with` block. Whatever was cached before is restored on exit either way,
    so the rest of this codebase is unaffected afterward.
    """
    conflicting_names = ("config", "model", "data", "transfuser_utils")

    saved_modules = {
        name: sys.modules.pop(name)
        for name in conflicting_names
        if name in sys.modules
    }

    try:
        yield
    finally:
        for name in conflicting_names:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


# -------------------------
# Model loading
# -------------------------

def _load_tfpp_config(camera_mount):
    """
    Loads the checkpoint config, preprocessing helpers, and original direct
    controller method needed by load_model_runtime() and
    load_model() both need before diverging on whether to also construct
    and load the network itself. Factored out so the two don't duplicate
    this loading logic.

    camera_mount selects where the model's input camera is mounted:
        "training": the checkpoint's own trained geometry, untouched
            (config.camera_pos/camera_rot_0 exactly as unpickled).
        "dashcam": DASHCAM_CAMERA_POS/DASHCAM_CAMERA_ROT above -- a
            windshield-height, forward-mounted driver's-POV camera.
    This has to be decided here, before LidarCenterNet(config) is
    constructed (in load_model()), not just when attach_model_sensors()
    spawns the actual sensor -- the model bakes a fixed camera->BEV
    projection geometry (transfuser_utils.create_projection_grid()) into
    itself once at construction time, computed from config.camera_pos/
    camera_rot_0 as they stand at that moment. Overriding the position only
    when spawning the sensor (and leaving the model's construction-time
    config at the training values) would physically move the camera without
    moving the model's internal idea of where it is, silently desyncing the
    two. attach_model_sensors() reads camera_pos/camera_rot_0 back off
    model.tfpp_config, so it automatically follows whichever mount was
    selected here -- this matters even for load_model_runtime(), which never
    constructs a network at all, because it's still what attach_model_sensors()
    reads to spawn the sensor in the right place.
    """
    if camera_mount not in ("training", "dashcam"):
        raise ValueError(
            f"Unknown camera_mount {camera_mount!r} -- expected 'training' or 'dashcam'."
        )

    config_path = os.path.join(WEIGHTS_DIR, "config.pickle")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing {config_path}. Run config/ml_model/setup/fetch_transfuser_model.py first."
        )

    with _pcla_transfuser_namespace():
        from config import GlobalConfig
        from data import CARLA_Data
        from model import LidarCenterNet
        import transfuser_utils as t_u

        with open(config_path, "rb") as config_file:
            loaded_config = pickle.load(config_file)

        config = GlobalConfig()
        config.__dict__.update(loaded_config.__dict__)

        # See SLOWER_SPEED_REDUCTION_MPS above -- matches sensor_agent.py's
        # SLOWER=1 default, which is applied in SensorAgent.setup() rather
        # than baked into the checkpoint's own config.
        config.target_speeds[2] -= SLOWER_SPEED_REDUCTION_MPS
        config.target_speeds[3] -= SLOWER_SPEED_REDUCTION_MPS
        config.brake_uncertainty_threshold = float(
            os.environ.get(
                "UNCERTAINTY_THRESHOLD",
                config.brake_uncertainty_threshold,
            )
        )

        if camera_mount == "dashcam":
            config.camera_pos = list(DASHCAM_CAMERA_POS)
            config.camera_rot_0 = list(DASHCAM_CAMERA_ROT)

        data_helper = CARLA_Data(root=[], config=config, shared_dict=None)

    return config, t_u, data_helper, LidarCenterNet.control_pid_direct


def load_model_runtime(camera_mount="training"):
    """
    Lightweight counterpart to load_model(): loads the checkpoint's config +
    transfuser_utils + data-preprocessing helper only, via _load_tfpp_config()
    -- no LidarCenterNet construction, no model_0030.pth checkpoint load, no
    network on CPU or anywhere else. For the live scenario path, where the
    forward pass now runs remotely (see remote_forward(), called from
    get_model_prediction()) and only tfpp_config/tfpp_transfuser_utils/
    tfpp_data_helper are ever read locally -- the network itself is never
    invoked here.

    Returns a plain object (types.SimpleNamespace, not a LidarCenterNet)
    stamped with the same tfpp_* attributes load_model() stamps on its real
    network. Nothing in this codebase isinstance()-checks the return value
    of load_model()/load_model_runtime(), so get_model_prediction() and
    attach_model_sensors() -- which only ever read attributes off it -- work
    unchanged against either.

    tfpp_device is still set to CPU: get_model_prediction() builds its
    input tensors on this device before sending them to remote_forward(),
    matching the Colab bridge's contract of CPU tensors over the wire.
    """
    config, t_u, data_helper, original_control_pid_direct = _load_tfpp_config(camera_mount)

    device = torch.device("cpu")

    runtime = types.SimpleNamespace(
        tfpp_config=config,
        tfpp_device=device,
        tfpp_transfuser_utils=t_u,
        tfpp_data_helper=data_helper,
        tfpp_rgb_queue=None,
        tfpp_lidar_queue=None,
        tfpp_camera_mount=camera_mount,
        tfpp_lidar_previous_sample=None,
        tfpp_uncertainty_weight=int(os.environ.get("UNCERTAINTY_WEIGHT", 1)),
        tfpp_controller=TFPPDirectController(
            config,
            t_u,
            original_control_pid_direct,
        ),
    )

    logging.debug(
        f"Loaded {MODEL_NAME} config for remote inference "
        f"(backbone={config.backbone}, target_speeds={config.target_speeds}, "
        f"camera_mount={camera_mount}, camera_pos={config.camera_pos}, "
        f"camera_rot_0={config.camera_rot_0})."
    )

    return runtime


def load_model(camera_mount="training"):
    """
    Loads the tfv4_l6_0 (TransFuser++ v4, longest6 benchmark, seed 0)
    checkpoint directly on CPU -- the same GlobalConfig/LidarCenterNet/
    load_state_dict loading test_transfuser.py already used, factored out
    here so both scripts share it instead of duplicating it.

    Deliberately does NOT instantiate PCLA (see PCLA.py): PCLA.__init__
    mandatorily runs setup_route()/setup_sensors(), which pulls in the
    leaderboard's GPS/UKF route-follower pipeline -- the exact machinery
    responsible for last night's circling bug, and there is no supported way
    to instantiate PCLA and skip it. Loading the network directly gives full
    control over what's fed to it each call, with none of that pipeline
    involved at all.

    Constructs and loads the real network on CPU -- see load_model_runtime()
    for the lighter config-only counterpart the live scenario now uses,
    now that the forward pass itself runs on the Colab GPU bridge. This
    full version remains for test_transfuser.py's standalone local smoke
    test, and any other local (Colab-independent) use.

    The returned model is an eval-mode LidarCenterNet on CPU, with a few
    extra attributes stashed on it (device, config, and a data-preprocessing
    helper) so get_model_prediction() and attach_model_sensors() have
    everything they need from this one object -- matching "load once, pass
    around" rather than threading a second config parameter through every
    call.
    """
    config, t_u, data_helper, original_control_pid_direct = _load_tfpp_config(camera_mount)

    device = torch.device("cpu")

    weight_path = os.path.join(WEIGHTS_DIR, "model_0030.pth")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"Missing {weight_path}. Run config/ml_model/setup/fetch_transfuser_model.py first."
        )

    with _pcla_transfuser_namespace():
        from model import LidarCenterNet

        net = LidarCenterNet(config)

        if config.sync_batch_norm:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

        state_dict = torch.load(weight_path, map_location=device)
        net.load_state_dict(state_dict, strict=False)

        net.to(device)
        net.eval()

    net.tfpp_config = config
    net.tfpp_device = device
    net.tfpp_transfuser_utils = t_u
    net.tfpp_data_helper = data_helper
    net.tfpp_rgb_queue = None
    net.tfpp_lidar_queue = None
    net.tfpp_camera_mount = camera_mount
    net.tfpp_lidar_previous_sample = None
    net.tfpp_uncertainty_weight = int(os.environ.get("UNCERTAINTY_WEIGHT", 1))
    net.tfpp_controller = TFPPDirectController(
        config,
        t_u,
        original_control_pid_direct,
    )

    logging.info(
        f"Loaded {MODEL_NAME} model on {device} "
        f"(backbone={config.backbone}, target_speeds={config.target_speeds}, "
        f"camera_mount={camera_mount}, camera_pos={config.camera_pos}, "
        f"camera_rot_0={config.camera_rot_0})."
    )

    return net


# -------------------------
# Model input sensors
# -------------------------

def attach_model_sensors(world, bp_lib, vehicle, model):
    """
    Attaches the camera + LiDAR sensors the tfv4_l6_0 checkpoint was trained
    on -- a different mount position/resolution/FOV than
    config.CARLA_actors.sensors.attach_standard_sensors() (which is for our
    own image/point-cloud recording, not model input).

    Sensor frames are pushed into a queue.Queue per sensor (model.tfpp_rgb_queue /
    model.tfpp_lidar_queue) rather than stashed as a single "latest" attribute --
    get_model_prediction() pops from these queues matched by exact .frame
    number against the current simulation tick (see _retrieve_sensor_frame()),
    not "whatever was last written". A blind-overwrite single-attribute
    approach has a race even in synchronous mode: world.tick() returning
    doesn't guarantee every sensor's listen() callback (a separate CARLA
    thread) has already fired for that frame, so a plain "read the latest
    attribute" could silently hand back the previous tick's frame instead --
    stale relative to the tick that was actually just requested.

    Requires the world to already be in synchronous mode (see
    carla_config.enable_synchronous_mode()) -- the warm-up loop below calls
    world.tick() explicitly rather than world.wait_for_tick(), which would
    otherwise block forever waiting for a tick nothing is producing.

    Returns:
        [camera_sensor, lidar_sensor] -- append these to the scenario's
        spawned_actors list so cleanup_actors() destroys them normally.
    """
    config = model.tfpp_config

    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp = setup_sensor_attributes(camera_bp, {
        "type": "sensor.camera.rgb",
        "width": config.camera_width,
        "height": config.camera_height,
        "fov": config.camera_fov,
    })
    camera_transform = carla.Transform(
        carla.Location(
            x=config.camera_pos[0],
            y=config.camera_pos[1],
            z=config.camera_pos[2]
        ),
        carla.Rotation(
            roll=config.camera_rot_0[0],
            pitch=config.camera_rot_0[1],
            yaw=config.camera_rot_0[2]
        )
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    rgb_queue = queue.Queue()
    camera.listen(rgb_queue.put)

    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp = setup_sensor_attributes(lidar_bp, {"type": "sensor.lidar.ray_cast"})
    lidar_transform = carla.Transform(
        carla.Location(
            x=config.lidar_pos[0],
            y=config.lidar_pos[1],
            z=config.lidar_pos[2]
        ),
        carla.Rotation(
            roll=config.lidar_rot[0],
            pitch=config.lidar_rot[1],
            yaw=config.lidar_rot[2]
        )
    )
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar_queue = queue.Queue()

    def capture_lidar_half_sweep(measurement):
        # Store every LiDAR callback, not just inference frames.  The pose is
        # captured with the measurement so get_model_prediction() can align
        # the immediately preceding half-sweep even while inference remains
        # throttled to every five CARLA ticks.
        ego_transform = vehicle.get_transform()
        ego_pose = (
            float(ego_transform.location.x),
            float(ego_transform.location.y),
            math.radians(float(ego_transform.rotation.yaw)),
        )
        lidar_queue.put((measurement, ego_pose))

    lidar.listen(capture_lidar_half_sweep)

    model.tfpp_rgb_queue = rgb_queue
    model.tfpp_lidar_queue = lidar_queue
    model.tfpp_lidar_previous_sample = None

    # Ensure both sensors have produced at least one queued frame before the
    # first get_model_prediction() call -- mirrors PCLA.setup_sensors()'s own
    # world.tick() right after spawning the agent's sensors, for the same
    # reason. Frames are left in the queues (not consumed here) for
    # get_model_prediction()'s frame-matching to retrieve.
    for _ in range(50):
        world.tick()
        if not rgb_queue.empty() and not lidar_queue.empty():
            break
    else:
        raise RuntimeError(
            "Model input sensors (camera/LiDAR) produced no data after 50 ticks."
        )

    logging.debug("Model input sensors (camera + LiDAR) attached.")

    return [camera, lidar]


def _retrieve_sensor_frame(sensor_queue, target_frame, timeout=2.0):
    """
    Pops from sensor_queue (fed by a sensor's listen() callback) until the
    item whose .frame matches target_frame is found, discarding any older
    frames along the way.

    Closes the race described in attach_model_sensors()'s docstring: even in
    synchronous mode, world.tick() returning doesn't guarantee the sensor's
    callback has already fired for that exact frame, so matching by frame
    number (CARLA's own documented CarlaSyncMode pattern) is what actually
    guarantees "the frame we get back is the one this call asked for", not
    just "some frame that happened to be sitting there".
    """
    while True:
        try:
            data = sensor_queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(
                f"No sensor frame matching tick {target_frame} arrived within {timeout}s."
            )

        if data.frame == target_frame:
            return data

        if data.frame > target_frame:
            raise RuntimeError(
                f"Missed sensor frame for tick {target_frame} -- queue is "
                f"already at frame {data.frame}."
            )

        # data.frame < target_frame: stale frame from before the one we
        # want (e.g. left over from attach_model_sensors()'s warm-up) --
        # discard it and keep looking.


@dataclass(frozen=True)
class TFPPPrediction:
    """Outputs needed by the original tfv4_l6 direct-control policy."""

    processed_target_speed: float
    pred_checkpoint: torch.Tensor | None
    pred_semantic: torch.Tensor
    pred_bev_semantic: torch.Tensor
    target_speed_probabilities: dict
    brake_probability: float
    signed_forward_speed: float


def get_signed_forward_speed(vehicle):
    """Match PCLA's speedometer: project velocity onto ego's forward axis."""
    velocity = vehicle.get_velocity()
    forward = vehicle.get_transform().get_forward_vector()
    return float(
        velocity.x * forward.x
        + velocity.y * forward.y
        + velocity.z * forward.z
    )


def _retrieve_lidar_half_sweeps(model, target_frame, timeout=2.0):
    """Return the immediately previous and current frame's LiDAR samples."""
    while True:
        try:
            measurement, ego_pose = model.tfpp_lidar_queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(
                f"No LiDAR frame matching tick {target_frame} arrived within {timeout}s."
            )

        sample = (measurement, ego_pose)
        if measurement.frame > target_frame:
            raise RuntimeError(
                f"Missed LiDAR frame for tick {target_frame} -- queue is "
                f"already at frame {measurement.frame}."
            )

        previous = model.tfpp_lidar_previous_sample
        model.tfpp_lidar_previous_sample = sample

        if measurement.frame == target_frame:
            if previous is None or previous[0].frame != target_frame - 1:
                return None, sample
            return previous, sample


def _lidar_measurement_to_ego(config, transfuser_utils, measurement):
    points = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
    return transfuser_utils.lidar_to_ego_coordinate(config, (None, points))


def _align_previous_lidar_to_current(
    previous_lidar,
    previous_pose,
    current_pose,
    transfuser_utils,
):
    """Exact pose-difference math from tfv4 SensorAgent.align_lidar()."""
    x, y, orientation = previous_pose
    x_target, y_target, orientation_target = current_pose
    pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
    rot_diff = transfuser_utils.normalize_angle(orientation_target - orientation)

    rotation_matrix = np.array([
        [np.cos(orientation_target), -np.sin(orientation_target), 0.0],
        [np.sin(orientation_target), np.cos(orientation_target), 0.0],
        [0.0, 0.0, 1.0],
    ])
    pos_diff = rotation_matrix.T @ pos_diff
    return transfuser_utils.algin_lidar(previous_lidar, pos_diff, rot_diff)


# -------------------------
# Model inference
# -------------------------

def _read_endpoint_url():
    """
    Reads ENDPOINT_FILE fresh on every call (not cached) -- it's hand-updated
    mid-session whenever the Colab notebook is restarted, and a cached value
    would keep pointing at a dead tunnel until this process restarted too.
    """
    with open(ENDPOINT_FILE, "r") as endpoint_file:
        return endpoint_file.read().strip()


# Reused across calls instead of a fresh connection per request -- the
# per-request TCP+TLS handshake to the ngrok tunnel is pure overhead when
# every check hits the same host, and requests.post() (module-level
# function) opens a new connection every time it's called.
_HTTP_SESSION = requests.Session()


def remote_forward(rgb_jpeg_bytes, lidar_bev, target_point, ego_vel, command, endpoint_url,
                    timeout=REMOTE_INFERENCE_TIMEOUT_SECONDS):
    """
    Sends one forward pass to the Colab GPU bridge's POST {endpoint}/predict
    and returns its response dict ({pred_target_speed, pred_checkpoint,
    pred_semantic, pred_bev_semantic}), or None if the endpoint is unreachable/times out/
    errors -- callers hold the last decision rather than crash the scenario
    over a single dropped request.

    rgb_jpeg_bytes is the already-JPEG-encoded camera frame (a few hundred
    KB), not a raw float32 tensor (~3MB at this checkpoint's 1024x256
    resolution) -- the server decodes it back into the CHW float tensor the
    model expects. pred_semantic/pred_bev_semantic in the response are
    likewise expected back as small uint8 argmaxed class-index maps
    (~320KB combined) rather than raw per-class logits (~10MB combined) --
    see interpret_semantic_output()'s docstring in controls.ml_controls,
    which only ever needed the argmax anyway. Together this cuts each
    round trip from ~13-14MB to well under 1MB, which is most of what made
    each decision check take 10+ seconds over a free ngrok tunnel.
    """
    inputs = {
        "rgb": rgb_jpeg_bytes,
        "lidar_bev": lidar_bev,
        "target_point": target_point,
        "ego_vel": ego_vel,
        "command": command,
    }

    buffer = io.BytesIO()
    torch.save(inputs, buffer)

    try:
        response = _HTTP_SESSION.post(f"{endpoint_url}/predict", data=buffer.getvalue(), timeout=timeout)
        response.raise_for_status()
        return torch.load(io.BytesIO(response.content), map_location="cpu", weights_only=False)
    except requests.exceptions.RequestException as e:
        logging.debug("Remote inference request failed: %s", e)
        return None


def _get_model_prediction_result(
    model,
    vehicle,
    world,
    forward_distance,
    use_pcla_speed_processing,
    require_pred_checkpoint,
    target_point=None,
    command=None,
):
    """
    Captures the vehicle's current camera/LiDAR frame (as delivered by the
    sensors attach_model_sensors() attached) and runs one forward pass.

    Route-aware callers pass an already resolved ego-local target_point and
    RoadOption command. Callers that omit both retain the historical
    (forward_distance, 0.0) + STRAIGHT fallback for backward compatibility.
    Supplying only one is rejected so a route target can never be paired
    silently with an unrelated synthetic command.

    world.get_snapshot().frame is used as the target frame to retrieve from
    model.tfpp_rgb_queue/tfpp_lidar_queue (see _retrieve_sensor_frame()) --
    in synchronous mode this is always the frame of the tick the caller most
    recently advanced to (whether via this scenario's own world.tick() call
    at the bottom of its control loop, or via force_traffic_light_state()'s
    advance_simulation()), so every call here is matched to "whatever just
    happened", not a stale leftover frame from a previous tick.

    Returns a TFPPPrediction, or None when a complete adjacent LiDAR pair or
    remote response is unavailable.  The control path contains the PCLA-
    processed target speed and raw predicted checkpoints; the legacy wrapper
    below retains the historical four-value API for other experiments.
    """
    config = model.tfpp_config
    device = model.tfpp_device
    t_u = model.tfpp_transfuser_utils
    data_helper = model.tfpp_data_helper

    if model.tfpp_rgb_queue is None or model.tfpp_lidar_queue is None:
        raise RuntimeError(
            "get_model_prediction() called before the model's input sensors "
            "were attached -- call attach_model_sensors() first."
        )

    target_frame = world.get_snapshot().frame
    rgb_image = _retrieve_sensor_frame(model.tfpp_rgb_queue, target_frame)
    previous_lidar_sample, current_lidar_sample = _retrieve_lidar_half_sweeps(
        model,
        target_frame,
    )

    if previous_lidar_sample is None:
        logging.warning(
            "PCLA LiDAR warm-up: frame %s has no immediately previous half-sweep; "
            "skipping inference and retaining the safe brake command.",
            target_frame,
        )
        return None

    # -------------------------
    # Camera -> RGB tensor
    # -------------------------
    # CARLA's raw camera buffer is BGRA; drop alpha to get BGR.
    bgra = np.frombuffer(rgb_image.raw_data, dtype=np.uint8).reshape(
        (rgb_image.height, rgb_image.width, 4)
    )
    bgr = np.copy(bgra[:, :, :3])

    # Also add jpg artifacts at inference time, because the training data was
    # saved as jpg -- matches sensor_agent.py's tick(). The encoded bytes
    # (a few hundred KB) are what actually gets sent to remote_forward() --
    # the server decodes them back into the CHW float tensor the model
    # expects, so the client never needs to build that ~3MB tensor itself.
    _, compressed = cv2.imencode(".jpg", bgr)
    rgb_jpeg_bytes = compressed.tobytes()

    # -------------------------
    # LiDAR -> BEV histogram tensor
    # -------------------------
    # At 10 Hz LiDAR / 20 Hz simulation, each callback is approximately one
    # half rotation.  Match SensorAgent.run_step(): transform both halves to
    # ego coordinates, align the previous half into the current ego frame,
    # then voxelize the concatenated full sweep.
    previous_measurement, previous_pose = previous_lidar_sample
    current_measurement, current_pose = current_lidar_sample
    previous_ego_lidar = _lidar_measurement_to_ego(config, t_u, previous_measurement)
    current_ego_lidar = _lidar_measurement_to_ego(config, t_u, current_measurement)
    previous_aligned_lidar = _align_previous_lidar_to_current(
        previous_ego_lidar,
        previous_pose,
        current_pose,
        t_u,
    )
    ego_lidar = np.concatenate((current_ego_lidar, previous_aligned_lidar), axis=0)
    lidar_histogram = data_helper.lidar_to_histogram_features(
        ego_lidar, use_ground_plane=config.use_ground_plane
    )
    lidar_bev = torch.from_numpy(lidar_histogram).unsqueeze(0).to(device, dtype=torch.float32)

    # -------------------------
    # Navigation conditioning
    # -------------------------
    if (target_point is None) != (command is None):
        raise ValueError("target_point and command must be supplied together.")

    if target_point is None:
        target_point = np.array([forward_distance, 0.0], dtype=np.float32)
        command_value = ROAD_OPTION_VALUES[FIXED_COMMAND]
    else:
        target_point = np.asarray(target_point, dtype=np.float32)
        if target_point.shape != (2,):
            raise ValueError(
                "target_point must have shape (2,) in ego-local coordinates; "
                f"received {target_point.shape}."
            )

        if isinstance(command, str):
            try:
                command_value = ROAD_OPTION_VALUES[command.upper()]
            except KeyError as error:
                raise ValueError(f"Unknown RoadOption command {command!r}.") from error
        else:
            command_value = int(getattr(command, "value", command))

        if command_value not in ROAD_OPTION_VALUES.values():
            raise ValueError(
                f"RoadOption command must be one of {sorted(ROAD_OPTION_VALUES.values())}; "
                f"received {command_value}."
            )

    target_point_tensor = torch.from_numpy(target_point).unsqueeze(0).to(device, dtype=torch.float32)
    command_one_hot = t_u.command_to_one_hot(command_value)
    command_tensor = torch.from_numpy(command_one_hot[np.newaxis]).to(device, dtype=torch.float32)

    # -------------------------
    # Ego velocity
    # -------------------------
    speed_mps = get_signed_forward_speed(vehicle)
    velocity_tensor = torch.tensor([[speed_mps]], device=device, dtype=torch.float32)

    # -------------------------
    # Forward pass (remote -- see remote_forward())
    # -------------------------
    outputs = remote_forward(
        rgb_jpeg_bytes,
        lidar_bev,
        target_point_tensor,
        velocity_tensor,
        command_tensor,
        _read_endpoint_url(),
    )

    if outputs is None:
        return None

    if not isinstance(outputs, dict):
        raise TypeError(
            "Remote /predict response must be a dictionary; "
            f"received {type(outputs).__name__}."
        )
    required_output_keys = {"pred_target_speed", "pred_semantic", "pred_bev_semantic"}
    missing_output_keys = sorted(required_output_keys.difference(outputs))
    if missing_output_keys:
        raise RuntimeError(
            "Remote /predict response is missing required keys: "
            + ", ".join(missing_output_keys)
        )

    # The control path additionally requires pred_checkpoint.  Older callers
    # retain compatibility until their experiment is migrated explicitly.
    raw_pred_target_speed = outputs["pred_target_speed"]
    pred_semantic = outputs["pred_semantic"]
    pred_bev_semantic = outputs["pred_bev_semantic"]
    pred_checkpoint = outputs.get("pred_checkpoint")

    if require_pred_checkpoint and pred_checkpoint is None:
        raise RuntimeError(
            "Remote /predict response is missing required 'pred_checkpoint'. "
            "Update the Colab bridge to return model.forward() output index 2 "
            "with shape (1, N, 2)."
        )

    raw_pred_target_speed = torch.as_tensor(raw_pred_target_speed)
    if raw_pred_target_speed.ndim != 2 or tuple(raw_pred_target_speed.shape) != (1, len(config.target_speeds)):
        raise ValueError(
            "pred_target_speed must contain raw logits with shape "
            f"(1, {len(config.target_speeds)}); received "
            f"{tuple(raw_pred_target_speed.shape)}."
        )

    probabilities = F.softmax(raw_pred_target_speed[0], dim=0)
    brake_probability = float(probabilities[0].item())

    if use_pcla_speed_processing and model.tfpp_uncertainty_weight:
        # Exact default UNCERTAINTY_WEIGHT=1 branch from tfv4 SensorAgent:
        # threshold the brake bin; otherwise use the expected target speed.
        if brake_probability > config.brake_uncertainty_threshold:
            pred_target_speed = float(config.target_speeds[0])
        else:
            pred_target_speed = float(
                sum(
                    probability * speed
                    for probability, speed in zip(
                        probabilities.detach().cpu().numpy(),
                        config.target_speeds,
                    )
                )
            )
    else:
        # Preserve PCLA's UNCERTAINTY_WEIGHT=0 behavior, and the legacy API's
        # existing behavior for experiments outside object_in_road.
        speed_index = int(torch.argmax(probabilities).item())
        pred_target_speed = float(config.target_speeds[speed_index])

    probabilities_numpy = probabilities.detach().cpu().numpy()
    target_speed_probabilities = {
        float(speed): float(probability)
        for speed, probability in zip(config.target_speeds, probabilities_numpy)
    }

    if pred_checkpoint is not None:
        pred_checkpoint = torch.as_tensor(pred_checkpoint).detach().cpu()

    return TFPPPrediction(
        processed_target_speed=pred_target_speed,
        pred_checkpoint=pred_checkpoint,
        pred_semantic=pred_semantic,
        pred_bev_semantic=pred_bev_semantic,
        target_speed_probabilities=target_speed_probabilities,
        brake_probability=brake_probability,
        signed_forward_speed=speed_mps,
    )


def get_model_prediction(model, vehicle, world, forward_distance=15.0):
    """Legacy four-value inference API retained outside object_in_road."""
    result = _get_model_prediction_result(
        model,
        vehicle,
        world,
        forward_distance,
        use_pcla_speed_processing=False,
        require_pred_checkpoint=False,
    )
    if result is None:
        return None, None, None, None
    return (
        result.processed_target_speed,
        result.pred_semantic,
        result.pred_bev_semantic,
        result.target_speed_probabilities,
    )


def get_model_control_prediction(
    model,
    vehicle,
    world,
    forward_distance=15.0,
    target_point=None,
    command=None,
):
    """Return tfv4_l6 outputs, optionally using explicit route navigation."""
    return _get_model_prediction_result(
        model,
        vehicle,
        world,
        forward_distance,
        use_pcla_speed_processing=True,
        require_pred_checkpoint=True,
        target_point=target_point,
        command=command,
    )
