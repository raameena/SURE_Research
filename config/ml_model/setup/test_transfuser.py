"""
Standalone CPU smoke test for the TransFuser++ (tfv4_l6_0) agent bundled in PCLA.

Uses model_loader.load_model() (the same loading code the live scenario
uses) and runs a single forward pass with dummy tensors shaped from the
model's own saved config. This only confirms the model loads and a forward
pass completes on CPU -- it does not exercise the CARLA integration (see
get_model_prediction() in model_loader.py for that).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import torch

from config.ml_model.setup.model_loader import load_model


def build_dummy_inputs(config, device):
    channels_per_frame = 1 + int(config.use_ground_plane)
    lidar_channels = config.lidar_seq_len * channels_per_frame

    rgb = torch.rand(1, 3, config.camera_height, config.camera_width, device=device)
    lidar_bev = torch.rand(1, lidar_channels, config.lidar_resolution_height, config.lidar_resolution_width, device=device)
    target_point = torch.zeros(1, 2, device=device)
    ego_vel = torch.zeros(1, 1, device=device)
    command = torch.zeros(1, 6, device=device)
    command[0, 3] = 1.0  # "follow lane"-ish default one-hot entry

    return rgb, lidar_bev, target_point, ego_vel, command


def main():
    net = load_model()
    config = net.tfpp_config
    device = net.tfpp_device

    print(f"backbone={config.backbone} camera={config.camera_width}x{config.camera_height} "
          f"lidar_res={config.lidar_resolution_width}x{config.lidar_resolution_height} "
          f"lidar_seq_len={config.lidar_seq_len} use_ground_plane={config.use_ground_plane}")
    print(f"target_speeds (post-SLOWER adjustment)={config.target_speeds}")

    rgb, lidar_bev, target_point, ego_vel, command = build_dummy_inputs(config, device)
    print("Built dummy input tensors")

    print("Running forward pass...")
    with torch.inference_mode():
        outputs = net.forward(
            rgb=rgb,
            lidar_bev=lidar_bev,
            target_point=target_point,
            ego_vel=ego_vel,
            command=command,
        )

    names = [
        "pred_wp", "pred_target_speed", "pred_checkpoint", "pred_semantic",
        "pred_bev_semantic", "pred_depth", "pred_bb_features", "attention_weights",
        "pred_wp_1", "selected_path",
    ]
    print("Forward pass succeeded. Outputs:")
    for name, value in zip(names, outputs):
        if isinstance(value, torch.Tensor):
            print(f"  {name}: tensor shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"  {name}: {value!r}")


if __name__ == "__main__":
    main()
