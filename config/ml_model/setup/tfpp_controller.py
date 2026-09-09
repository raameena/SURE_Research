"""Stateful adapter for the original TransFuser++ direct controller."""

from dataclasses import dataclass
import math

import carla
import torch


@dataclass(frozen=True)
class TFPPControlResult:
    """Control and steering checkpoint selected exactly as PCLA selects it."""

    control: carla.VehicleControl
    checkpoint_x: float
    checkpoint_y: float
    predicted_angle: float


class TFPPDirectController:
    """
    Supplies the state expected by LidarCenterNet.control_pid_direct().

    The live object scenarios load only the checkpoint configuration locally;
    the network itself runs remotely.  Consequently there is no local
    LidarCenterNet instance to own PCLA's stateful PID windows.  This adapter
    creates those windows with PCLA's own PIDController and invokes the
    original, unmodified LidarCenterNet.control_pid_direct method passed in by
    model_loader.
    """

    def __init__(self, config, transfuser_utils, original_control_pid_direct):
        self.config = config
        self.make_histogram = False
        self.speed_histogram = []
        self.turn_controller_direct = transfuser_utils.PIDController(
            k_p=config.turn_kp,
            k_i=config.turn_ki,
            k_d=config.turn_kd,
            n=config.turn_n,
        )
        self.speed_controller_direct = transfuser_utils.PIDController(
            k_p=config.speed_kp,
            k_i=config.speed_ki,
            k_d=config.speed_kd,
            n=config.speed_n,
        )
        self._original_control_pid_direct = original_control_pid_direct

    def run(self, pred_checkpoint, processed_target_speed, signed_forward_speed):
        """Convert raw checkpoint output and target speed into VehicleControl."""
        checkpoint = torch.as_tensor(pred_checkpoint).detach().cpu()
        if checkpoint.ndim != 3 or checkpoint.shape[0] != 1 or checkpoint.shape[-1] != 2:
            raise ValueError(
                "pred_checkpoint must have shape (1, N, 2); "
                f"received {tuple(checkpoint.shape)}."
            )
        if checkpoint.shape[1] < 2:
            raise ValueError("pred_checkpoint must contain at least two checkpoints.")

        # Exact selection and angle conversion from tfv4 SensorAgent.run_step().
        pred_aim_wp = checkpoint[0][1].numpy()
        pred_angle = -math.degrees(
            math.atan2(-pred_aim_wp[1], pred_aim_wp[0])
        ) / 90.0

        speed = torch.tensor([signed_forward_speed], dtype=torch.float32)
        steer, throttle, brake = self._original_control_pid_direct(
            self,
            float(processed_target_speed),
            pred_angle,
            speed,
        )

        control = carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake),
        )
        return TFPPControlResult(
            control=control,
            checkpoint_x=float(pred_aim_wp[0]),
            checkpoint_y=float(pred_aim_wp[1]),
            predicted_angle=float(pred_angle),
        )
