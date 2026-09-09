import carla
import logging
from enum import Enum


class VehicleAction(str, Enum):
    """
    High-level vehicle actions.

    Later, the ML model can predict one of these actions.
    Then apply_action() converts that decision into CARLA controls.
    """
    STOP = "stop"
    GO = "go"
    GO_FAST = "go_fast"
    SLOW = "slow"
    COAST = "coast"
    HARD_BRAKE = "hard_brake"
    EMERGENCY_BRAKE = "emergency_brake"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    REVERSE = "reverse"


# -------------------------
# Action configuration
# -------------------------

ACTION_CONTROL_MAP = {
    VehicleAction.STOP: {
        "throttle": 0.0,
        "steer": 0.0,
        "brake": 1.0
    },

    VehicleAction.GO: {
        "throttle": 0.5,
        "steer": 0.0,
        "brake": 0.0
    },

    VehicleAction.GO_FAST: {
        "throttle": 0.8,
        "steer": 0.0,
        "brake": 0.0
    },

    VehicleAction.SLOW: {
        "throttle": 0.2,
        "steer": 0.0,
        "brake": 0.0
    },

    VehicleAction.COAST: {
        "throttle": 0.0,
        "steer": 0.0,
        "brake": 0.0
    },

    VehicleAction.HARD_BRAKE: {
        "throttle": 0.0,
        "steer": 0.0,
        "brake": 1.0
    },

    VehicleAction.EMERGENCY_BRAKE: {
        "throttle": 0.0,
        "steer": 0.0,
        "brake": 1.0,
        "hand_brake": True
    },

    VehicleAction.TURN_LEFT: {
        "throttle": 0.3,
        "steer": -0.4,
        "brake": 0.0
    },

    VehicleAction.TURN_RIGHT: {
        "throttle": 0.3,
        "steer": 0.4,
        "brake": 0.0
    },

    VehicleAction.REVERSE: {
        "throttle": 0.3,
        "steer": 0.0,
        "brake": 0.0,
        "reverse": True
    }
}


# -------------------------
# Core action function
# -------------------------

def apply_action(vehicle, action, **overrides):
    """
    Applies a high-level vehicle action to a CARLA vehicle.

    Example:
        apply_action(vehicle, VehicleAction.GO)
        apply_action(vehicle, "stop")
        apply_action(vehicle, VehicleAction.GO, throttle=0.7)

    The overrides let you adjust values without creating a whole new action.
    """

    if isinstance(action, str):
        action = VehicleAction(action)

    if action not in ACTION_CONTROL_MAP:
        raise ValueError(f"Unknown vehicle action: {action}")

    control_values = ACTION_CONTROL_MAP[action].copy()
    control_values.update(overrides)

    logging.debug(f"Applying action: {action.value} | controls={control_values}")

    control = carla.VehicleControl(
        throttle=control_values.get("throttle", 0.0),
        steer=control_values.get("steer", 0.0),
        brake=control_values.get("brake", 0.0),
        hand_brake=control_values.get("hand_brake", False),
        reverse=control_values.get("reverse", False)
    )

    vehicle.apply_control(control)


# -------------------------
# Convenience action functions
# -------------------------

def drive_forward(vehicle, throttle=0.5):
    """
    Drives the vehicle forward.
    """
    apply_action(vehicle, VehicleAction.GO, throttle=throttle)


def stop_vehicle(vehicle):
    """
    Stops the vehicle.
    """
    apply_action(vehicle, VehicleAction.STOP)


def slow_vehicle(vehicle, throttle=0.2):
    """
    Moves the vehicle slowly forward.
    Useful for creeping toward a traffic light trigger area.
    """
    apply_action(vehicle, VehicleAction.SLOW, throttle=throttle)


def go_fast_vehicle(vehicle, throttle=0.8):
    """
    Drives the vehicle forward quickly.
    """
    apply_action(vehicle, VehicleAction.GO_FAST, throttle=throttle)


def coast_vehicle(vehicle):
    """
    Removes throttle and brake.
    The vehicle may still roll because of momentum.
    """
    apply_action(vehicle, VehicleAction.COAST)


def hard_brake(vehicle):
    """
    Applies full braking force.
    """
    apply_action(vehicle, VehicleAction.HARD_BRAKE)


def emergency_brake(vehicle):
    """
    Applies full brake and hand brake.
    """
    apply_action(vehicle, VehicleAction.EMERGENCY_BRAKE)


def turn_left(vehicle, throttle=0.3, steer=-0.4):
    """
    Turns the vehicle left while moving forward.
    """
    apply_action(
        vehicle,
        VehicleAction.TURN_LEFT,
        throttle=throttle,
        steer=steer
    )


def turn_right(vehicle, throttle=0.3, steer=0.4):
    """
    Turns the vehicle right while moving forward.
    """
    apply_action(
        vehicle,
        VehicleAction.TURN_RIGHT,
        throttle=throttle,
        steer=steer
    )


def reverse_vehicle(vehicle, throttle=0.3):
    """
    Moves the vehicle backward.
    """
    apply_action(vehicle, VehicleAction.REVERSE, throttle=throttle)


# -------------------------
# Speed utilities
# -------------------------

def get_vehicle_speed_mps(vehicle):
    """
    Returns vehicle speed in meters per second.
    """
    velocity = vehicle.get_velocity()

    speed = (
        velocity.x ** 2 +
        velocity.y ** 2 +
        velocity.z ** 2
    ) ** 0.5

    return speed


def get_vehicle_speed_kmh(vehicle):
    """
    Returns vehicle speed in kilometers per hour.
    """
    return get_vehicle_speed_mps(vehicle) * 3.6


def get_vehicle_speed_mph(vehicle):
    """
    Returns vehicle speed in miles per hour.
    """
    return get_vehicle_speed_mps(vehicle) * 2.23694