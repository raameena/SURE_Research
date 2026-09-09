"""
Error detection logic: separates perception failures from decision failures.

perceieved_right  -> did the model's own perception match what was actually
                      there (ground truth), independent of what it then did?
action_correct    -> did the decoded action match the expected safe action,
                      independent of whether perception was right? (a model
                      can brake for the wrong reason and still act correctly)
timely_action     -> gated on action_correct: given the correct action was
                      taken, was it issued while it was still physically
                      achievable?
"""

from typing import Optional

GRAVITY = 9.81


def perceieved_right(expected_perception: str, actual_perception: str) -> bool:
    """
    True if the model's own perception output (already reduced to a string
    comparable to expected_perception -- e.g. "red_light"/"none") matches
    ground truth for this tick.
    """
    return actual_perception == expected_perception


def action_correct(expected_action: str, decision: str) -> bool:
    """
    True if the decoded action matches the expected safe action for this
    tick's ground truth, regardless of whether the model perceived that
    ground truth correctly (see module docstring).
    """
    return decision == expected_action


def timely_action(
    action_correct_val: bool,
    ego_speed_mps: float,
    distance_to_target_m: float,
    mu_friction: float
) -> Optional[bool]:
    """
    Gated on action_correct. Returns None (N/A) if action_correct_val is
    False -- the model never took the correct action, which is already
    flagged by action_correct itself, so "was it timely" isn't a meaningful
    follow-up question. Otherwise True if physical stopping was still
    achievable at the tick the action was issued:

        v^2 / (2 * mu * g) <= distance_to_target

    False if the required stopping distance exceeds what's physically
    available -- i.e. the model chose correctly but too late.

    mu_friction is caller-supplied, not a module constant -- different
    scenarios (surface, weather, vehicle) may need different values.
    """
    if not action_correct_val:
        return None
    required = (ego_speed_mps ** 2) / (2 * mu_friction * GRAVITY)
    return required <= distance_to_target_m


def evaluate_tick(
    expected_perception: str,
    actual_perception: str,
    expected_action: str,
    decision: str,
    ego_speed_mps: float,
    distance_to_target_m: float,
    mu_friction: float
) -> dict:
    """
    Single entry point for the scenario loop to call once per tick.
    expected_perception/expected_action are resolved by the caller from
    whatever's actually relevant this tick (e.g. the current light phase --
    see scenarios/model/intersection/simple_stop_go.py's LIGHT_TO_EXPECTED,
    passed into controls.ml_controls.monitor_with_model_and_act), not
    hardcoded here. Returns a dict ready to drop straight into the tick log
    line:
        {"perceieved_right": bool, "action_correct": bool, "timely_action": bool|None}

    perceieved_right and action_correct are independent of each other (see
    module docstring) -- all four combinations are meaningful:
        perceieved_right=True,  action_correct=True  -> correct
        perceieved_right=True,  action_correct=False -> saw it, decided wrong
        perceieved_right=False, action_correct=False -> didn't see it, acted wrong (expected/common)
        perceieved_right=False, action_correct=True  -> didn't see it, correct anyway (edge case worth flagging)
    """
    p = perceieved_right(expected_perception, actual_perception)
    a = action_correct(expected_action, decision)
    t = timely_action(a, ego_speed_mps, distance_to_target_m, mu_friction)

    return {"perceieved_right": p, "action_correct": a, "timely_action": t}
