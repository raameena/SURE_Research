"""PCLA-style world-route navigation for standalone CARLA scenarios."""

import logging
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PCLA_DIR = Path(__file__).resolve().parents[3] / "PCLA"
if str(PCLA_DIR) not in sys.path:
    sys.path.insert(0, str(PCLA_DIR))

from leaderboard_codes.global_route_planner import GlobalRoutePlanner
from leaderboard_codes.global_route_planner_dao import GlobalRoutePlannerDAO
from leaderboard_codes.local_planner import RoadOption
from leaderboard_codes.route_manipulation import downsample_route
from pcla_agents.transfuserv4.nav_planner import RoutePlanner
from pcla_agents.transfuserv4.transfuser_utils import inverse_conversion_2d


ROUTE_PLANNER_MIN_DISTANCE_M = 7.5
ROUTE_PLANNER_MAX_DISTANCE_M = 50.0
ROUTE_DOWNSAMPLE_DISTANCE_M = 50.0
INITIAL_TARGET_MIN_ROUTE_DISTANCE_M = 50.0
ROUTE_TRACE_RESOLUTION_M = 1.0
ROUTE_LOOKAHEAD_M = 250.0
WAYPOINT_STEP_M = 2.0
MAX_WAYPOINT_STEPS = 2000


@dataclass(frozen=True)
class PCLANavigationInput:
    """One route-derived navigation input and its debugging metadata."""

    target_point: np.ndarray
    command_value: int
    command_name: str
    active_route_waypoint_index: int
    active_route_world_x: float
    active_route_world_y: float


def world_to_ego_target(target_world_xy, ego_world_xy, ego_yaw_degrees):
    """Apply tfv4's exact R(yaw).T @ (target - ego) 2-D conversion."""
    return inverse_conversion_2d(
        np.asarray(target_world_xy, dtype=np.float64),
        np.asarray(ego_world_xy, dtype=np.float64),
        math.radians(float(ego_yaw_degrees)),
    ).astype(np.float32)


def _signed_yaw_delta_degrees(current_yaw, candidate_yaw):
    return (float(candidate_yaw) - float(current_yaw) + 180.0) % 360.0 - 180.0


def _select_straightest_continuation(current_waypoint, candidates):
    """Choose a deterministic forward continuation for the standalone route."""
    if not candidates:
        return None

    current_yaw = current_waypoint.transform.rotation.yaw
    return min(
        candidates,
        key=lambda waypoint: (
            abs(_signed_yaw_delta_degrees(
                current_yaw,
                waypoint.transform.rotation.yaw,
            )),
            waypoint.road_id,
            waypoint.section_id,
            waypoint.lane_id,
            waypoint.transform.location.x,
            waypoint.transform.location.y,
        ),
    )


def _find_route_destination(start_waypoint, route_lookahead_m):
    """Follow CARLA's fixed road topology to obtain a distant world endpoint."""
    current = start_waypoint
    distance_travelled = 0.0

    for _ in range(MAX_WAYPOINT_STEPS):
        if distance_travelled >= route_lookahead_m:
            break

        next_waypoint = _select_straightest_continuation(
            current,
            current.next(WAYPOINT_STEP_M),
        )
        if next_waypoint is None:
            break

        step_distance = current.transform.location.distance(
            next_waypoint.transform.location
        )
        if step_distance <= 1e-6:
            raise RuntimeError("CARLA waypoint.next() returned a zero-length route step.")

        distance_travelled += step_distance
        current = next_waypoint

    if distance_travelled < ROUTE_PLANNER_MAX_DISTANCE_M * 2.0:
        raise RuntimeError(
            "Unable to construct a sufficiently long Object in Road route: "
            f"only {distance_travelled:.1f} m available ahead of the ego spawn."
        )

    return current, distance_travelled


def _keep_initial_target_beyond_minimum(world_route, sampled_indices):
    """Keep the first active target at least 50 route-metres from spawn."""
    cumulative_distance = 0.0
    first_target_index = None
    for index in range(1, len(world_route)):
        cumulative_distance += world_route[index - 1][0].location.distance(
            world_route[index][0].location
        )
        if cumulative_distance >= INITIAL_TARGET_MIN_ROUTE_DISTANCE_M:
            first_target_index = index
            break

    if first_target_index is None:
        raise RuntimeError(
            "The generated world route ends before the required initial "
            f"{INITIAL_TARGET_MIN_ROUTE_DISTANCE_M:.1f} m target lookahead."
        )

    # If PCLA's option-change sampling produced an earlier point, retain the
    # first actual maneuver command on the skipped segment while anchoring the
    # navigation target itself beyond the obstacle area. This fixed minimum
    # is route-only and never reads obstacle state or position.
    segment_commands = [
        road_option
        for _, road_option in world_route[1:first_target_index + 1]
        if road_option not in (RoadOption.VOID, RoadOption.LANEFOLLOW)
    ]
    first_target_transform, first_target_command = world_route[first_target_index]
    if segment_commands:
        first_target_command = segment_commands[0]

    sampled_route = [
        world_route[sampled_indices[0]],
        (first_target_transform, first_target_command),
    ]
    sampled_route.extend(
        world_route[index]
        for index in sampled_indices
        if index > first_target_index
    )
    return sampled_route, cumulative_distance


def build_pcla_world_route(world_map, start_waypoint, route_lookahead_m=ROUTE_LOOKAHEAD_M):
    """Trace and downsample a fixed world route using PCLA's own helpers."""
    destination_waypoint, traced_lookahead_m = _find_route_destination(
        start_waypoint,
        route_lookahead_m,
    )

    dao = GlobalRoutePlannerDAO(world_map, ROUTE_TRACE_RESOLUTION_M)
    global_route_planner = GlobalRoutePlanner(dao)
    global_route_planner.setup()
    dense_route = global_route_planner.trace_route(
        start_waypoint.transform.location,
        destination_waypoint.transform.location,
    )
    if len(dense_route) < 2:
        raise RuntimeError("PCLA GlobalRoutePlanner returned fewer than two route points.")

    world_route = [
        (waypoint.transform, road_option)
        for waypoint, road_option in dense_route
    ]
    sampled_indices = downsample_route(
        world_route,
        ROUTE_DOWNSAMPLE_DISTANCE_M,
    )
    sampled_route, initial_target_route_distance = _keep_initial_target_beyond_minimum(
        world_route,
        sampled_indices,
    )
    if len(sampled_route) < 2:
        raise RuntimeError("PCLA route downsampling returned fewer than two route points.")

    logging.info(
        "PCLA navigation route initialized: traced_lookahead=%.1fm, "
        "dense_points=%d, sampled_points=%d, initial_target_route_distance=%.1fm, "
        "planner_min=%.1fm, planner_max=%.1fm.",
        traced_lookahead_m,
        len(world_route),
        len(sampled_route),
        initial_target_route_distance,
        ROUTE_PLANNER_MIN_DISTANCE_M,
        ROUTE_PLANNER_MAX_DISTANCE_M,
    )
    return sampled_route


class PCLAWorldRouteNavigation:
    """Stateful tfv4 RoutePlanner adapter over a fixed CARLA world route."""

    def __init__(self, world_route):
        if len(world_route) < 2:
            raise ValueError("PCLA navigation requires at least two world route points.")

        self._initial_route_length = len(world_route)
        self._route_start_index = 0
        self._route_planner = RoutePlanner(
            ROUTE_PLANNER_MIN_DISTANCE_M,
            ROUTE_PLANNER_MAX_DISTANCE_M,
        )
        self._route_planner.set_route(world_route, gps=False)

        # Exact SensorAgent initialization and second-last command selection
        # reproduce the original one-route-transition command lag.
        self._commands = deque((RoadOption.LANEFOLLOW.value,) * 2, maxlen=2)
        self._target_point_prev = np.array([1e5, 1e5], dtype=np.float64)

    @classmethod
    def from_spawn_waypoint(
        cls,
        world_map,
        start_waypoint,
        route_lookahead_m=ROUTE_LOOKAHEAD_M,
    ):
        return cls(build_pcla_world_route(
            world_map,
            start_waypoint,
            route_lookahead_m=route_lookahead_m,
        ))

    def run_step(self, ego_transform):
        ego_world = np.array([
            ego_transform.location.x,
            ego_transform.location.y,
        ], dtype=np.float64)

        route_length_before = len(self._route_planner.route)
        waypoint_route = self._route_planner.run_step(ego_world)
        self._route_start_index += route_length_before - len(waypoint_route)

        if not waypoint_route:
            raise RuntimeError("PCLA navigation route unexpectedly became empty.")

        target_offset = 1 if len(waypoint_route) > 1 else 0
        target_world, far_command = waypoint_route[target_offset]

        # Preserve SensorAgent's target transition check and commands[-2]
        # behavior instead of issuing the new command immediately.
        if np.not_equal(target_world, self._target_point_prev).all():
            self._target_point_prev = target_world.copy()
            self._commands.append(far_command.value)

        command_value = self._commands[-2]
        command_name = RoadOption(command_value).name
        target_local = world_to_ego_target(
            target_world,
            ego_world,
            ego_transform.rotation.yaw,
        )

        return PCLANavigationInput(
            target_point=target_local,
            command_value=command_value,
            command_name=command_name,
            active_route_waypoint_index=self._route_start_index + target_offset,
            active_route_world_x=float(target_world[0]),
            active_route_world_y=float(target_world[1]),
        )

    @property
    def initial_route_length(self):
        return self._initial_route_length
