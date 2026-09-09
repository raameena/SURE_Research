"""Runnable entry point for the Late Interception variant."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from config.ml_model.actors.intersection_interception.late_car import SCENARIO_CONFIG
from scenarios.model.intersection_interception.run_interception_scenario import (
    run_interception_scenario,
)


def main():
    run_interception_scenario(SCENARIO_CONFIG)


if __name__ == "__main__":
    main()
