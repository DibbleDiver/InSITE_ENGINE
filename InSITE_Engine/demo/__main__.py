"""Generate the demo site: ``python -m InSITE_Engine.demo [--days N] [--seed S] [--out DIR] [--start DATE]``.

Without ``--out`` the data goes to the ``input_path`` of
``InSITE_Engine/config/config.yaml`` (relative to the current directory, like
the pipeline itself); the ground truth is written next to that folder as
``demo_ground_truth.json``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml

from .generate import DEFAULT_DAYS, DEFAULT_SEED, DEFAULT_START, generate_demo_data

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
FALLBACK_INPUT_PATH = os.path.join("InSITE_Engine", "data", "raw")


def default_input_path() -> str:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg.get("input_path") or FALLBACK_INPUT_PATH
    except OSError:
        return FALLBACK_INPUT_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m InSITE_Engine.demo", description="Generate synthetic demo data with embedded scenarios.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help=f"days of 5-minute telemetry (default {DEFAULT_DAYS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default {DEFAULT_SEED})")
    parser.add_argument("--out", default=None, help="input_path to write into (default: input_path from config.yaml)")
    parser.add_argument("--start", default=DEFAULT_START, help=f"first day, ISO date (default {DEFAULT_START})")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    out = args.out or default_input_path()
    truth = generate_demo_data(out, days=args.days, seed=args.seed, start=args.start)
    print(f"Demo data written to {os.path.abspath(out)} ({args.days} days from {truth['start']}, seed {args.seed})")
    for ds_id, path in truth["files"].items():
        print(f"  {ds_id:24s} {path}")
    print("Embedded scenarios:")
    for sc in truth["scenarios"]:
        tag = "pipeline" if sc["detectable_by_pipeline"] else "Mind"
        print(f"  [{tag:8s}] {sc['id']:14s} {sc['start']} -> {sc['end']}: {sc['name']}")
    print(f"Ground truth: {truth['ground_truth_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
