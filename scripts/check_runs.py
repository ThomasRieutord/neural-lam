#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neural-LAM: Graph-based neural weather prediction models for Limited Area Modeling

Check datasets
"""
import argparse
from neural_lam import utils

parser = argparse.ArgumentParser(
    description="Give info about runs",
    epilog="Example: python check_runs.py --runs graph_lam-4x64-08_08_09-4123",
)
parser.add_argument(
    "--runs",
    help="Name (or path) of the dataset",
    default=None
)
args = parser.parse_args()

if args.runs is not None:
    runs = args.runs.split(",")
else:
    runs = args.runs

utils.experiment_summary(runs)
