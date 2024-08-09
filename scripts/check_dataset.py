#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neural-LAM: Graph-based neural weather prediction models for Limited Area Modeling

Check datasets
"""
import argparse
import os
import datetime as dt
import numpy as np
import yaml
from pprint import pprint
from neural_lam import PACKAGE_ROOTDIR

parser = argparse.ArgumentParser(
    description="Check datasets",
    epilog="Example: python check_dataset.py --dataset mera_example",
)
parser.add_argument(
    "--dataset",
    help="Name (or path) of the dataset",
)
args = parser.parse_args()

print("----------------------------")
print(f"  Dataset {args.dataset}  ")
print("----------------------------")

if os.path.isdir(args.dataset):
    sample_dir = os.path.join(args.dataset, "samples", "train")
    cstfile = os.path.join(args.dataset, "static", "constants.yaml")
else:
    sample_dir = os.path.join(PACKAGE_ROOTDIR, "data", args.dataset, "samples", "train")
    cstfile = os.path.join(PACKAGE_ROOTDIR, "data", args.dataset, "static", "constants.yaml")

with open(cstfile, "r") as yf:
    constants = yaml.safe_load(yf)

pprint(constants)

print("----------------------------")
npy_files = sorted([f for f in os.listdir(sample_dir) if f.endswith("mbr000.npy")])
npy_file = npy_files[0]

x = np.load(os.path.join(sample_dir, npy_file))
print(f"{npy_file} has data of size {x.shape}")

print("----------------------------")
firstdate = dt.datetime.strptime(npy_files[0].split("_")[1], "%Y%m%d%H")
seconddate = dt.datetime.strptime(npy_files[1].split("_")[1], "%Y%m%d%H")
lastdate = dt.datetime.strptime(npy_files[-1].split("_")[1], "%Y%m%d%H")
print(f"Starts {firstdate} | Ends {lastdate} | Step {seconddate - firstdate}")
print(f"Length = {len(npy_files)}")
