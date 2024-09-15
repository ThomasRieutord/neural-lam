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
    epilog="Example: python check_datasets.py --datasets mera_example,meps_example,data/mera_8years_fullres",
)
parser.add_argument(
    "--datasets",
    help="Name (or path) of the dataset",
)
parser.add_argument(
    "--subset",
    default="train",
    help="Subset of sample to look at (train, test, val)",
)
args = parser.parse_args()

if args.datasets is not None:
    datasets = args.datasets.split(",")
else:
    datasets = os.listdir(os.path.join(PACKAGE_ROOTDIR, "data"))

print(f"Checking on {len(datasets)} datasets")

for dataset in datasets:
    print("\n----------------------------")
    print(f"  Dataset {dataset}  ({args.subset})")
    print("----------------------------")
    
    if os.path.isdir(dataset):
        sample_dir = os.path.join(dataset, "samples", args.subset)
        cstfile = os.path.join(dataset, "static", "constants.yaml")
    else:
        sample_dir = os.path.join(PACKAGE_ROOTDIR, "data", dataset, "samples", args.subset)
        cstfile = os.path.join(PACKAGE_ROOTDIR, "data", dataset, "static", "constants.yaml")
    
    with open(cstfile, "r") as yf:
        constants = yaml.safe_load(yf)
    
    print("CONSTANTS:")
    pprint(constants)
    
    print("SHAPES:")
    npy_files = sorted([f for f in os.listdir(sample_dir) if f.endswith("mbr000.npy")])
    npy_file = npy_files[0]
    
    x = np.load(os.path.join(sample_dir, npy_file))
    n_ldt, n_x, n_y, n_vars = x.shape
    print(f"  {npy_file} has data of size {x.shape}")
    print(f"  {n_ldt} time steps in forecast")
    print(f"  {(n_x, n_y)} spatial extent")
    print(f"  {n_vars} atmospheric variables")
    
    print("EXTENT:")
    firstdate = dt.datetime.strptime(npy_files[0].split("_")[1], "%Y%m%d%H")
    seconddate = dt.datetime.strptime(npy_files[1].split("_")[1], "%Y%m%d%H")
    lastdate = dt.datetime.strptime(npy_files[-1].split("_")[1], "%Y%m%d%H")
    print(f"  Starts {firstdate} | Ends {lastdate} | Step {seconddate - firstdate}")
    print(f"  Length = {len(npy_files)} items")
    
