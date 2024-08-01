#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neural-LAM: Graph-based neural weather prediction models for Limited Area Modeling

Sandbox for dataset instantiation and manipulation

For the variables index, here is the order:
    0    -> air_pressure_at_surface_level            (pres_0g)
    1    -> air_pressure_at_sea_level                (pres_0s)
    2    -> net_upward_longwave_flux_in_air          (nlwrs)
    3    -> net_upward_shortwave_flux_in_air         (nswrs)
    4    -> relative_humidity_at_2_metres            (r_2)
    5    -> relative_humidity_at_12_metres           (r_65)
    6    -> air_temperature_at_2_metres              (t_2)
    7    -> air_temperature_at_12_metres             (t_65)
    8    -> air_temperature_at_500_hPa               (t_500)
    9    -> air_temperature_at_850_hPa               (t_850)
    10   -> eastward_wind_at_12_metres               (u_65)
    11   -> northward_wind_at_12_metres              (v_65)
    12   -> eastward_wind_at_850_hPa                 (u_850)
    13   -> northward_wind_at_850_hPa                (v_850)
    14   -> atmosphere_mass_content_of_water_vapor   (wvint)
    15   -> z_height_above_ground                    (* UNUSED *)
    16   -> geopotential_at_500_hPa                  (z_500)
    17   -> geopotential_at_1000_hPa                 (z_1000)
"""
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import datetime as dt

# First-party
from metplotlib import plots
from neural_lam import package_rootdir
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.hi_lam import HiLAM
from neural_lam.models.hi_lam_parallel import HiLAMParallel
from neural_lam.weather_dataset import WeatherDataset

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
runid = "offline-run-20240408_092733-y8jh1fn6"
modelid = "graph_lam-4x64-07_09_10-7607"
idx = 79  # sample index
ldt_idx = 4  # leadtime index (ldt_idx * 3h)
il_idx = 1  # isoline var: air_pressure_at_sea_level
cl_idx = 6  # color level: air_temperature_at_2_metres

# Load model
# ----------
ckptpath = os.path.join(
    package_rootdir, "saved_models", modelid, "min_val_loss.ckpt"
)
print(f"Loading checkpoint from {ckptpath}")
ckpt = torch.load(ckptpath, map_location=device)


saved_args = ckpt["hyper_parameters"]["args"]
datasetname = saved_args.dataset
batch_size = saved_args.batch_size

# Load model parameters Use new args for model
model_class = MODELS[saved_args.model]
model = model_class.load_from_checkpoint(ckptpath, args=saved_args)


# Load some data
# --------------
wds = WeatherDataset(
    datasetname, subsample_step=1, split="val", control_only=True
)
print(f"Dataset {datasetname} has {len(wds)} samples")

init_states, target_states, forcing = wds[idx]
print(f"Getting data from sample {idx}: {wds.sample_names[idx]}")
print(f"Shapes: {init_states.shape}, {target_states.shape}, {forcing.shape}")
basetime = dt.datetime.strptime(wds.sample_names[idx].split("_")[0], "%Y%m%d%H")

init_states = init_states.to(device).unsqueeze(0)
target_states = target_states.to(device).unsqueeze(0)
forcing = forcing.to(device).unsqueeze(0)

# Load lat/lon
lonlatfile = os.path.join(
    os.path.dirname(os.path.dirname(wds.sample_dir_path)),
    "static",
    "nwp_xy.npy",
)
xy = np.load(lonlatfile)

lambert_proj_params = wds.constants["LAMBERT_PROJ_PARAMS"]
crs = ccrs.LambertConformal(
    central_longitude=lambert_proj_params["lon_0"],
    central_latitude=lambert_proj_params["lat_0"],
    standard_parallels=(
        lambert_proj_params["lat_1"],
        lambert_proj_params["lat_2"],
    ),
)


# Apply model
# -----------
pred_states, _ = model.unroll_prediction(init_states, forcing, target_states)
print(
    f"Shapes: pred_states={pred_states.shape} target_states={target_states.shape}"
)
if wds.standardize:
    std = wds.data_std.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
    mean = wds.data_mean.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
    pred_states = pred_states * std + mean
    target_states = target_states * std + mean

# Reshape and detach
ex = lambda x: x.reshape(wds.constants["GRID_SHAPE"]).detach().cpu().numpy()
for ldt_idx in range(10):
    pred_mslp = ex(pred_states[0, ldt_idx, :, il_idx]) / 100
    true_mslp = ex(target_states[0, ldt_idx, :, il_idx]) / 100
    pred_t2m = ex(pred_states[0, ldt_idx, :, cl_idx]) - 273.15
    true_t2m = ex(target_states[0, ldt_idx, :, cl_idx]) - 273.15

    # Crop the borders
    bw = (
        model.border_mask.reshape(wds.constants["GRID_SHAPE"])
        .sum(axis=0)
        .min()
        .short()
        .item()
    )
    pred_mslp = pred_mslp[bw:-bw, bw:-bw]
    true_mslp = true_mslp[bw:-bw, bw:-bw]
    pred_t2m = pred_t2m[bw:-bw, bw:-bw]
    true_t2m = true_t2m[bw:-bw, bw:-bw]

    titles = np.array(
        [
            ["Predicted", "Target"],
            ["Diff (MSLP)", "Diff (T2m)"],
        ]
    )
    fig, axs = plots.twovar_comparison(
        pred_mslp,
        true_mslp,
        pred_t2m,
        true_t2m,
        lons=xy[0, bw:-bw, bw:-bw],
        lats=xy[1, bw:-bw, bw:-bw],
        cl_varfamily="temp",
        figcrs=crs,
        datcrs=crs,
        titles=titles,
    )
    fig.suptitle(f"{basetime.strftime('%Y-%m-%d:%H')}+{3*(ldt_idx+1)}H")
    fig.savefig(f"{modelid}_ldt{ldt_idx}.png")
    print(f"Saved {modelid}_ldt{ldt_idx}.png")
