# -*- coding: utf-8 -*-
"""Neural-LAM: Graph-based neural weather prediction models for Limited Area Modeling

Module defining the forecaster class. It is used to make weather forecasts in inference mode.
"""

import os
import numpy as np
import torch
from neural_lam import package_rootdir
from neural_lam import scalers
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.hi_lam import HiLAM
from neural_lam.models.hi_lam_parallel import HiLAMParallel

MODEL_DIRECTORY = os.path.join(package_rootdir, "saved_models")
MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}

class Forecaster:
    """Abstract class for forecasters"""
    def __init__(self):
        self.shortname = self.__class__.__name__.lower()
        
        # Scalers for normalization
        self.flux_scaler = scalers.IdentityScaler()
        self.data_scaler = scalers.IdentityScaler()
    
    
    def forecast(self, analysis, forcings, borders):
        return NotImplemented


class Persistence(Forecaster):
    """Return the same weather as the analysis"""
    def forecast(self, analysis, forcings, borders):
        return np.broadcast_to(analysis[1], borders.shape)


class GradientIncrement(Forecaster):
    """Takes the previous state plus an increment based on the gradient"""
    incrementcoeff = 0.1
    
    def forecast(self, analysis, forcings, borders):
        nt, n_grid, nv = borders.shape
        forecast = np.zeros_like(borders)
        
        forecast[0] = analysis[1] + self.incrementcoeff*(analysis[1] - analysis[0])
        forecast[1] = forecast[0] + self.incrementcoeff*(forecast[0] - analysis[1])
        
        for t in range(1, nt-1):
            forecast[t+1] = forecast[t] + self.incrementcoeff*(forecast[t] - forecast[t-1])
        
        return forecast

class NeuralLAMforecaster(Forecaster):
    """Make prediction from a pre-trained Neural-LAM model"""
    def __init__(self, ckptpath, device ="cpu"):
        ckpt = torch.load(ckptpath, map_location=device)
        saved_args = ckpt["hyper_parameters"]["args"]
        epoch = ckpt["epoch"]
        model_class = MODELS[saved_args.model]
        modelid = os.path.basename(os.path.dirname(ckptpath))
        modelid = ''.join(c for c in modelid if c.isalnum())
        
        self.model = model_class.load_from_checkpoint(ckptpath, args=saved_args)
        self.shortname = f"{modelid}_{epoch}e{saved_args.batch_size}b"
        self.flux_scaler = scalers.FluxScaler(saved_args.dataset)
        self.data_scaler = scalers.DataScaler(saved_args.dataset)
    
    def forecast(self, analysis, forcings, borders):
        analysis, forcings, borders = [torch.tensor(_) for _ in (analysis, forcings, borders)]
        analysis, forcings, borders = [_.unsqueeze(0).float() for _ in (analysis, forcings, borders)]
        
        with torch.no_grad():
            forecast, _ = self.model.unroll_prediction(analysis, forcings, borders)
        
        return forecast.squeeze().detach().numpy()
