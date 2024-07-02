# -*- coding: utf-8 -*-
"""Neural-LAM: Graph-based neural weather prediction models for Limited Area Modeling

Module defining the forecaster class. It is used to make weather forecasts in inference mode.
"""

import os
import numpy as np


class Forecaster:
    """Abstract class for forecasters"""
    def __init__(self):
        self.shortname = self.__class__.__name__.lower()
    
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
