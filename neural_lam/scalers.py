# -*- coding: utf-8 -*-
"""Neural-LAM: Graph-based neural weather prediction models for Limited Area Modeling

Module defining scalers object, used for the normalization of the data.
Inspired from the [Scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler).
"""
import os
import torch
from neural_lam import package_rootdir


class _Scaler:
    """Abstract class for Scalers.
    Child classes will differ only from the list of files containing the stats and the way to load them.
    """
    stats_files = ["diff_mean.pt", "diff_std.pt", "flux_stats.pt", "parameter_mean.pt", "parameter_std.pt"]
    
    def __init__(self, dataset_name, device = "cpu"):
        static_dir_path = os.path.join(package_rootdir, "data", dataset_name, "static")
        for filename in self.stats_files:
            assert os.path.isfile(os.path.join(static_dir_path, filename)), f"File {filename} is missing for dataset {dataset_name}. Have you run create_parameter_weights.py?"

        self.static_dir_path = static_dir_path
        self.device = device
        self._load_stats()
    
    def _load_stats(self):
        raise NotImplementedError(f"The class {self.__class__.__name__} is abstract. Please use one of the child classes")

    def _with_numpy(func):
        """Decorator to convert input array to tensor and convert the output array back to numpy"""
        def wrapper(self, array):
            array = torch.tensor(array)
            result = func(self, array)
            return result.detach().numpy()
        return wrapper

    @_with_numpy
    def transform(self, data):
        return (data - self.mean)/self.std
    
    @_with_numpy
    def inverse_transform(self, data):
        return data * self.std + self.mean


class IdentityScaler(_Scaler):
    """Does not transform the data"""
    stats_files = []
    
    def __init__(self):
        self._load_stats()
    
    def _load_stats(self):
        self.mean = torch.tensor(0)
        self.std = torch.tensor(1)


class FluxScaler(_Scaler):
    """Flux data scaler"""
    stats_files = ["flux_stats.pt"]
    
    def _load_stats(self):
        flux_stats = torch.load(os.path.join(self.static_dir_path, "flux_stats.pt"), map_location = self.device) # (2,)
        self.mean = flux_stats[0]
        self.std = flux_stats[1]


class DataScaler(_Scaler):
    """Instantaneous data scaler"""
    stats_files = ["parameter_mean.pt", "parameter_std.pt"]
    
    def _load_stats(self):
        self.mean = torch.load(os.path.join(self.static_dir_path, "parameter_mean.pt"), map_location = self.device) # (d_features,)
        self.std = torch.load(os.path.join(self.static_dir_path, "parameter_std.pt"), map_location = self.device) # (d_features,)


if __name__ == "__main__":
    dataset_name = "meps_example"
    print(f"Testing scalers on dataset {dataset_name}")
    x = torch.rand(4, 256, 17)
    for sc in [IdentityScaler(), FluxScaler(dataset_name), DataScaler(dataset_name)]:
        y = sc.inverse_transform(x)
        x0 = sc.transform(y)
        assert (x - x0).sum() < 1e-5, f"Gap with original data exceeds threshold for {sc}"
    
    print("All scalers tested tranform and inverse_transform successfully")
