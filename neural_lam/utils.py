# Standard library
import os
import shutil

# Third-party
import numpy as np
import torch
from torch import nn
from tueplots import bundles, figsizes

# First-party
from neural_lam import constants, PACKAGE_ROOTDIR


def load_dataset_stats(dataset_name, device="cpu"):
    """
    Load arrays with stored dataset statistics from pre-processing
    """
    static_dir_path = os.path.join(
        PACKAGE_ROOTDIR, "data", dataset_name, "static"
    )

    def loads_file(fn):
        return torch.load(
            os.path.join(static_dir_path, fn), map_location=device
        )

    data_mean = loads_file("parameter_mean.pt")  # (d_features,)
    data_std = loads_file("parameter_std.pt")  # (d_features,)

    flux_stats = loads_file("flux_stats.pt")  # (2,)
    flux_mean, flux_std = flux_stats

    return {
        "data_mean": data_mean,
        "data_std": data_std,
        "flux_mean": flux_mean,
        "flux_std": flux_std,
    }


def load_static_data(dataset_name, device="cpu"):
    """
    Load static files related to dataset
    """
    static_dir_path = os.path.join(
        PACKAGE_ROOTDIR, "data", dataset_name, "static"
    )

    def loads_file(fn):
        return torch.load(
            os.path.join(static_dir_path, fn), map_location=device
        )

    # Load border mask, 1. if node is part of border, else 0.
    border_mask_np = np.load(os.path.join(static_dir_path, "border_mask.npy"))
    border_mask = (
        torch.tensor(border_mask_np, dtype=torch.float32, device=device)
        .flatten(0, 1)
        .unsqueeze(1)
    )  # (N_grid, 1)

    grid_static_features = loads_file(
        "grid_features.pt"
    )  # (N_grid, d_grid_static)

    # Load step diff stats
    step_diff_mean = loads_file("diff_mean.pt")  # (d_f,)
    step_diff_std = loads_file("diff_std.pt")  # (d_f,)

    # Load parameter std for computing validation errors in original data scale
    data_mean = loads_file("parameter_mean.pt")  # (d_features,)
    data_std = loads_file("parameter_std.pt")  # (d_features,)

    # Load loss weighting vectors
    param_weights = torch.tensor(
        np.load(os.path.join(static_dir_path, "parameter_weights.npy")),
        dtype=torch.float32,
        device=device,
    )  # (d_f,)

    return {
        "border_mask": border_mask,
        "grid_static_features": grid_static_features,
        "step_diff_mean": step_diff_mean,
        "step_diff_std": step_diff_std,
        "data_mean": data_mean,
        "data_std": data_std,
        "param_weights": param_weights,
    }


class BufferList(nn.Module):
    """
    A list of torch buffer tensors that sit together as a Module with no
    parameters and only buffers.

    This should be replaced by a native torch BufferList once implemented.
    See: https://github.com/pytorch/pytorch/issues/37386
    """

    def __init__(self, buffer_tensors, persistent=True):
        super().__init__()
        self.n_buffers = len(buffer_tensors)
        for buffer_i, tensor in enumerate(buffer_tensors):
            self.register_buffer(f"b{buffer_i}", tensor, persistent=persistent)

    def __getitem__(self, key):
        return getattr(self, f"b{key}")

    def __len__(self):
        return self.n_buffers

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def load_graph(graph_name, device="cpu"):
    """
    Load all tensors representing the graph
    """
    # Define helper lambda function
    graph_dir_path = os.path.join(PACKAGE_ROOTDIR, "graphs", graph_name)

    def loads_file(fn):
        return torch.load(os.path.join(graph_dir_path, fn), map_location=device)

    # Load edges (edge_index)
    m2m_edge_index = BufferList(
        loads_file("m2m_edge_index.pt"), persistent=False
    )  # List of (2, M_m2m[l])
    g2m_edge_index = loads_file("g2m_edge_index.pt")  # (2, M_g2m)
    m2g_edge_index = loads_file("m2g_edge_index.pt")  # (2, M_m2g)

    n_levels = len(m2m_edge_index)
    hierarchical = n_levels > 1  # Nor just single level mesh graph

    # Load static edge features
    m2m_features = loads_file("m2m_features.pt")  # List of (M_m2m[l], d_edge_f)
    g2m_features = loads_file("g2m_features.pt")  # (M_g2m, d_edge_f)
    m2g_features = loads_file("m2g_features.pt")  # (M_m2g, d_edge_f)

    # Normalize by dividing with longest edge (found in m2m)
    longest_edge = max(
        torch.max(level_features[:, 0]) for level_features in m2m_features
    )  # Col. 0 is length
    m2m_features = BufferList(
        [level_features / longest_edge for level_features in m2m_features],
        persistent=False,
    )
    g2m_features = g2m_features / longest_edge
    m2g_features = m2g_features / longest_edge

    # Load static node features
    mesh_static_features = loads_file(
        "mesh_features.pt"
    )  # List of (N_mesh[l], d_mesh_static)

    # Some checks for consistency
    assert (
        len(m2m_features) == n_levels
    ), "Inconsistent number of levels in mesh"
    assert (
        len(mesh_static_features) == n_levels
    ), "Inconsistent number of levels in mesh"

    if hierarchical:
        # Load up and down edges and features
        mesh_up_edge_index = BufferList(
            loads_file("mesh_up_edge_index.pt"), persistent=False
        )  # List of (2, M_up[l])
        mesh_down_edge_index = BufferList(
            loads_file("mesh_down_edge_index.pt"), persistent=False
        )  # List of (2, M_down[l])

        mesh_up_features = loads_file(
            "mesh_up_features.pt"
        )  # List of (M_up[l], d_edge_f)
        mesh_down_features = loads_file(
            "mesh_down_features.pt"
        )  # List of (M_down[l], d_edge_f)

        # Rescale
        mesh_up_features = BufferList(
            [
                edge_features / longest_edge
                for edge_features in mesh_up_features
            ],
            persistent=False,
        )
        mesh_down_features = BufferList(
            [
                edge_features / longest_edge
                for edge_features in mesh_down_features
            ],
            persistent=False,
        )

        mesh_static_features = BufferList(
            mesh_static_features, persistent=False
        )
    else:
        # Extract single mesh level
        m2m_edge_index = m2m_edge_index[0]
        m2m_features = m2m_features[0]
        mesh_static_features = mesh_static_features[0]

        (
            mesh_up_edge_index,
            mesh_down_edge_index,
            mesh_up_features,
            mesh_down_features,
        ) = ([], [], [], [])

    return hierarchical, {
        "g2m_edge_index": g2m_edge_index,
        "m2g_edge_index": m2g_edge_index,
        "m2m_edge_index": m2m_edge_index,
        "mesh_up_edge_index": mesh_up_edge_index,
        "mesh_down_edge_index": mesh_down_edge_index,
        "g2m_features": g2m_features,
        "m2g_features": m2g_features,
        "m2m_features": m2m_features,
        "mesh_up_features": mesh_up_features,
        "mesh_down_features": mesh_down_features,
        "mesh_static_features": mesh_static_features,
    }


def make_mlp(blueprint, layer_norm=True):
    """
    Create MLP from list blueprint, with
    input dimensionality: blueprint[0]
    output dimensionality: blueprint[-1] and
    hidden layers of dimensions: blueprint[1], ..., blueprint[-2]

    if layer_norm is True, includes a LayerNorm layer at
    the output (as used in GraphCast)
    """
    hidden_layers = len(blueprint) - 2
    assert hidden_layers >= 0, "Invalid MLP blueprint"

    layers = []
    for layer_i, (dim1, dim2) in enumerate(zip(blueprint[:-1], blueprint[1:])):
        layers.append(nn.Linear(dim1, dim2))
        if layer_i != hidden_layers:
            layers.append(nn.SiLU())  # Swish activation

    # Optionally add layer norm to output
    if layer_norm:
        layers.append(nn.LayerNorm(blueprint[-1]))

    return nn.Sequential(*layers)


def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of
    the page width.
    """
    bundle = bundles.neurips2023(usetex=True, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] / fraction,
        original_figsize[1],
    )
    return bundle


def init_wandb_metrics(wandb_logger):
    """
    Set up wandb metrics to track
    """
    experiment = wandb_logger.experiment
    experiment.define_metric("val_mean_loss", summary="min")
    for step in constants.VAL_STEP_LOG_ERRORS:
        experiment.define_metric(f"val_loss_unroll{step}", summary="min")


def checkpointpath_to_runname(checkpoint_path):
    """Return the run name from a checkpoint path


    Parameters
    ----------
    checkpoint_path: str
        Path to the corresponding checkpoint


    Returns
    -------
    run_name: str
        Run name or direct path to the checkpoint


    Examples
    --------
    >>> # With experiment name
    >>> checkpointpath_to_runname("saved_models/graph_lam-4x64-07_19_15-2217/min_val_loss.ckpt")
    'graph_lam-4x64-07_19_15-2217'
    """
    return os.path.basename(os.path.dirname(checkpoint_path))


def runname_to_checkpointpath(run_name):
    """Return the absolute path to the checkpoint from a run name


    Parameters
    ----------
    run_name: str
        Run name or direct path to the checkpoint


    Returns
    -------
    checkpoint_path: str
        Absolute path to the corresponding checkpoint


    Examples
    --------
    >>> # With experiment name
    >>> runname_to_checkpointpath("graph_lam-4x64-07_19_15-2217")
    '/.../neural-lam/saved_models/graph_lam-4x64-07_19_15-2217/min_val_loss.ckpt'
    
    >>> # With direct path
    >>> runname_to_checkpointpath("saved_models/graph_lam-4x64-07_19_15-2217/min_val_loss.ckpt")
    '/.../neural-lam/saved_models/graph_lam-4x64-07_19_15-2217/min_val_loss.ckpt'
    """
    candidates = [
        os.path.abspath(run_name),
        os.path.join(PACKAGE_ROOTDIR, "saved_models", run_name, "min_val_loss.ckpt")
    ]

    checkpoint_path = None
    for candidate in candidates:
        if os.path.isfile(candidate) and candidate.endswith(".ckpt"):
            checkpoint_path = candidate
            break

    if checkpoint_path is None:
        raise ValueError(f"Unable to find checkpoint for run_name={run_name}")

    return checkpoint_path


def get_stale_runnames(min_epochs=2, max_loss=1000):
    """List stale run names.

    Runs are considered as stale if there is no checkpoint or no log stored
    or if they do not reach a minimum number of epochs or a loss threshold.


    Parameters
    ----------
    min_epochs: int
        Epoch threshold (runs below this threshold are seen as stale)
    
    max_loss: float
        Loss threshold (runs above this threshold are seen as stale)


    Returns
    -------
    runnames_stale: list of str
        List of run names considered as stale
    """
    logdir = os.path.join(PACKAGE_ROOTDIR, "logs")
    ckpdir = os.path.join(PACKAGE_ROOTDIR, "saved_models")
    runnames_logs = os.listdir(logdir)
    runnames_ckpt = os.listdir(ckpdir)

    something_missing = set(runnames_logs + runnames_ckpt) - (set(runnames_logs) & set(runnames_ckpt))

    not_good_enough = []
    for run_name in runnames_ckpt:
        ckptpath = runname_to_checkpointpath(run_name)
        ckpt = torch.load(ckptpath, map_location="cpu")
        cbd = next(iter(ckpt['callbacks'].values()))
        if ckpt["epoch"] < min_epochs or cbd["current_score"].item() > max_loss:
            not_good_enough.append(run_name)

    return list(something_missing | set(not_good_enough))


def remove_stale_runnames(runnames_stale, runnames_directory):
    """Remove the runs listed as stale from the given directory.


    Parameters
    ----------
    runnames_stale: list of str
        List of run names considered as stale

    runnames_directory: str
        Path to the directory to clean
    """
    runnames = os.listdir(runnames_directory)
    for run_name in runnames:
        to_remove = os.path.join(runnames_directory, run_name)
        if run_name in runnames_stale and os.path.isdir(to_remove):
            shutil.rmtree(to_remove)
            print(f"Removed: {to_remove}")


def cleanup_experiments(min_epochs=2, max_loss=1000, remove_ckpt=False, remove_logs=False):
    """Identify and remove stale runs. Use with caution.


    Parameters
    ----------
    min_epochs: int
        Epoch threshold (runs below this threshold are seen as stale)

    max_loss: float
        Loss threshold (runs above this threshold are seen as stale)

    remove_ckpt: bool
        If True, remove the checkpoints of the stale runs

    remove_logs: bool
        If True, remove the log files of the stale runs
    """
    logdir = os.path.join(PACKAGE_ROOTDIR, "logs")
    ckpdir = os.path.join(PACKAGE_ROOTDIR, "saved_models")

    runnames_stale = get_stale_runnames(min_epochs=min_epochs, max_loss=max_loss)
    print(f"{len(runnames_stale)} stale runnames found: {runnames_stale}")

    if remove_ckpt:
        remove_stale_runnames(runnames_stale, ckpdir)

    if remove_logs:
        remove_stale_runnames(runnames_stale, logdir)

def experiment_summary(run_names=None):
    """Print some key variables in the checkpoints of run names.
    
    
    Parameters
    ----------
    run_names: list of str
        List of run names. If not provided, takes all the checkpoints available
    """
    if run_names is None:
        ckpdir = os.path.join(PACKAGE_ROOTDIR, "saved_models")
        run_names = os.listdir(ckpdir)

    for run_name in run_names:
        ckptpath = runname_to_checkpointpath(run_name)
        ckpt = torch.load(ckptpath, map_location="cpu")
        cbd = next(iter(ckpt['callbacks'].values()))
        hp = vars(next(iter(ckpt['hyper_parameters'].values())))
        msg = f"""
-----------------------------------
    {run_name}  
-----------------------------------
epoch={ckpt["epoch"]}
global_step={ckpt["global_step"]}
current_score={cbd["current_score"].item()}
HYPER-PARAMETERS:
""" + "\n".join(
    [
        f"  {k}={v}" for k,v in hp.items()
    ]
)
        print(msg)

# End of file
