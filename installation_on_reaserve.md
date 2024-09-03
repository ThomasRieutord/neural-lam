# Installation of Neural-LAM on Reaserve

This document summarizes the procedure to install the [Neural-LAM](https://github.com/mllam/neural-lam) code on the Met Éireann research server (Reaserve).
The main additions to the procedure written in the original repo are the use of [Mamba](https://mamba.readthedocs.io/en/latest/) environments and the extra dependencies induced by the use of the [MERA explorer](https://github.com/ThomasRieutord/mera-explorer) code.
It is assumed that the following commands are run on a Linux machine without root priviledges.
While the step 1 can be skipped if you already have a clean Mamba environment with Python 3.9, the order of the following install must be respected (Mamba first, then pip)

  * Last update: 3 Sep 2024 (Thomas Rieutord)


## 1. Creating a Mamba environment

If Mamba is not already installed, you can get it with the following commands:
```
wget https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Mambaforge-23.11.0-0-Linux-x86_64.sh
bash Mambaforge-23.11.0-0-Linux-x86_64.sh
```

Once installed, create a new environment with Python 3.9:
```
mamba create -n neurallam python=3.9
mamba activate neurallam
```

## 2. Mamba installations

```
mamba install --file neural-lam/requirements.txt
mamba install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install eccodes cfgrib xarray h5py easydict paramiko netCDF4 h5py
```

## 3. Pip installations

Make sure to run all installations with Mamba prior to the ones with pip.

```
pip install epygram climetlab tueplots mlflow
pip install pyg-lib==0.2.0 torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.1 torch-geometric==2.3.1 -f https://pytorch-geometric.com/whl/torch-2.0.1+cu118.html
cd <neural-lam directory containing the pyproject.toml>
pip install -e .
cd <mera-explorer directory containing the pyproject.toml>
pip install -e .
```
