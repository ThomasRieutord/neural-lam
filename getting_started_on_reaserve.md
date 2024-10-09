# Installation of Neural-LAM on Reaserve

This document summarizes the procedure to install the [Neural-LAM](https://github.com/mllam/neural-lam) code on the Met Éireann research server (Reaserve).
The main additions to the procedure written in the original repo are the use of [Mamba](https://mamba.readthedocs.io/en/latest/) environments and the extra dependencies induced by the use of the [MERA explorer](https://github.com/ThomasRieutord/mera-explorer) code.
It is assumed that the following commands are run on a Linux machine without root priviledges.
While the step 1 can be skipped if you already have a clean Mamba environment with Python 3.9, the order of the following install must be respected (Mamba first, then pip)

  * Last update: 10 Sep 2024 (Thomas Rieutord)

## 1. Download the code bases

In the current usage at Met Éireann, Neural-LAM relies on a few code bases that must be downloaded from Github.
The current download uses HTTPS protocol, which is useful for reading the code only. If you plan to push any modification on the code bases, you should prefer SSH protocol.
```
git clone https://github.com/ThomasRieutord/mera-explorer.git
git clone https://github.com/ThomasRieutord/metplotlib.git
git clone https://github.com/ThomasRieutord/neural-lam.git
```

Then, make sure you are on the correct branch of the Neural-LAM repository. This doc refers to the branch `dev-tr`:
```
cd neural-lam
git checkout dev-tr
```
Make sure the current file is present in your local directory before you continue.

## 2. Creating a Mamba environment

If Mamba is not already installed, you can get it with the following commands:
```
# OPTIONAL: don't you have it installed already?
wget https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Mambaforge-23.11.0-0-Linux-x86_64.sh
bash Mambaforge-23.11.0-0-Linux-x86_64.sh
```

Once installed, create a new environment with Python 3.9:
```
mamba create -n neurallam python=3.9
mamba activate neurallam
```

## 3. Mamba installations

```
mamba install --file neural-lam/requirements.txt
mamba install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install eccodes cfgrib xarray h5py easydict paramiko netCDF4 h5py
```

## 4. Pip installations

Make sure to run all installations with Mamba prior to the ones with pip.

```
pip install epygram climetlab tueplots mlflow
pip install pyg-lib==0.2.0 torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.1 torch-geometric==2.3.1 -f https://pytorch-geometric.com/whl/torch-2.0.1+cu118.html
pip install -e .
cd ../mera-explorer #<mera-explorer directory containing the pyproject.toml>
pip install -e .
cd ../metplotlib #<metplotlib directory containing the setup.py>
pip install -e .
```

## 5. Set up links for inputs and outputs

For the **bulk inputs**, The MERA are stored in different location depending on the machine you use. This document gives the ones for Reaserve.
Edit the file `../mera-explorer/local/paths.txt` and put the following values:
```
MERAROOTDIR = "/data/trieutord/MERA/grib-all" # Parent directory of all MERA GRIB files
MERACLIMDIR = "/data/trieutord/MERA/meraclim" # Directory where are stored climatology data (in particular the m05.grib)
```

Make sure the path is correct: the following command must return the same path as in the file.
```
python -c "from mera_explorer import MERAROOTDIR;print(MERAROOTDIR)"
```

For the **prepared datasets**, create a directory called `neurallam-datasets` in your data partition and link it to `data` in the Neural-LAM repository.
```
mkdir /data/<username>/neurallam-datasets
cd ../neural-lam # Make sure you are back to the directory containing the pyproject.toml
ln -s /data/<username>/neurallam-datasets data
```

For the **outputs**, if you don't already have one, create a `$SCRATCH` variable with a path to your data partition.
```
export SCRATCH=/data/<username>/scratch
```
Models weights will be stored in the `saved_models` directory. Inference outputs will be stored in the `$SCRATCH/neurallam-inference-outputs` directory.

## 6. Use cases

You are now ready to use Neural-LAM on Reaserve. Some scripts already exist for the current use cases.

### Create datasets

Edit and run the script `sbatch/create_mera_dataset.sh`

### Train a model

If you are training a model on a given dataset for the first time, edit and run `sbatch/prep_train_model.sh`.
If you already did a training on the same dataset, you can skip it.
Then, edit and run `sbatch/train_model.sh`.

### Make inference

As this part is specific to MERA at the moment, the scripts for inference are in the mera-explorer code base.
If you are running inference for the first time, you must create the initial and boundary conditions from MERA. Otherwise you can skip this line.
```
python ../mera-explorer/scripts/write_gribs_for_neurallam_init.py --sdate 2017-01-01 --edate 2017-12-29
```

To have a qualitative evalution of the forecast on the storm Ophelia (Oct 2017), use
```
python ../mera-explorer/scripts/ophelia_forecast_plot.py --forecaster neurallam:graph_lam-4x64-09_03_18-2112 --figdir ophelia-figures
```

To write inference output in GRIB later used in HARP, use
```
python ../mera-explorer/scripts/make_fake_forecast.py --sdate 2017-01-01 --edate 2017-03-15 --max-leadtime 65h --forecaster neurallam:graph_lam-4x64-09_03_18-2112
```
