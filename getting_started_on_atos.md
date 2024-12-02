# Getting started with Neural-LAM on Reaserve

This document summarizes the procedure to install the [Neural-LAM](https://github.com/mllam/neural-lam) code and use it on the ECMWF supercomputer (Atos).
The main additions to the procedure written in the original repo are the use of [virtual environments](https://docs.python.org/3/library/venv.html). and the extra dependencies induced by the use of the [MERA explorer](https://github.com/ThomasRieutord/mera-explorer) code.
It is assumed that the following commands are run on a Linux machine without root priviledges but with access to the [GPU partition](https://confluence.ecmwf.int/display/UDOC/HPC2020%3A+GPU+usage+for+AI+and+Machine+Learning)

  * Last update: 28 Nov 2024 (Thomas Rieutord)

## 1. Installation

These instructions start from scratch. Please skip any step you already have done.

### 1.1 Download the code bases

In the current usage at Met Éireann, Neural-LAM relies on a few code bases that must be downloaded from Github.
The current download uses HTTPS protocol, which is useful for reading the code only. If you plan to push any modification on the code bases, you should prefer SSH protocol.
```
cd ~
git clone https://github.com/ThomasRieutord/mera-explorer.git
git clone https://github.com/ThomasRieutord/metplotlib.git
git clone https://github.com/ThomasRieutord/neural-lam.git
```

Then, make sure you are on the correct branch of the Neural-LAM repository. This doc refers to the branch `met-eireann`:
```
cd ~/neural-lam
git checkout atos
```
Make sure the current file is present in your local directory before you continue.


### 1.2 Set up your virtual environments

We recommend using isolate Python environment to prevent any disruption with exisitng installation.
The chosen solution is to use [virtual environments](https://docs.python.org/3/library/venv.html).
The installation procedure starts from scratch, please skip any step you have already done.

As virtual environment can be reach relatively large sizes (with deep learning library, ~5GB per venv), we recommend to store them in a large enough file system.
Typically, $HOME directories may be too small.
We store the virtual environments in the variable `VENVROOT`, here set to `$HPCPERM/venvs`.
Additionally, the `pip` cache directory is by default `$TMPDIR`, which is only 3GB so it might raise a quota error.
Therefore we define the variable `PIPCACHEXL`, here set to `SCRATCH/cache/pip`, that will be used to move the `pip` cache to a larger filesystem.
Please adjust the paths in the `venv-utils.sh` file and append it to your `.bashrc`:

```
echo "# Added: $(date) from $PWD/venv-utils.sh" >> ~/.bashrc
cat venv-utils.sh >> ~/.bashrc
source ~/.bashrc
```

### 1.3 Create a new environment and update pip

After sourcing the .bashrc or restarting your shell, you will be able to create a new environment with the following command.
```
module load python3
venv-create neurallam python3.11 --upgrade-pip
```

### 1.4 Install the packages in your environment


```
venv-activate neurallam

TMPDIR=$PIPCACHEXL
pip install -e "neural-lam/.[dev,meteireann]" --cache-dir $PIPCACHEXL
pip install -e mera-explorer/. --cache-dir $PIPCACHEXL
pip install -e metplotlib/. --cache-dir $PIPCACHEXL
```

**NB:** to get out of your environment and retrieve the default Python installationm the command is `deactivate`.
Make sure to activate your environment prior to use Neural-LAM.

### 1.5 Set up links for inputs and outputs

For the **bulk inputs**, The MERA are stored in different location depending on the machine you use. This document gives the ones for Reaserve.
Edit the file `~/mera-explorer/local/paths.txt` and put the following values:
```
MERAROOTDIR = "/ec/res4/scratch/dutr" # Parent directory of all MERA GRIB files
MERACLIMDIR = "/perm/dutr/mera" # Directory where are stored climatology data (in particular the m05.grib)
```

Make sure the path is correct: the following command must return the same path as in the file.
```
python -c "from mera_explorer import MERAROOTDIR;print(MERAROOTDIR)"
```

For the **prepared datasets**, create a directory called `neurallam-datasets` in your data partition and link it to `data` in the Neural-LAM repository.
```
mkdir $SCRATCH/neurallam-datasets
cd ~/neural-lam # Make sure you are back to the directory containing the pyproject.toml
ln -s $SCRATCH/neurallam-datasets data
```

For the **outputs**, the directory will be created under the directory indicated by the `$SCRATCH` variable, usually equal to `/data/<username>`.
Models weights will be stored in the `saved_models` directory. Inference outputs will be stored in the `$SCRATCH/neurallam-inference-outputs` directory.

Last, be aware the index files created when reading GRIB will be written in the `~/tmp` directory, in case it causes any problem.

### 1.6 Check the installation

You can check the installation by running the testing program `import_tests.py` (within your environment):
```
python tests/import_tests.py
```
The result should look like this:
```
(neurallam)
```

## 2. Use cases

You are now ready to use Neural-LAM on the Atos. Some scripts already exist for the current use cases.

### 2.1 Create datasets

Edit and run the script `~/neural-lam/sbatch/1_create_mera_dataset.sh`:
```
venv-activate neurallam
cd ~/neural-lam/sbatch
sbatch 1_create_mera_dataset.sh
```


### 2.2 Train a model

If you are training a model on a given dataset for the first time, edit and run
```
sbatch ~/neural-lam/sbatch/2_prep_train_model.sh
```
If you already did a training on the same dataset, you can skip it.
Then, edit and run 
```
sbatch ~/neural-lam/sbatch/3_train_model.sh
```


### 2.3 Make inference

As this part is specific to MERA at the moment, the scripts for inference are in the mera-explorer code base.
Inference can only be made with a pre-trained model, which is identified by its run name.
In this tutorial, the run name we use is `graph_lam-4x64-09_03_18-2112` but it must be changed to one of the run names listed in the `~/neural-lam/saved_models` directory.
If you are running inference for the first time, you must create the initial and boundary conditions from MERA. Otherwise you can skip this line.
```
python ~/mera-explorer/scripts/write_gribs_for_neurallam_init.py --sdate 2017-01-01 --edate 2017-12-29
```

To have a qualitative evaluation of the forecast on the storm Ophelia (Oct 2017), use
```
python ~/mera-explorer/scripts/ophelia_forecast_plot.py --forecaster neurallam:graph_lam-4x64-09_03_18-2112 --figdir ophelia-figures
```

To write inference output in GRIB later used in HARP, use
```
python ~/mera-explorer/scripts/make_inference_forecast.py --sdate 2017-01-01 --edate 2017-03-15 --max-leadtime 65h --forecaster neurallam:graph_lam-4x64-09_03_18-2112
```
