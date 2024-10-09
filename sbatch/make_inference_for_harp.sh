#!/bin/bash
# The job name
#SBATCH --job-name=makast
# Set the error and output files
#SBATCH --output=makast-%J.out
#SBATCH --error=makast-%J.out
# Set the initial working directory
#SBATCH --chdir=/scratch/dume/spool
# Choose the queue
#SBATCH --qos=nf
#SBATCH --mem=64G
# Wall clock time limit
#SBATCH --time=30:05:00
# Send an email on failure
#SBATCH --mail-type=FAIL
# This is the job
date
echo "Running on $HOSTNAME:$PWD"

module load conda
mamba activate mllam_v2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/perm/dume/conda/envs/mllam_v2/lib
echo "Env successfully loaded!"
python --version
date

SDATE=2017-01-01
EDATE=2017-04-01
MAXLDT=65h

python $HOME/mera-explorer/scripts/write_gribs_for_neurallam_init.py --sdate $SDATE --edate $EDATE --max-leadtime $MAXLDT 

python $HOME/mera-explorer/scripts/make_fake_forecast.py --sdate $SDATE --edate $EDATE --max-leadtime $MAXLDT --forecaster neurallam:graph_lam-4x64-08_12_08-9132

date
