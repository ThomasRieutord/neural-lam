#!/bin/bash
# The job name
#SBATCH --job-name=makast
# Set the error and output files
#SBATCH --output=makast-%J.out
#SBATCH --error=makast-%J.out
# Set the initial working directory
#SBATCH --chdir=/scratch/dutr/spool
# Choose the queue
#SBATCH --qos=nf
#SBATCH --mem=128G
# Wall clock time limit
#SBATCH --time=30:05:00
# Send an email on failure
#SBATCH --mail-type=FAIL
# This is the job
date
echo "Running on $HOSTNAME:$PWD"

module load conda
mamba activate neurallam

echo "Env successfully loaded!"
python --version
date

SDATE=2017-01-01
EDATE=2017-02-01
MAXLDT=65h

python $HOME/mera-explorer/scripts/make_fake_forecast.py --sdate $SDATE --edate $EDATE --max-leadtime $MAXLDT --forecaster neurallam

date
