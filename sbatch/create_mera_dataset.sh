#!/bin/bash
# The job name
#SBATCH --job-name=pynf
# Set the error and output files
#SBATCH --output=pynf-%J.out
#SBATCH --error=pynf-%J.out
# Set the initial working directory
#SBATCH --chdir=/scratch/dutr/spool
# Choose the queue
#SBATCH --qos=nf
#SBATCH --mem=32G
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

DATASETNAME=mera_4years
SDATE=1981-01-01
EDATE=1985-01-01
SUBSAMPLE=1

python $HOME/mera-explorer/scripts/create_static_features.py --indirclim $PERM/mera --outdirmllam $SCRATCH/neurallam/$DATASETNAME --subsample $SUBSAMPLE --writefiles

date

python $HOME/mera-explorer/scripts/create_mera_sample.py --indirclim $PERM/mera --indirgrib $SCRATCH --outdir $SCRATCH/neurallam/$DATASETNAME/samples --subsample $SUBSAMPLE --sdate $SDATE --edate $EDATE --writefiles

date
