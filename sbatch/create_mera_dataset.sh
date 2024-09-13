#!/bin/bash
# The job name
#SBATCH --job-name=creset
# Set the error and output files
#SBATCH --output=creset-%J.out
#SBATCH --error=creset-%J.out
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

DATASETNAME=test_links
SDATE=1991-01-02
EDATE=1991-03-02
SUBSAMPLE=1

python $HOME/mera-explorer/scripts/create_static_features.py --outdir $SCRATCH/neurallam-datasets/$DATASETNAME --subsample $SUBSAMPLE --writefiles

date

python $HOME/mera-explorer/scripts/create_mera_sample.py --outdir $SCRATCH/neurallam-datasets/$DATASETNAME --subsample $SUBSAMPLE --sdate $SDATE --edate $EDATE --writefiles

date
