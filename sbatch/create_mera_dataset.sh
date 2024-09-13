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
#SBATCH --time=48:00:00
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

DATASETNAME=mera_37years_24h
SDATE=1981-01-02
EDATE=2017-12-30
SUBSAMPLE=1
TEXTRACT=24h

python $HOME/mera-explorer/scripts/create_static_features.py --outdir $SCRATCH/neurallam-datasets/$DATASETNAME --subsample $SUBSAMPLE --writefiles

date

python $HOME/mera-explorer/scripts/create_mera_sample.py --outdir $SCRATCH/neurallam-datasets/$DATASETNAME --subsample $SUBSAMPLE --sdate $SDATE --edate $EDATE --textract $TEXTRACT --writefiles

date
