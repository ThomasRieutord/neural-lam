#!/bin/bash
# The job name
#SBATCH --job-name=creset
# Set the error and output files
#SBATCH --output=creset-%J.out
#SBATCH --error=creset-%J.out
# Set the initial working directory
#SBATCH --chdir=/scratch/dume/spool
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
mamba activate neural_lam
# On reaserve, environment must be loaded before executing the code
#module load conda
#mamba activate neurallam

echo "Env successfully loaded!"
python --version
date

DATASETNAME=mera_8years_fullres
SDATE=2000-01-02
EDATE=2008-01-02
SUBSAMPLE=1

python $HOME/mera-explorer/scripts/create_static_features.py --indirclim /perm/dutr/mera --outdirmllam $SCRATCH/neurallam-datasets/$DATASETNAME --subsample $SUBSAMPLE --writefiles

date

python $HOME/mera-explorer/scripts/create_mera_sample.py --indirclim /perm/dutr/mera --indirgrib /scratch/dutr/mera --outdir $SCRATCH/neurallam/$DATASETNAME/samples --subsample $SUBSAMPLE --sdate $SDATE --edate $EDATE --writefiles

date
