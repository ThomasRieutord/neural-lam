#!/bin/bash
# The job name
#SBATCH --job-name=prelam
# Set the error and output files
#SBATCH --output=prelam-%J.out
#SBATCH --error=prelam-%J.out
# Set the initial working directory
#SBATCH --chdir=/scratch/dutr/spool
# Choose the queue
#SBATCH --qos=nf
#SBATCH --mem=64GB
# Wall clock time limit
#SBATCH --time=24:00:00
# Send an email on failure
#SBATCH --mail-type=FAIL
# This is the job
date
echo "Running on $HOSTNAME:$PWD"

# On reaserve, environment must be loaded before executing the code
#module load conda
#mamba activate neurallam

echo "Env successfully loaded!"
python --version
date

DATASET=mera_8years_fullres
GRAPH=hierarchical
HIERARCHICAL=1
LEVELS=4
BATCHSIZE=8

python $HOME/neural-lam/scripts/create_mesh.py --dataset $DATASET --hierarchical $HIERARCHICAL --graph $GRAPH --levels $LEVELS
python $HOME/neural-lam/scripts/create_grid_features.py --dataset $DATASET
python $HOME/neural-lam/scripts/create_parameter_weights.py --dataset $DATASET --batch_size $BATCHSIZE --step_length 1

date
