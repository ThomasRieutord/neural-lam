#!/bin/bash
# The job name
#SBATCH --job-name=pynf
# Set the error and output files
#SBATCH --output=pynf-%J.out
#SBATCH --error=pynf-%J.out
# Set the initial working directory
#SBATCH --chdir=/scratch/dume/spool
# Choose the queue
#SBATCH --qos=nf
#SBATCH --mem=32G
# Wall clock time limit
#SBATCH --time=01:05:00
# Send an email on failure
#SBATCH --mail-type=FAIL
# This is the job
date
echo "Running on $HOSTNAME:$PWD"

module load conda
mamba activate neural_lam_v2

echo "Env successfully loaded!"
python --version
date

python $HOME/neural-lam/scripts/create_parameter_weights.py --dataset mera_dataset_10years --batch_size 4 --step_length 3

date
