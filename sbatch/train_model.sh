#!/bin/bash
# The job name
#SBATCH --job-name=pyng
# Set the error and output files
#SBATCH --output=pyng-%J.out
#SBATCH --error=pyng-%J.out
# Set the initial working directory
#SBATCH --chdir=/scratch/dutr/spool
# Choose the queue
#SBATCH --qos=ng
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
# Wall clock time limit
#SBATCH --time=24:00:00
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

python $HOME/neural-lam/scripts/train_model.py --dataset mera_dataset_10years --batch_size 2 --step_length 1 --ar_steps 2 --epochs 3 --n_workers 8

date
