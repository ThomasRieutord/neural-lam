#!/bin/bash
# The job name
#SBATCH --job-name=trainlam
# Set the error and output files
#SBATCH --output=trainlam-%J.out
#SBATCH --error=trainlam-%J.out
# Set the initial working directory
#SBATCH --chdir=/scratch/dutr/spool
# Choose the queue
#SBATCH --qos=ng
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96GB
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

DATASET=mera_4years_fullres
BATCHSIZE=2
EPOCHS=20
N_WORKERS=8

set -vx

python $HOME/neural-lam/scripts/train_model.py --dataset $DATASET --batch_size $BATCHSIZE --ar_steps 2 --step_length 1 --control_only 1 --epochs $EPOCHS --n_workers $N_WORKERS

date
