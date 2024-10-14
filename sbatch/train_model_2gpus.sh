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
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128GB
# Wall clock time limit
#SBATCH --time=4:00:00
# Send an email on failure
#SBATCH --mail-type=FAIL
# This is the job
date
echo "Running on $HOSTNAME:$PWD"

# On reaserve, environment must be loaded before executing the code
module load conda
mamba activate neurallam

echo "Env successfully loaded!"
python --version
date

nvidia-smi

DATASET=mera_4years_fullres
GRAPH=hierarchical
MODEL=hi_lam
BATCHSIZE=2
EPOCHS=50
N_WORKERS=8
AR_STEPS=1

set -vx

python $HOME/neural-lam/scripts/train_model.py \
--dataset $DATASET \
--graph $GRAPH \
--model $MODEL \
--batch_size $BATCHSIZE \
--ar_steps $AR_STEPS \
--step_length 1 \
--control_only 1 \
--epochs $EPOCHS \
--n_workers $N_WORKERS \
--track_emissions True \
--seed 193 \
--gpus 4

date
