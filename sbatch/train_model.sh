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
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128GB
# Wall clock time limit
#SBATCH --time=48:00:00
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

# Check hardware allocation
echo " ====== HARDWARE ALLOCATION ====== "
free -h
lscpu
nvidia-smi
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_CPUS_PER_GPU=$SLURM_CPUS_PER_GPU"

# Hardware variables (must be equal to the SBATCH arguments)
N_WORKERS=16
N_GPUS=2

DATASET=mera_37years_24h
GRAPH=hierarchical
MODEL=hi_lam
BATCHSIZE=2
EPOCHS=60
AR_STEPS=1
STARTFROM=hi_lam-4x64-10_11_13-2725

set -vx

srun --cpus-per-gpu $N_WORKERS python $HOME/neural-lam/scripts/train_model.py \
--load $STARTFROM \
--dataset $DATASET \
--graph $GRAPH \
--model $MODEL \
--batch_size $BATCHSIZE \
--ar_steps $AR_STEPS \
--step_length 1 \
--control_only 1 \
--epochs $EPOCHS \
--n_workers $N_WORKERS \
--gpus $N_GPUS \
--track_emissions True \
--restore_opt 1 \
--seed 756

date
