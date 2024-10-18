#!/bin/bash
# The job name
#SBATCH --job-name=trainlam
# Set the error and output files
#SBATCH --output=trainlam-%J.out
#SBATCH --error=trainlam-%J.out
# Set the initial working directory
#SBATCH --chdir=/scratch/dume/spool
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

module load conda
mamba activate mllam_v2
# On reaserve, environment must be loaded before executing the code
#module load conda
#mamba activate neurallam

echo "Env successfully loaded!"
python --version
date

#DATASET=mera_10years_fullres
#MODEL=hi_lam # what other models>
#BATCHSIZE=2
#EPOCHS=150
#N_WORKERS=8
#GRAPH=multiscale
#AR_STEPS=6

#set -vx

#python $HOME/neural-lam/scripts/train_model.py --dataset $DATASET --graph $GRAPH --model $MODEL --batch_size $BATCHSIZE --ar_steps $AR_STEPS --step_length 1 --control_only 1 --epochs $EPOCHS --n_workers $N_WORKERS --track_emissions False

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

DATASET=mera_3years_fullres
GRAPH=multiscale
MODEL=graph_lam   # graph_lam, hi_lam or hi_lam_parallel
BATCHSIZE=2
EPOCHS=100
AR_STEPS=1
#STARTFROM= 

set -vx

srun --cpus-per-gpu $N_WORKERS python $HOME/neural-lam/scripts/train_model.py \
--dataset $DATASET \
--graph $GRAPH \
--model $MODEL \
--batch_size $BATCHSIZE \
--ar_steps $AR_STEPS \
--step_length 1 \
--control_only 1 \
--epochs $EPOCHS \
--n_workers $N_WORKERS \
--track_emissions False \
--restore_opt 1 \
--seed 756

date
