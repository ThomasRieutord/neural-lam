#!/usr/bin/bash
#
# Run all required steps up to the training of Neural-LAM
#

DATASET=mera_example_emca
BATCHSIZE=8
EPOCHS=200
N_WORKERS=8

set -vx

python create_mesh.py --dataset $DATASET
python create_grid_features.py --dataset $DATASET
python create_parameter_weights.py --dataset $DATASET --batch_size $BATCHSIZE --step_length 1
python train_model.py --dataset $DATASET --batch_size $BATCHSIZE --ar_steps 2 --step_length 1 --control_only 1 --epochs $EPOCHS --n_workers $N_WORKERS

