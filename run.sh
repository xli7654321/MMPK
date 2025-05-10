#!/bin/bash

DEVICE=${1:-0}
FOLDER=${2:-"mmpk"}
PK_PARAMS=${3:-"auc cmax tmax hl cl vz mrt f"}

for n_fold in {1..10}; do
    echo "Running Fold $n_fold"

    MODEL_PATH="checkpoint/$FOLDER/fold_$n_fold.pth"
    RESULTS_PATH="results/$FOLDER/fold_$n_fold.csv"
    
    python train.py \
        --device $DEVICE \
        --n_fold $n_fold \
        --pk_params $PK_PARAMS \
        --model_path $MODEL_PATH \
        --results_path $RESULTS_PATH \
        --batch_size 32 \
        --lr 1e-3 \
        --weight_decay 1e-5 \
        --gnn_type_mol gat \
        --gnn_type_sub gin \
        --dropout_mol 0.1 \
        --dropout_sub 0.1
done

echo "Calculating 10-fold CV results..."
RESULTS_FOLDER="results/$FOLDER/"
python utils/calc_cv.py --folder_path $RESULTS_FOLDER
echo "Done"

# bash run.sh 0 mmpk "auc cmax tmax hl cl vz mrt f"