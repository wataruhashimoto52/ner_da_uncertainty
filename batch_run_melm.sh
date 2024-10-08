#!/bin/bash -eu
#SBATCH --job-name=melm
#SBATCH --cpus-per-task=8
#SBATCH --output=output.%J.log
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00


export BASE_DATA_DIR=/home/data
export BASE_DIR=/home/ner_da_uncertainty
export DATA_NAME=multiconer_EN
export DATA_PATH=$BASE_DATA_DIR/ner_data/$DATA_NAME
export LABEL_PATH=$DATA_PATH/labels.txt
export MASK_RATE=0.7  # (0.3, 0.5, 0.7)
export MU_RATIO=0.7  # (0.3, 0.5, 0.7)
export OUTPUT_PATH=$BASE_DIR/models/$DATA_NAME-MELM-mask$MASK_RATE-mu$MU_RATIO-mdeberta
export MODEL_NAME_OR_PATH=microsoft/mdeberta-v3-base


mkdir -p $OUTPUT_PATH

module load singularity

cd $BASE_DIR

singularity run --nv /home/ner_da_uncertainty/research.sif /opt/conda/bin/python src/melm_augment.py \
 --seed 42 \
 --base_model_name_or_path $MODEL_NAME_OR_PATH \
 --label_path $LABEL_PATH \
 --data_path $DATA_PATH \
 --output_path $OUTPUT_PATH \
 --mask_rate $MASK_RATE \
 --mu_ratio $MU_RATIO
