#!/bin/bash -eu
#SBATCH --job-name=calibration_ner
#SBATCH --cpus-per-task=8
#SBATCH --output=output.%J.log
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00


export BASE_DATA_DIR=/home/data
export BASE_DIR=/home/ner_da_uncertainty
export TODAYDATE=$(date '+%Y-%m-%d-%H-%M-%S')
export MAX_LENGTH=512
export BERT_MODEL=microsoft/mdeberta-v3-base
export MODEL_TYPE=MentionReplacementDeBERTaV3ForTokenClassification
export BATCH_SIZE=32
export DATA_NAME=ontonotes5_bn
export TEST_DATA_NAME=ontonotes5_bn
export DATA_PATH=$BASE_DATA_DIR/ner_data/$DATA_NAME
export TEST_DATA_PATH=$BASE_DATA_DIR/ner_data/$TEST_DATA_NAME
export SEED=1
export TAU=1.10
export SMOOTHING=0.01


OUTPUT_DIR="$BASE_DIR/models/$DATA_NAME-$TEST_DATA_NAME-$MODEL_TYPE"


if [[ "$MODEL_TYPE" == *"LabelSmoothing"* ]] ; then
    OUTPUT_DIR+="-smooth$SMOOTHING"
fi

export NUM_EPOCHS=200
export EVAL_STEPS=200
export LEARNING_RATE=1e-5
export RUN_MULTIPLE_SEEDS=true
export NUM_MONTE_CARLO=20


mkdir -p $OUTPUT_DIR

cd $BASE_DIR

module load singularity

singularity run --nv /home/ner_da_uncertainty/research.sif /opt/conda/bin/python src/train_from_json.py \
--model_name_or_path $BERT_MODEL \
--data_path $DATA_PATH \
--test_data_path $TEST_DATA_PATH \
--labels $DATA_PATH/labels.txt \
--output_dir $OUTPUT_DIR \
--model_type $MODEL_TYPE \
--max_seq_length $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--seed $SEED \
--use_fast \
--bf16 \
--learning_rate $LEARNING_RATE \
--eval_steps $EVAL_STEPS \
--run_multiple_seeds $RUN_MULTIPLE_SEEDS \
--tau $TAU \
--num_monte_carlo $NUM_MONTE_CARLO \
--smoothing $SMOOTHING
