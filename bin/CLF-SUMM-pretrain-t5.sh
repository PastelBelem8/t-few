#!/usr/bin/bash
set -e

export NICL_ROOT=`pwd`
export PYTHONPATH=$NICL_ROOT:$PYTHONPATH
export CONFIG_PATH=$NICL_ROOT/configs

export CUDA_VISIBLE_DEVICES=$1
export HF_HOME=~/.cache/huggingface

MODEL_NAME=$2
EXP_DESC="_balanced"

for n_classes in 2
do
    for DATASET in realsumm
    do
        export OUTPUT_PATH=$NICL_ROOT/experiments${EXP_DESC}/${DATASET}_${n_classes}class
        DATASET_CLASSES="dataset_classes=${n_classes}"

        MODEL_CONFIG="origin_model=\"${MODEL_NAME}\" compute_precision=bf16"
        MODEL_NAME=$(echo $MODEL_NAME | tr - _)

        EXP_NAME="exp_name=baseline__${MODEL_NAME}_pretrain_1k_no_ia3"
        python -m src.pl_train -c ${DATASET}.json+pretrain.json -k $MODEL_CONFIG $EXP_NAME allow_skip_exp=False batch_size=8 num_steps=1000 eval_epoch_interval=1000  $DATASET_CLASSES eval_before_training=False # compute_strategy="ddp"

        EXP_NAME="exp_name=baseline__${MODEL_NAME}_pretrain_10k_no_ia3"
        python -m src.pl_train -c ${DATASET}.json+pretrain.json -k $MODEL_CONFIG $EXP_NAME allow_skip_exp=False batch_size=8 num_steps=10000 eval_epoch_interval=1000 $DATASET_CLASSES eval_before_training=False # compute_strategy="ddp"

        EXP_NAME="exp_name=baseline__${MODEL_NAME}_pretrain_ia3_without_ul_and_ln"
        python -m src.pl_train -c ${DATASET}.json+pretrain.json+ia3_without_ul_and_ln.json -k $MODEL_CONFIG $EXP_NAME allow_skip_exp=False batch_size=8 eval_epoch_interval=100 $DATASET_CLASSES eval_before_training=False  # compute_strategy="ddp"
    done
done