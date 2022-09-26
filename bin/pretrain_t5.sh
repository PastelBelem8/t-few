#!/usr/bin/bash

export NICL_ROOT=`pwd`
export PYTHONPATH=$NICL_ROOT:$PYTHONPATH
export CONFIG_PATH=$NICL_ROOT/configs

export CUDA_VISIBLE_DEVICES=$1
export HF_HOME=~/.cache/huggingface

CUDA_VISIBLE_DEVICES=$1
MODEL_NAME=$2

for DATASET in realsumm_reg
do
    MODEL_CONFIG="origin_model=${MODEL_NAME} compute_precision=bf16"
    EXP_NAME="exp_name=${MODEL_NAME}_pretrain"
    python -m src.pl_train -c ${DATASET}.json+pretrain.json -k compute_strategy="ddp" $MODEL_CONFIG $EXP_NAME allow_skip_exp=False batch_size=8

    # EXP_NAME="exp_name=${MODEL_NAME}_pretrain_ia3_without_ul_and_ln"
    #python -m src.pl_train -c ia3_without_ul_and_ln.json+pretrain.json -k compute_strategy="ddp" exp_name=t03b_pretrain_ia3 allow_skip_exp=False batch_size=8
done