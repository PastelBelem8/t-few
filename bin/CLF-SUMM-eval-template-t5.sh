#!/usr/bin/bash

set -exu

export NICL_ROOT=`pwd`
# export PYTHONPATH=$NICL_ROOT:$PYTHONPATH
export CONFIG_PATH=$NICL_ROOT/configs
export CUDA_VISIBLE_DEVICES=$1
export HF_HOME=~/.cache/huggingface

MODEL_NAME=$2
EXP_DESC="_balanced"

for n_classes in 2
do
    for DATASET in realsumm
    do
        exp_dir=$NICL_ROOT/experiments${EXP_DESC}/${DATASET}_${n_classes}class
        export OUTPUT_PATH=${exp_dir}/evals

        DATASET_CLASSES="dataset_classes=${n_classes}"
        MODEL_CONFIG="origin_model=\"${MODEL_NAME}\" compute_precision=bf16"
        MODEL_NAME=$(echo $MODEL_NAME | tr - _)

        orig_exp_name="baseline__${MODEL_NAME}_pretrain_1k_no_ia3"
        python -m src.pl_train -c ${DATASET}.json+pretrain.json -k $MODEL_CONFIG load_weight=\"${exp_dir}/${orig_exp_name}/finish.pt\" save_model=False exp_name=${orig_exp_name}_template_0 allow_skip_exp=False num_steps=0 eval_template_idx=0 eval_before_training=True

        orig_exp_name="baseline__${MODEL_NAME}_pretrain_10k_no_ia3"
        python -m src.pl_train -c ${DATASET}.json+pretrain.json -k $MODEL_CONFIG load_weight=\"${exp_dir}/${orig_exp_name}/finish.pt\" save_model=False exp_name=${orig_exp_name}_template_0 allow_skip_exp=False num_steps=0 eval_template_idx=0 eval_before_training=True
        orig_exp_name="baseline__${MODEL_NAME}_pretrain_ia3_without_ul_and_ln"
        python -m src.pl_train -c ${DATASET}.json+pretrain.json+ia3_without_ul_and_ln.json -k $MODEL_CONFIG load_weight=\"${exp_dir}/${orig_exp_name}/finish.pt\" save_model=False exp_name=${orig_exp_name}_template_0 allow_skip_exp=False num_steps=0 eval_template_idx=0 eval_before_training=True

    done
done
