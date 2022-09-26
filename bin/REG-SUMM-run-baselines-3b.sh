#!/usr/bin/bash
export NICL_ROOT=`pwd`
export PYTHONPATH=$NICL_ROOT:$PYTHONPATH
export CONFIG_PATH=$NICL_ROOT/configs
export HF_HOME=~/.cache/huggingface

export CUDA_VISIBLE_DEVICES=$1
EXP_DESC="_balanced"
model="t03b"

set -e

for dataset in realsumm_reg
do
    export OUTPUT_PATH=$NICL_ROOT/experiments${EXP_DESC}/${dataset}

    echo ------------------------------------------------------------------------------
    echo "Baseline 1: Fine tune TO on training set (w/ IA3): from checkpoint"
    echo ------------------------------------------------------------------------------
    experiment_name=${model}__baseline_ia3_pretrained100k_ft_train_ia3
    python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True exp_name=${experiment_name}
   
    echo ------------------------------------------------------------------------------
    echo "Baseline 2: Fine tune TO on training set (w/ IA3): no checkpoint"
    echo ------------------------------------------------------------------------------
    experiment_name=${model}__baseline_no_ckpt_ft_train_ia3
    python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k save_model=True exp_name=${experiment_name}
    
    echo ------------------------------------------------------------------------------
    echo "Baseline 3: Fine tune TO on training set (w/o IA3)"
    echo ------------------------------------------------------------------------------
    experiment_name=${model}__baseline_no_ckpt_ft_train_no_ia3
    python -m src.pl_train -c ${model}.json+pretrain.json+${dataset}.json -k save_model=True exp_name=${experiment_name} num_steps=1000

    echo ------------------------------------------------------------------------------
    echo "Baseline 4: Fine tune TO on training set (w/o IA3): from checkpoint"
    echo ------------------------------------------------------------------------------
    experiment_name=${model}__baseline__ia3_pretrained100k_ft_train_no_ia3
    python -m src.pl_train -c ${model}.json+ia3.json+pretrain.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True exp_name=${experiment_name} num_steps=1000

    # echo ------------------------------------------------------------------------------
    # echo "Baseline 4: Fine tune TO on training set (w/ IA3) for 10k steps"
    # echo ------------------------------------------------------------------------------
    # experiment_name=${model}__baseline_ia3_pretrained100k_ft_train_ia3_10k_steps
    # python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True exp_name=${experiment_name}
    
    # echo ------------------------------------------------------------------------------
    # echo "Baseline 5: Fine tune TO on training set (w/o IA3) for 10k steps without checkpoint"
    # echo ------------------------------------------------------------------------------
    # experiment_name=${model}__baseline_no_ckpt_ft_train_ia3_10k_steps_no_ckpt
    # python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k save_model=True exp_name=${experiment_name} num_steps=10000
    
    # echo ------------------------------------------------------------------------------
    # echo "Baseline 6: Fine tune TO on training set (w/o IA3): 10k steps"
    # echo ------------------------------------------------------------------------------
    # experiment_name=${model}__baseline_no_ckpt_ft_train_no_ia3_10k_steps
    # python -m src.pl_train -c ${model}.json+pretrain.json+${dataset}.json -k save_model=True exp_name=${experiment_name} num_steps=10000 
done