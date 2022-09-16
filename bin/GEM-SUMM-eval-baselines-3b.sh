#!/usr/bin/bash

set -e

export NICL_ROOT=`pwd`
export PYTHONPATH=$NICL_ROOT:$PYTHONPATH
export CONFIG_PATH=$NICL_ROOT/configs

export CUDA_VISIBLE_DEVICES=$1
export HF_HOME=~/.cache/huggingface

EXP_DESC="_balanced"
model="t03b"

for n_classes in 2
do
    for dataset in realsumm
    do
        export OUTPUT_PATH=$NICL_ROOT/experiments${EXP_DESC}/${dataset}/${n_classes}class

        echo ------------------------------------------------------------------------------
        echo Baseline 1: TO 0-shot 
        echo ------------------------------------------------------------------------------
        echo "Evaluate $model in a zero-shot baseline"
        experiment_name=${model}_baseline
        python -m src.pl_eval -c ${model}.json+${dataset}.json -k dataset_classes=${n_classes} exp_name=${experiment_name} save_model=False

        echo ------------------------------------------------------------------------------
        echo "Baseline 2: Fine tune TO on training set (w/ IA3)"
        echo ------------------------------------------------------------------------------
        experiment_name=${model}_ia3_pretrained100k__baseline
        python -m src.pl_eval -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" dataset_classes=${n_classes} exp_name=${experiment_name} save_model=False
        

        echo ------------------------------------------------------------------------------
        echo "Baseline 2: Fine tune TO on training set (w/ IA3)"
        echo ------------------------------------------------------------------------------
        experiment_name=${model}_ia3_pretrained100k__baseline_train_ia3
        python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True dataset_classes=${n_classes} exp_name=${experiment_name}
        
        echo ------------------------------------------------------------------------------
        echo "Baseline 3: Fine tune TO on training set (w/o IA3)"
        echo ------------------------------------------------------------------------------
        experiment_name=${model}__baseline_ft_train_no_ia3
        python -m src.pl_train -c ${model}.json+pretrain.json+${dataset}.json -k save_model=True dataset_classes=${n_classes} exp_name=${experiment_name} num_steps=1000

        echo ------------------------------------------------------------------------------
        echo "Baseline 4: Fine tune TO on training set (w/ IA3)"
        echo ------------------------------------------------------------------------------
        experiment_name=${model}_ia3_pretrained100k_baseline_train_ia3_5k_steps
        python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True dataset_classes=${n_classes} exp_name=${experiment_name}
        
        echo ------------------------------------------------------------------------------
        echo "Baseline 5: Fine tune TO on training set (w/o IA3)"
        echo ------------------------------------------------------------------------------
        echo "Fine tuning ${model} w/ (IA)3 for 10k steps"
        experiment_name=${model}_ia3_pretrained100k__baseline_ft_train_5k_steps
        python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True dataset_classes=${n_classes} exp_name=${experiment_name} num_steps=10000
        
        echo ------------------------------------------------------------------------------
        echo "Baseline 6: Fine tune TO on training set (w/o IA3)"
        echo ------------------------------------------------------------------------------
        echo "Fine tuning ${model} w/o (IA)3 for 10k steps"
        experiment_name=${model}__baseline_ft_train_no_ia3
        python -m src.pl_train -c ${model}.json+pretrain.json+${dataset}.json -k save_model=True dataset_classes=${n_classes} exp_name=${experiment_name} num_steps=10000 compute_strategy="ddp"
    done
done