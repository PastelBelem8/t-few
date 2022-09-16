#!/usr/bin/bash
export NICL_ROOT=`pwd`
export PYTHONPATH=$NICL_ROOT:$PYTHONPATH
export CONFIG_PATH=$NICL_ROOT/configs

export CUDA_VISIBLE_DEVICES=$1
export HF_HOME=~/.cache/huggingface

EXP_DESC=$2
seed=12315
model="t03b"

set -e

for dataset in realsumm_reg
do
    export OUTPUT_PATH=$NICL_ROOT/results/${EXP_DESC}/${dataset}/regression
    echo ------------------------------------------------------------------------------
    echo
    echo 
    echo "Running Evaluation for regression experiment at ${EXP_DESC}"
    echo 
    echo
    echo ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # Baseline 1: TO 0-shot 
    # ------------------------------------------------------------------------------
    experiment_name=${model}_${dataset}_baseline
    echo "--> Evaluate $model in a zero-shot baseline: ${experiment_name}"
    python -m src.pl_eval -c ${model}.json+${dataset}.json -k exp_name=${experiment_name} save_model=False seed=${seed}
    
    # ------------------------------------------------------------------------------
    # Baseline 2: Fine tune TO on training set (w/ IA3)
    # ------------------------------------------------------------------------------
    experiment_name=${model}_${dataset}_baseline_train_ia3
    echo "--> Fine-tuning ${model} w/ (IA)3: ${experiment_name}"
    python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True exp_name=${experiment_name} seed=${seed}
    
    # ------------------------------------------------------------------------------
    # Baseline 4: Fine tune TO on training set (w/ IA3) 5K steps
    # ------------------------------------------------------------------------------
    experiment_name=${model}_${dataset}_baseline_train_ia3_5k_steps
    echo "--> Fine-tuning ${model} w/ (IA)3 for 10k steps: ${experiment_name}"
    python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True exp_name=${experiment_name} seed=${seed}
    
    # ------------------------------------------------------------------------------
    # Baseline 3: Fine tune TO on training set (w/o IA3)
    # ------------------------------------------------------------------------------
    export CUDA_VISIBLE_DEVICES=2,3,4,5
    experiment_name=${model}_${dataset}_baseline_ft_train_no_ia3
    echo "--> Fine-tuning ${model} w/o (IA)3: ${experiment_name}"
    python -m src.pl_train -c ${model}.json+pretrain.json+${dataset}.json -k save_model=True exp_name=${experiment_name} num_steps=1000 compute_strategy="ddp" eval_epoch_interval=500 seed=${seed}

    # ------------------------------------------------------------------------------
    # Baseline 35: Fine tune TO on training set (w/o IA3) 5K steps
    # ------------------------------------------------------------------------------
    experiment_name=${model}_${dataset}_baseline_ft_train_no_ia3_5k_steps
    echo "--> Fine-tuning ${model} w/o (IA)3 for 10k steps: ${experiment_name}"
    python -m src.pl_train -c ${model}.json+pretrain.json+${dataset}.json -k save_model=True exp_name=${experiment_name} num_steps=10000 compute_strategy="ddp" eval_epoch_interval=500 seed=${seed}
done