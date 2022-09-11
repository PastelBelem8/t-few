#!/usr/bin/bash
# ------------------------------------------------------------------------------
# We aim to run the baselines for the different datasets. The baselines
# include running the evaluation on the dev set for the following models:
# - T0
# - T0-finetuned on the whole training set
# - T0-finetuned on the whole dataset (still evaluated just on the dev set)
# ------------------------------------------------------------------------------
export NICL_ROOT=`pwd`
export PYTHONPATH=$NICL_ROOT:$PYTHONPATH
export CONFIG_PATH=$NICL_ROOT/configs


export CUDA_VISIBLE_DEVICES=$1
export HF_HOME=~/.cache/huggingface

EXP_DESC="_2022_sep_10"
model="t03b"

for n_classes in 2 5
do
    # ------------------------------------------------------------------------------
    # Baseline 1: TO 0-shot 
    # ------------------------------------------------------------------------------
    # Evaluate Models in a zero shot (no intermediate pretraining). It is a lower
    # bound and gives us an idea of how much we have to improve.
    # ------------------------------------------------------------------------------
    echo "Evaluate $model in a zero-shot baseline"
    for dataset in realsumm
    do
        export OUTPUT_PATH=$NICL_ROOT/exp_out${EXP_DESC}/${dataset}/${n_classes}class
        export orig_exp_name=${model}_${dataset}_baseline
        python -m src.pl_eval -c ${model}.json+${dataset}.json -k dataset_classes=${n_classes} exp_name=${orig_exp_name} save_model=False
    done


    # ------------------------------------------------------------------------------
    # Baseline 2: Fine tune TO on training set (w/ IA3)
    # ------------------------------------------------------------------------------
    # Evaluate Models in a zero shot (no intermediate pretraining). It is a lower
    # bound and gives us an idea of how much we have to improve.
    # ------------------------------------------------------------------------------
    echo "Fine tuning ${model} w/ (IA)3"
    for dataset in realsumm
    do
        export OUTPUT_PATH=$NICL_ROOT/exp_out${EXP_DESC}/${dataset}/${n_classes}class
        export orig_exp_name=${model}_${dataset}_baseline_ft_train
        python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True dataset_classes=${n_classes} exp_name=${orig_exp_name}
    done

    echo "Fine tuning ${model} w/ (IA)3 for 10k steps"
    for dataset in realsumm
    do
        export OUTPUT_PATH=$NICL_ROOT/exp_out${EXP_DESC}/${dataset}/${n_classes}class
        export orig_exp_name=${model}_${dataset}_baseline_ft_train_10k_steps
        python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True dataset_classes=${n_classes} exp_name=${orig_exp_name} num_steps=10000
    done


    # ------------------------------------------------------------------------------
    # Baseline 3: Fine tune TO on all sets (w/ IA3)
    # ------------------------------------------------------------------------------
    # Evaluate Models in a zero shot (no intermediate pretraining). It is a lower
    # bound and gives us an idea of how much we have to improve.
    # ------------------------------------------------------------------------------
    echo "Fine tuning ${model} w/ (IA)3 on all data"
    for dataset in realsumm
    do
        export OUTPUT_PATH=$NICL_ROOT/exp_out${EXP_DESC}/${dataset}/${n_classes}class
        export orig_exp_name=${model}_${dataset}_baseline_ft_all
        python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" fine_tune_with_all=True save_model=true dataset_classes=${n_classes} exp_name=${orig_exp_name} few_shot=False
    done

    echo "Fine tuning ${model} w/ (IA)3 on all data (num_steps=20k)"
    for dataset in realsumm
    do
        export OUTPUT_PATH=$NICL_ROOT/exp_out${EXP_DESC}/${dataset}/${n_classes}class
        export orig_exp_name=${model}_${dataset}_baseline_ft_all_10k_steps
        python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" fine_tune_with_all=True save_model=true dataset_classes=${n_classes} exp_name=${orig_exp_name} few_shot=False num_steps=10000
    done
done