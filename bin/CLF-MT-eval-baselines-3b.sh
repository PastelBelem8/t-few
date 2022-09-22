#!/usr/bin/bash
set -e

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

EXP_DESC="_balanced"
model="t03b"

for n_classes in 2
do
    
    for src_lang in de zh ru "fi" gu lt kk
    do
        dataset=${src_lang}-en
        echo --------------------------------------------------------------------------------
        echo
        echo
        echo ${dataset} ${n_classes} classes
        echo
        echo
        echo --------------------------------------------------------------------------------

        export OUTPUT_PATH=$NICL_ROOT/experiments${EXP_DESC}/${dataset}/${n_classes}class

        CONFIG_DATASET="WMT_${src_lang}_en"
        CONFIG_DATA_DIR=\"/home/cbelem/projects/GEM-workshop-peft-experiments/datasets/mt_data/preproc/${dataset}/quantile\"        

        echo ------------------------------------------------------------------------------
        echo "Baseline 1: Fine tune TO on training set (w/ IA3): from checkpoint"
        echo ------------------------------------------------------------------------------
        experiment_name=${model}__baseline_ia3_pretrained100k_ft_train_ia3
        python -m src.pl_train -c ${model}.json+ia3.json+wmt.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True exp_name=${experiment_name} dataset=${CONFIG_DATASET} data_dir=${CONFIG_DATA_DIR}
    
        echo ------------------------------------------------------------------------------
        echo "Baseline 2: Fine tune TO on training set (w/ IA3): no checkpoint"
        echo ------------------------------------------------------------------------------
        experiment_name=${model}__baseline_no_ckpt_ft_train_ia3
        python -m src.pl_train -c ${model}.json+ia3.json+wmt.json -k save_model=True exp_name=${experiment_name} dataset=${CONFIG_DATASET} data_dir=${CONFIG_DATA_DIR}
        
        echo ------------------------------------------------------------------------------
        echo "Baseline 3: Fine tune TO on training set (w/o IA3)"
        echo ------------------------------------------------------------------------------
        experiment_name=${model}__baseline_no_ckpt_ft_train_no_ia3
        python -m src.pl_train -c ${model}.json+pretrain.json+wmt.json -k save_model=True exp_name=${experiment_name} num_steps=1000 dataset=${CONFIG_DATASET} data_dir=${CONFIG_DATA_DIR}

        # echo ------------------------------------------------------------------------------
        # echo "Baseline 4: Fine tune TO on training set (w/o IA3): from checkpoint"
        # echo ------------------------------------------------------------------------------
        # experiment_name=${model}__baseline__ia3_pretrained100k_ft_train_no_ia3
        # python -m src.pl_train -c ${model}.json+pretrain.json+wmt.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True exp_name=${experiment_name} num_steps=1000 dataset=${CONFIG_DATASET} data_dir=${CONFIG_DATA_DIR}
        done
done