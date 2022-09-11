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

EXP_DESC="_2022_sep_10"
model="t03b"

for n_classes in 2 5
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
        OUTPUT_PATH=$NICL_ROOT/exp_out${EXP_DESC}/${dataset}/${n_classes}class
        
        CONFIG_FILES=${model}.json+ia3.json+wmt.json
        CONFIG_DATASET="WMT_${src_lang}_en"
        CONFIG_DATA_DIR=\"/home/cbelem/projects/GEM-workshop-peft-experiments/experiments/mt_data/preproc/${dataset}\"
        
        # ------------------------------------------------------------------------------
        # Baseline 1: TO 0-shot 
        # ------------------------------------------------------------------------------
        # Evaluate Models in a zero shot (no intermediate pretraining). It is a lower
        # bound and gives us an idea of how much we have to improve.
        # ------------------------------------------------------------------------------
        echo "Evaluate $model in a zero-shot baseline"
        orig_exp_name=${model}_${dataset}_baseline
        python -m src.pl_eval -c ${CONFIG_FILES} -k dataset_classes=${n_classes} exp_name=${orig_exp_name} save_model=False dataset=${CONFIG_DATASET} data_dir=${CONFIG_DATA_DIR}


        # ------------------------------------------------------------------------------
        # Baseline 2: Fine tune TO on training set (w/ IA3)
        # ------------------------------------------------------------------------------
        # Evaluate Models in a zero shot (no intermediate pretraining). It is a lower
        # bound and gives us an idea of how much we have to improve.
        # ------------------------------------------------------------------------------
        echo "Fine tuning ${model} w/ (IA)3"
        orig_exp_name=${model}_${dataset}_baseline_ft_train
        python -m src.pl_train -c ${CONFIG_FILES} -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True dataset_classes=${n_classes} exp_name=${orig_exp_name} dataset=${CONFIG_DATASET} data_dir=${CONFIG_DATA_DIR}
    

        echo "Fine tuning ${model} w/ (IA)3 for 10k steps"
        orig_exp_name=${model}_${dataset}_baseline_ft_train_10k_steps
        python -m src.pl_train -c ${CONFIG_FILES} -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True dataset_classes=${n_classes} exp_name=${orig_exp_name} num_steps=10000 dataset=${CONFIG_DATASET} data_dir=${CONFIG_DATA_DIR}
 
        # ------------------------------------------------------------------------------
        # Baseline 3: Fine tune TO on all sets (w/ IA3)
        # ------------------------------------------------------------------------------
        # Evaluate Models in a zero shot (no intermediate pretraining). It is a lower
        # bound and gives us an idea of how much we have to improve.
        # ------------------------------------------------------------------------------
        echo "Fine tuning ${model} w/ (IA)3 on all data"
        orig_exp_name=${model}_${dataset}_baseline_ft_all
        python -m src.pl_train -c ${CONFIG_FILES} -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" fine_tune_with_all=True save_model=true dataset_classes=${n_classes} exp_name=${orig_exp_name} few_shot=False dataset=${CONFIG_DATASET} data_dir=${CONFIG_DATA_DIR}
    

        echo "Fine tuning ${model} w/ (IA)3 on all data (num_steps=10k)"
        orig_exp_name=${model}_${dataset}_baseline_ft_all_10k_steps
        python -m src.pl_train -c ${CONFIG_FILES} -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" fine_tune_with_all=True save_model=true dataset_classes=${n_classes} exp_name=${orig_exp_name} few_shot=False num_steps=10000 dataset=${CONFIG_DATASET} data_dir=${CONFIG_DATA_DIR}
    done
done