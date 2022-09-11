#!/usr/bin/bash
set -e

export NICL_ROOT=`pwd`
export PYTHONPATH=$NICL_ROOT:$PYTHONPATH
export CONFIG_PATH=$NICL_ROOT/configs


export CUDA_VISIBLE_DEVICES=$1
export HF_HOME=~/.cache/huggingface


EXP_DESC="_2022_sep_10"

seed=$2
for n_classes in 2 # 5
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
        export OUTPUT_PATH=$NICL_ROOT/exp_out${EXP_DESC}/${dataset}/${n_classes}class

        CONFIG_FILES=t03b.json+ia3.json+wmt.json
        CONFIG_DATASET="WMT_${src_lang}_en"
        CONFIG_DATA_DIR=\"/home/cbelem/projects/GEM-workshop-peft-experiments/experiments/mt_data/preproc/${dataset}\"

        for k in 128 100 64 32 16 4 2
        do
            EXP_NAME=t03b_${dataset}_seed${seed}_shots${k}_ia3_pretrained100k 
            python -m src.pl_train -c ${CONFIG_FILES} -k load_weight="pretrained_checkpoints/t03b_ia3_finish.pt" dataset_classes=${n_classes} exp_name=${EXP_NAME} few_shot=True num_shot=${k} few_shot_random_seed=${seed} seed=${seed} dataset=${CONFIG_DATASET} data_dir=${CONFIG_DATA_DIR}
        done
    done
done
