
export NICL_ROOT=`pwd`
export PYTHONPATH=$NICL_ROOT:$PYTHONPATH
export CONFIG_PATH=$NICL_ROOT/configs


export CUDA_VISIBLE_DEVICES=$1
export HF_HOME=~/.cache/huggingface

seed=$2
for n_classes in 2 5
do
    for dataset in realsumm
    do
        export OUTPUT_PATH=$NICL_ROOT/exp_out_2022_sep_10/${dataset}/${n_classes}class
        CONFIG_FILES=t03b.json+ia3.json+${dataset}.json
        for k in 128 100 64 32 16 4 2
        do
            EXP_NAME=t03b_${dataset}_seed${seed}_shots${k}_ia3_pretrained100k 
            python -m src.pl_train -c ${CONFIG_FILES} -k load_weight="pretrained_checkpoints/t03b_ia3_finish.pt" dataset_classes=${n_classes} exp_name=${EXP_NAME} few_shot=True num_shot=${k} few_shot_random_seed=${seed} seed=${seed}
        done
    done
done
