
export NICL_ROOT=`pwd`
export PYTHONPATH=$NICL_ROOT:$PYTHONPATH
export CONFIG_PATH=$NICL_ROOT/configs
export HF_HOME=~/.cache/huggingface

export CUDA_VISIBLE_DEVICES=$1
seed=$2

EXP_DESC="_balanced"
MODEL="t03b"


for n_classes in 2 # 5
do
    for dataset in realsumm
    do
        export OUTPUT_PATH=$NICL_ROOT/experiments${EXP_DESC}/${dataset}/${n_classes}class
        CONFIG_FILES=${MODEL}.json+ia3.json+${dataset}.json

        for k in 200 100 64 32 16 4 2
        do
            EXP_NAME=${MODEL}_ia3_pretrained100k__shots${k}_seed${seed}
            python -m src.pl_train -c ${CONFIG_FILES} -k load_weight="pretrained_checkpoints/${MODEL}_ia3_finish.pt" dataset_classes=${n_classes} exp_name=${EXP_NAME} few_shot=True num_shot=${k} few_shot_random_seed=${seed} seed=${seed}
            # EXP_NAME=t03b_${dataset}_seed${seed}_shots${k}_ia3_pretrained100k_ce_only
            # python -m src.pl_train -c ${CONFIG_FILES} -k load_weight="pretrained_checkpoints/t03b_ia3_finish.pt" dataset_classes=${n_classes} exp_name=${EXP_NAME} few_shot=True num_shot=${k} few_shot_random_seed=${seed} seed=${seed} mc_loss=0 unlikely_loss=0 length_norm=0   

        done
    done
done
