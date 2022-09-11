
export NICL_ROOT=`pwd`
export PYTHONPATH=$NICL_ROOT:$PYTHONPATH
export CONFIG_PATH=$NICL_ROOT/configs


export CUDA_VISIBLE_DEVICES=$1
export HF_HOME=~/.cache/huggingface

seed=$2
n_classes=2
for dataset in realsumm # wmt-de-en wmt-zh-en
do
    export OUTPUT_PATH=$NICL_ROOT/exp_out/${dataset}
    #for seed in 123 # 532 89273 9124 9845 9284 923572 858 7445 12315
    #do
    CONFIG_FILES=t03b.json+ia3.json+${dataset}_${n_classes}class.json
    for k in 100 64 32 16 4 2
    do
        EXP_NAME=t03b_${dataset}_${n_classes}class_seed${seed}_shots${k}_ia3_pretrained100k_std_loss
        python -m src.pl_train -c ${CONFIG_FILES} -k load_weight="pretrained_checkpoints/t03b_ia3_finish.pt" exp_name=${EXP_NAME} few_shot=True num_shot=${k} few_shot_random_seed=${seed} seed=${seed} lr=3e-4 mc_loss=0 length_norm=0 unlikely_loss=0
    done
    #done
done
