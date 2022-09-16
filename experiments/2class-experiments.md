# Experiments Schedule 

## REALSumm 

In this section, we plan and describe the experiments conducted using the REALSumm dataset. 
We separate this experiment into a few stages: 
(1) balanced vs non-balanced; 
(2) numerical vs word;
(3) selection of few-shot examples.

### Balanced, word, random sampling (stratified by `bin`)

We collect 15 different seeds for different numbers of shots: $\{100, 64, 16, 4 , 2\}$. 
We set `add_special_tokens=False` to avoid length normalization problems. 
We'll use the data available in `./datasets/summ_data/REALSumm/quantile` with `dataset_classes=2`. 

#### Baselines 


| Name               | Machine  | Seed | Started | Command | Results |
| ------------------ | -------- | ---- | ------- | ------- | ------- |
| T0                 | s5, 3    | 100  | NO | `$ cd ~/projects/t-few && source ./bin/GEM-SUMM-eval-baselines-3b.sh 4 | [TBD]() |
| T0-FT-TRAIN        | s5, 4    | 100  | NO | --- | [TBD]() |
| T0-FT-TRAIN-PEFT   | s5, 5    | 100  | NO | --- | [TBD]() |
| T0-FT-5k          | s5, 6    | 100  | NO | --- | [TBD]() |
| T0-FT-5k          | s5, 7    | 100  | NO | --- | [TBD]() |

The script used to run this experiment was the following:

```bash
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
        experiment_name=${model}_${dataset}_baseline
        python -m src.pl_eval -c ${model}.json+${dataset}.json -k dataset_classes=${n_classes} exp_name=${experiment_name} save_model=False
        
        echo ------------------------------------------------------------------------------
        echo Baseline 2: Fine tune TO on training set (w/ IA3)
        echo ------------------------------------------------------------------------------
        experiment_name=${model}_${dataset}_baseline_train_ia3
        python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True dataset_classes=${n_classes} exp_name=${experiment_name}
        

        echo ------------------------------------------------------------------------------
        echo Baseline 3: Fine tune TO on training set (w/o IA3)
        echo ------------------------------------------------------------------------------
        experiment_name=${model}_${dataset}_baseline_ft_train_no_ia3
        python -m src.pl_train -c ${model}.json+pretrain.json+${dataset}.json -k save_model=True dataset_classes=${n_classes} exp_name=${experiment_name} num_steps=1000

        echo ------------------------------------------------------------------------------
        echo Baseline 4: Fine tune TO on training set (w/ IA3)
        echo ------------------------------------------------------------------------------
        experiment_name=${model}_${dataset}_baseline_train_ia3_5k_steps
        python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True dataset_classes=${n_classes} exp_name=${experiment_name}
        

        echo ------------------------------------------------------------------------------
        echo Baseline 5: Fine tune TO on training set (w/o IA3)
        echo ------------------------------------------------------------------------------
        echo "Fine tuning ${model} w/ (IA)3 for 10k steps"
        experiment_name=${model}_${dataset}_baseline_ft_train_5k_steps
        python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" save_model=True dataset_classes=${n_classes} exp_name=${experiment_name} num_steps=10000
        
        echo "Fine tuning ${model} w/o (IA)3 for 10k steps"
        experiment_name=${model}_${dataset}_baseline_ft_train_no_ia3
        python -m src.pl_train -c ${model}.json+pretrain.json+${dataset}.json -k save_model=True dataset_classes=${n_classes} exp_name=${experiment_name} num_steps=10000 compute_strategy="ddp"
    done
done
```


#### 

| Machine  | Seed | Started | Command | Results |
| -------- | ---- | ------- | ------- | ------- |
| s5, 4 | 9667 | 09/16 14h22 | `conda activate tfew && cd ~/projects/GEM-workshop-peft-experiments/t-few-master && source ./bin/GEM-SUMM-few-shot-pretrained-3b-100k.sh 4 9667` | TBD |
| s5, 5 | 6914 | 09/16 14h22 | `conda activate tfew && cd ~/projects/GEM-workshop-peft-experiments/t-few-master && source ./bin/GEM-SUMM-few-shot-pretrained-3b-100k.sh 5 6914` | TBD |
| s5, 6 | 572 | 09/16 14h22 | `conda activate tfew && cd ~/projects/GEM-workshop-peft-experiments/t-few-master && source ./bin/GEM-SUMM-few-shot-pretrained-3b-100k.sh 6 572` | TBD |
| s5, 7 | 2234 | 09/16 14h22 | `conda activate tfew && cd ~/projects/GEM-workshop-peft-experiments/t-few-master && source ./bin/GEM-SUMM-few-shot-pretrained-3b-100k.sh 7 2234` | TBD |
| s4, 0 | 7623 |
| s4, 1 | 823 |
| s4, 2 | 1360 |
| s4, 3 | 7968 |
| s4, 4 | 3434 |
| s4, 5 | 6555 |
| s4, 6 | 5492 |
| s4, 7 | 2495 |
| TBD   | 1190 | 
| TBD   | 1983 | 
| TBD | 1637 | 09/16 14h22 | `conda activate tfew && cd ~/projects/GEM-workshop-peft-experiments/t-few-master && source ./bin/GEM-SUMM-few-shot-pretrained-3b-100k.sh 3 1637` | TBD |
