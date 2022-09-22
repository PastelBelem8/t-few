#!/usr/bin/env bash

set -exu

model=$1
dataset=$2
template_idx=$3
exp_dir=$4

export NICL_ROOT=`pwd`
export CONFIG_PATH=$NICL_ROOT/configs
export HF_HOME=~/.cache/huggingface
export OUTPUT_PATH=$NICL_ROOT/${exp_dir}/evals
export CUDA_VISIBLE_DEVICES=1

echo ------------------------------------------------------------------------------
echo "Evaluate: $model in a zero-shot fashion"
echo ------------------------------------------------------------------------------
experiment_name=${model}__baseline
python -m src.pl_train -c ${model}.json+${dataset}.json -k exp_name=${experiment_name}_template_${template_idx} save_model=False allow_skip_exp=False num_steps=0 eval_template_idx=${template_idx} eval_before_training=True

echo ------------------------------------------------------------------------------
echo "Evaluate: $model with intermediate pretraining in a zero-shot fashion"
echo ------------------------------------------------------------------------------
experiment_name=${model}__baseline_ia3_pretrained100k
python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" exp_name=${experiment_name}_template_${template_idx} save_model=False allow_skip_exp=False num_steps=0 eval_template_idx=${template_idx} eval_before_training=True

echo ------------------------------------------------------------------------------
echo "Baseline 1: Fine tune TO on training set (w/ IA3): from checkpoint"
echo ------------------------------------------------------------------------------
orig_exp_name=${model}__baseline_ia3_pretrained100k_ft_train_ia3
python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight=${exp_dir}/${orig_exp_name}/finish.pt save_model=False exp_name=${orig_exp_name}_template_${template_idx} allow_skip_exp=False num_steps=0 eval_template_idx=${template_idx} eval_before_training=True

echo ------------------------------------------------------------------------------
echo "Baseline 2: Fine tune TO on training set (w/ IA3): no checkpoint"
echo ------------------------------------------------------------------------------
orig_exp_name=${model}__baseline_no_ckpt_ft_train_ia3
python -m src.pl_train -c ${model}.json+ia3.json+${dataset}.json -k load_weight=${exp_dir}/${orig_exp_name}/finish.pt save_model=False exp_name=${orig_exp_name}_template_${template_idx} allow_skip_exp=False num_steps=0 eval_template_idx=${template_idx} eval_before_training=True

echo ------------------------------------------------------------------------------
echo "Baseline 3: Fine tune TO on training set (w/o IA3)"
echo ------------------------------------------------------------------------------
orig_exp_name=${model}__baseline_no_ckpt_ft_train_no_ia3
python -m src.pl_train -c ${model}.json+pretrain.json+${dataset}.json -k load_weight=${exp_dir}/${orig_exp_name}/finish.pt save_model=False exp_name=${orig_exp_name}_template_${template_idx} allow_skip_exp=False num_steps=0 eval_template_idx=${template_idx}  eval_before_training=True
