from dataclasses import is_dataclass
import os
import torch
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.data import FinetuneDatasetWithTemplate, get_dataset_reader, create_collate_fn
from src.models.modify_model import modify_transformer
from src.utils.Config import Config
from src.utils.util import ParseKwargs, set_seeds


def get_transformer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.origin_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.origin_model, low_cpu_mem_usage=True)

    tokenizer.model_max_length = config.max_seq_len
    model = modify_transformer(model, config)
    return tokenizer, model


def main(config):
    """
    Trains the model

    :param config:
    :return:
    """

    tokenizer, model = get_transformer(config)
    dataset_reader = get_dataset_reader(config)
    
    extra_args = {"add_special_tokens": config.add_special_tokens}

    dev_dataset = dataset_reader.read_orig_dataset("validation") # Dataset
    dev_dataset = FinetuneDatasetWithTemplate(
        dev_dataset, dataset_reader.get_eval_template(), tokenizer, **extra_args
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=create_collate_fn(tokenizer.pad_token_id, pretrain=False),
        num_workers=min([config.eval_batch_size, config.num_workers]),
    )

    count_seqs_truncated = 0
    dataset_size = 0
    ids = []

    for i, batch in enumerate(dev_loader):
        dataset_size += len(batch)
        batch_ids = batch["idx"]
        input_seq_len = batch["input_seq_len"]
        truncated_exs = [(l, i.item()) for l, i in zip(input_seq_len, batch_ids) if l >= 256]
        if truncated_exs:
            input_seq_len, batch_ids = zip(*truncated_exs)
            count_seqs_truncated += len(input_seq_len)
            ids.extend(batch_ids)


    print()
    print("Truncated (out of total dev examples):", count_seqs_truncated, f"({dataset_size})")
    print(ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_files, args.kwargs)
    print(f"Start experiment {config.exp_name}")
    # Setup config
    assert config.compute_strategy in ["none", "ddp", "deepspeed_stage_3_offload", "deepspeed_stage_3"]
    if config.fishmask_mode == "create":
        print("Detecting fishmask_mode=create, override batch_size, num_step, fishmask_path")
        config.batch_size = 1
        config.num_steps = config.num_shot
        config.eval_before_training = False
        config.fishmask_path = None

    print(config.to_json())

    if config.allow_skip_exp and os.path.exists(config.finish_flag_file):
        print(f"Skip finished experiment {config.exp_name}")
    else:
        print(f"Mark experiment {config.exp_name} as claimed")
        with open(config.finish_flag_file, "a+") as f:
            f.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + "\n")
        set_seeds(config.seed)
        main(config)
