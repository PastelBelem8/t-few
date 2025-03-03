import os
import torch
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import EvalDataModule, get_dataset_reader
from src.models.EncoderDecoder import EncoderDecoder, EncoderDecoderRegression
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
    datamodule = EvalDataModule(config, tokenizer, dataset_reader)
    if config.use_regress:
        model = EncoderDecoderRegression(config, tokenizer, model, dataset_reader)
    else:
        model = EncoderDecoder(config, tokenizer, model, dataset_reader)
    logger = TensorBoardLogger(config.exp_dir, name="log")

    trainer = Trainer(
        enable_checkpointing=False,
        gpus=torch.cuda.device_count(),
        precision=config.compute_precision,
        amp_backend="native",
        strategy=config.compute_strategy if config.compute_strategy != "none" else None,
        logger=logger,
    )

    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)


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

    print(f"Mark experiment {config.exp_name} as claimed")
    with open(config.finish_flag_file, "a+") as f:
        f.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + "\n")
    set_seeds(config.seed)
    main(config)
