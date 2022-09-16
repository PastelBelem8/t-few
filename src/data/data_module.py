import torch
import numpy as np
from pytorch_lightning import LightningDataModule


def encode_and_truncate(input_str, add_special_tokens, tokenizer):
    output = tokenizer(
        input_str,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
        # Note: overflowing tokens requires adding truncation and padding
        return_overflowing_tokens=True, 
        truncation=True, 
        padding="max_length",
    )

    overflow_map = output["overflow_to_sample_mapping"].tolist()

    keep_ids = np.unique(overflow_map) 
    trunc_ids = [i for i in range(len(overflow_map)) if i not in keep_ids]
    
    input_ids = output["input_ids"]
    num_truncated = (
        input_ids[trunc_ids,:] != tokenizer.pad_token_id
    ).sum().item()

    return input_ids.squeeze(0), num_truncated


class FinetuneDataModule(LightningDataModule):
    def __init__(self, config, tokenizer, dataset_reader):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_reader = dataset_reader

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        if self.config.few_shot:
            _ = self.dataset_reader.read_few_shot_dataset()

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if self.config.few_shot:
            self.train_dataset = self.dataset_reader.read_few_shot_dataset() # list of dicts
        else:
            self.train_dataset = self.dataset_reader.read_orig_dataset("train")
        self.dev_dataset = self.dataset_reader.read_orig_dataset("validation") # Dataset

        extra_args = {"add_special_tokens": self.config.add_special_tokens}
        self.train_dataset = FinetuneDatasetWithTemplate(
            self.train_dataset, self.dataset_reader.get_train_template(), self.tokenizer, **extra_args
        )
        self.dev_dataset = FinetuneDatasetWithTemplate(
            self.dev_dataset, self.dataset_reader.get_eval_template(), self.tokenizer, **extra_args
        )
        print(f"Train size {len(self.train_dataset)}")
        print(f"Eval size {len(self.dev_dataset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            drop_last=True, # Weird that they're dropping the last?
            num_workers=min([self.config.batch_size, self.config.num_workers]),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
        )


class FinetuneDatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, templates, tokenizer, add_special_tokens=True):
        super().__init__()
        self.dataset = dataset
        self.templates = templates
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(self.templates, list):
            template = np.random.choice(self.templates)
        else:
            template = self.templates
        example = self.dataset[key]
        input_str, target_str = template.apply(example)
        
        answer_choices = template.get_answer_choices_list(example)
        if isinstance(input_str, list):
            encodings = (encode_and_truncate(input_field, False, self.tokenizer) for input_field in input_str[:-1])
            encodings_input, num_truncated = zip(*encodings)
            encoding_last, num_truncated_last = encode_and_truncate(input_str[-1], self.add_special_tokens, self.tokenizer) 

            input_ids = torch.cat(encodings_input, encoding_last)
            num_truncated = sum(num_truncated) + num_truncated_last
        else:
            input_ids, num_truncated = encode_and_truncate(
                input_str, self.add_special_tokens, self.tokenizer)
        
        # We assume the targets are usually small and therefore we will not
        # consider they truncation.
        target_ids, _ = encode_and_truncate(
            target_str, add_special_tokens=self.add_special_tokens, tokenizer=self.tokenizer
        )
        answer_choices_ids = [
            encode_and_truncate(answer_choice, self.add_special_tokens, self.tokenizer)[0]
            for answer_choice in answer_choices
        ]
        label = torch.LongTensor([example["label"]])
        idx = torch.LongTensor([example["idx"]])
        return input_ids, target_ids, answer_choices_ids, label, idx, torch.LongTensor([num_truncated])


class PretrainDataModule(LightningDataModule):
    def __init__(self, config, tokenizer, dataset_reader):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_reader = dataset_reader

    def setup(self, stage):
        self.train_datasets = self.dataset_reader.read_orig_dataset("train")
        self.base_templates = self.dataset_reader.get_template()
        self.train_datasets_withtemplate = []
        for index, train_dataset in enumerate(self.train_datasets):
            self.train_datasets_withtemplate.append(
                PretrainDatasetWithTemplate(train_dataset, self.base_templates[index], self.tokenizer)
            )

        self.train_dataset = torch.utils.data.ConcatDataset(self.train_datasets_withtemplate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=True),
            drop_last=True,
            num_workers=min([self.config.batch_size, self.config.num_workers]),
        )


class PretrainDatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, templates, tokenizer):
        super().__init__()
        self.dataset = dataset
        self.templates = templates
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(self.templates, list):
            template = np.random.choice(self.templates)
        else:
            template = self.templates
        example = self.dataset[key]
        input_target_str = template.apply(example)
        if len(input_target_str) == 2:
            input_str, target_str = input_target_str
            if target_str == "":
                target_str = "<NO LABEL>"
        else:
            input_str = "<NO INPUT>"
            target_str = "<NO LABEL>"
        input_ids = self.tokenizer(input_str, return_tensors="pt", truncation=True).input_ids.squeeze(0)
        target_ids = self.tokenizer(target_str, return_tensors="pt", truncation=True).input_ids.squeeze(0)
        return input_ids, target_ids


def create_collate_fn(pad_token_id, pretrain):
    def collate_fn(batch):
        if not pretrain:
            input_ids, target_ids, answer_choices_ids, labels, idx, num_truncated = zip(*batch)
        else:
            input_ids, target_ids = zip(*batch)

        # Update 2022-08-14 @pastelbelem8
        # ---------------------------------------------------------------------
        # In an attempt to know exactly how many sequences are being truncated
        # and how its related with the model's performance.
        # ---------------------------------------------------------------------
        seq_lens = [len(b[0]) for b in batch]
        # ---------------------------------------------------------------------
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=pad_token_id)
        output_batch = {
            "input_seq_len": seq_lens,
            "input_ids": input_ids,
            "target_ids": target_ids,
        }

        if not pretrain:
            flat_answer_choice_ids = [choice for list_choices in answer_choices_ids for choice in list_choices]
            num_choice = [len(list_choices) for list_choices in answer_choices_ids]
            if max(num_choice) != min(num_choice):
                raise NotImplementedError("The collate_fn is not implmented for variable number of choices")
            flat_answer_choices_ids = torch.nn.utils.rnn.pad_sequence(
                flat_answer_choice_ids, batch_first=True, padding_value=pad_token_id
            )
            answer_choices_ids = flat_answer_choices_ids.view(len(answer_choices_ids), max(num_choice), -1).contiguous()
            labels = torch.cat(labels)
            idx = torch.cat(idx)
            output_batch.update(
                {
                    "answer_choices_ids": answer_choices_ids,
                    "labels": labels,
                    "idx": idx,
                    "num_truncated": num_truncated,
                }
            )
        return output_batch

    return collate_fn



class EvalDataModule(LightningDataModule):
    def __init__(self, config, tokenizer, dataset_reader, test_only=False):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_reader = dataset_reader
        self.test_only = test_only

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        if self.config.few_shot:
            _ = self.dataset_reader.read_few_shot_dataset()

    def setup(self, stage):
        self.test_dataset = self.dataset_reader.read_orig_dataset("test") # Dataset

        extra_args = {"add_special_tokens": self.config.add_special_tokens}
        
        self.test_dataset = FinetuneDatasetWithTemplate(
            self.test_dataset, self.dataset_reader.get_eval_template(), self.tokenizer, **extra_args
        )
        print(f"Test size {len(self.test_dataset)}")

        if not self.test_only:
            self.dev_dataset = self.dataset_reader.read_orig_dataset("validation") # Dataset
            self.dev_dataset = FinetuneDatasetWithTemplate(
                self.dev_dataset, self.dataset_reader.get_eval_template(), self.tokenizer, **extra_args
            )
            print(f"Dev size {len(self.dev_dataset)}")
        
    def val_dataloader(self):
        if not self.test_only:
            return torch.utils.data.DataLoader(
                self.dev_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False,
                collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
                num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
            )
        return None

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
        )
