from abc import ABC, abstractmethod
from collections import defaultdict
import os
import json
import numpy as np
from datasets import load_dataset, load_from_disk
from promptsource.templates import DatasetTemplates
import pkg_resources
from promptsource import templates
import csv
from typing import Dict, List, Optional, Tuple
import re
import pandas as pd
from .custom_templates import SemanticCovTemplates, AdequacyTemplates, T5Templates
from .sampling import BalancedSampler


def get_dataset_reader(config):
    dataset_class = {
        "T0Mixture": T0MixtureReader,
        "rte": RTEReader,
        "h-swag": HSwagReader,
        "copa": COPAReader,
        "wic": WiCReader,
        "winogrande": WinograndeReader,
        "cb": CBReader,
        "storycloze": StoryClozeReader,
        "anli-r1": ANLIR1Reader,
        "anli-r2": ANLIR2Reader,
        "anli-r3": ANLIR3Reader,
        "wsc": WSCFixedReader,
        "ade_corpus_v2": RaftReader,
        "banking_77": RaftReader,
        "terms_of_service": RaftReader,
        "tai_safety_research": RaftReader,
        "neurips_impact_statement_risks": RaftReader,
        "overruling": RaftReader,
        "systematic_review_inclusion": RaftReader,
        "one_stop_english": RaftReader,
        "tweet_eval_hate": RaftReader,
        "twitter_complaints": RaftReader,
        "semiconductor_org_types": RaftReader,
    }

    # 2022-09-12 Update
    if config.use_regress:
        dataset_class.update({
            "REALSumm": REALSummRegressionReader,
            "WMT_de_en": WMTRegressionReader,
            "WMT_zh_en": WMTRegressionReader,
            "WMT_fi_en": WMTRegressionReader,
            "WMT_ru_en": WMTRegressionReader,
            "WMT_kk_en": WMTRegressionReader,
            "WMT_gu_en": WMTRegressionReader,
            "WMT_lt_en": WMTRegressionReader,
        })
    else:
        dataset_class.update({
        # 2022-09-05 Update
        "REALSumm": REALSummClassificationReader,
        "WMT_de_en": WMTClassificationReader,
        "WMT_zh_en": WMTClassificationReader,
        "WMT_fi_en": WMTClassificationReader,
        "WMT_ru_en": WMTClassificationReader,
        "WMT_kk_en": WMTClassificationReader,
        "WMT_gu_en": WMTClassificationReader,
        "WMT_lt_en": WMTClassificationReader,
    })

    dataset_class = dataset_class[config.dataset]
    return dataset_class(config)


DATASETS_OFFLINE = "/fruitbasket/datasets/datasets_offline"
MAX_EXAMPLES_PER_DATASET = 500_000
TASK_BLACKLIST = [
    # Tasks which often tokenize to > 1024 tokens currently
    "hotpot_qa_distractor_Generate_Explanations",
    "hotpot_qa_fullwiki_Generate_Explanations",
    "hotpot_qa_distractor_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer",
    "hotpot_qa_distractor_Generate_Answer",
    "hotpot_qa_distractor_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Question",
    "hotpot_qa_fullwiki_Generate_Question",
    "tab_fact_tab_fact_tab_fact_3",
    "tab_fact_tab_fact_tab_fact_2",
    "tab_fact_tab_fact_tab_fact_1",
    "tab_fact_tab_fact_tab_fact_7",
    "tab_fact_tab_fact_tab_fact_4",
    "tab_fact_tab_fact_tab_fact_5",
    "tab_fact_tab_fact_tab_fact_6",
    "wiki_hop_masked_Choose_Best_Object_Candidate",
    "wiki_hop_masked_Indirect_Question_about_Birthplace_Citizenship_Place_of_Death",
    "narrativeqa_Template_05",
    "ecthr_cases_alleged_violation_prediction_silver_rationales",
    # "amazon_polarity/amazon_polarity",
    # "quail_context_question_answer_description_id",
    # "quail_context_question_description_answer_text",
    # "quail_context_question_answer_description_text",
    # "quail_context_question_description_answer_id",
    # "quail_context_question_answer_description_id",
    # "quail_context_question_description_answer_text",
    # "quail_context_question_answer_description_text",
    # "quail_context_question_description_answer_id",
    # "quail_description_context_question_text",
    # "quail_description_context_question_answer_text",
    # 'quail_context_description_question_answer_id',
    # 'quail_context_description_question_answer_text',
    # 'quail_context_description_question_text',
    # 'quail_context_question_answer_description_text',
    # 'quail_context_question_description_answer_id',
    # 'quail_context_question_description_text',
    # 'quail_description_context_question_answer_id',
    # 'quail_description_context_question_answer_text',
    # 'quail_description_context_question_text',
    # 'quail_no_prompt_id',
    # 'quail_no_prompt_text',
    # Tasks with broken cached files
    "gigaword_summarize_",
]


class BaseDatasetReader(object):
    """
    DatasetReader is responsible for reading and processing dataset
    """

    def __init__(self, config, dataset_stash):
        """
        :param config:
        """
        self.config = config
        self.dataset_stash = dataset_stash

        self.templates = DatasetTemplates(*self.dataset_stash)
        self.train_template = self.get_template(self.config.train_template_idx)
        self.eval_template = self.get_template(self.config.eval_template_idx)

    def get_template(self, template_idx):
        template_names = self.templates.all_template_names
        if template_idx >= 0:
            return self.templates[template_names[template_idx]]
        elif template_idx == -1:

            list_idx = []
            list_templates = []
            for idx, template_name in enumerate(template_names):
                if self.templates[template_name].metadata.original_task:
                    list_idx.append(idx)
                    list_templates.append(self.templates[template_name])
            print(list_idx)

            return list_templates
        elif template_idx == -2:
            return [self.templates[template_name] for template_name in template_names]

    def get_train_template(self):
        return self.train_template

    def get_eval_template(self):
        return self.eval_template

    def read_orig_dataset(self, split):
        """
        Read the original dataset

        :param split: split of data
        """
        if os.path.exists(DATASETS_OFFLINE):
            orig_data = load_from_disk(os.path.join(DATASETS_OFFLINE, *self.dataset_stash))[split]
        else:
            orig_data = load_dataset(*self.dataset_stash, split=split, cache_dir=os.environ["HF_HOME"])
        return orig_data

    def read_few_shot_dataset(self):
        file_dir = os.path.join("data", "few_shot", self.config.dataset, f"{self.config.num_shot}_shot")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_path = os.path.join(file_dir, f"{self.config.few_shot_random_seed}_seed.jsonl")

        if os.path.exists(file_path):
            with open(file_path, "r") as fin:
                data = []
                for idx, line in enumerate(fin.readlines()):
                    data.append(json.loads(line.strip("\n")))

            return data
        else:
            orig_data = self.read_orig_dataset("train")
            selected_data = self._sample_few_shot_data(orig_data)

            with open(file_path, "w+") as fout:
                for example in selected_data:
                    fout.write(json.dumps(example) + "\n")
            return selected_data

    def _sample_few_shot_data(self, orig_data):
        saved_random_state = np.random.get_state()
        np.random.seed(self.config.few_shot_random_seed)
        orig_data = [x for x in orig_data]
        np.random.shuffle(orig_data)
        selected_data = orig_data[: self.config.num_shot]
        np.random.set_state(saved_random_state)
        return selected_data

    def compute_metric(self, accumulated, *args, **kwargs):
        matching = [a == b for a, b in zip(accumulated["prediction"], accumulated["label"])]
        accuracy = sum(matching) / len(matching)
        return {"accuracy": accuracy}


class StoryClozeReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("story_cloze", "2016"))

    def read_orig_dataset(self, split):
        if split == "train":
            split = "validation"
        elif split == "validation":
            split = "test"

        if os.path.exists(DATASETS_OFFLINE):
            orig_data = load_from_disk(os.path.join(DATASETS_OFFLINE, *self.dataset_stash))[split]
        else:
            orig_data = load_dataset(
                *self.dataset_stash, split=split, data_dir="/fruitbasket/datasets/hugging_face/story_cloze"
            )
        orig_data = [example for example in orig_data]
        for idx, example in enumerate(orig_data):
            example["label"] = example["answer_right_ending"] - 1
            example["idx"] = idx
        return orig_data


class ANLIR1Reader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("anli",))

    def read_orig_dataset(self, split):
        if split == "validation":
            split = "test"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r1")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data


class ANLIR2Reader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("anli",))

    def read_orig_dataset(self, split):
        if split == "validation":
            split = "test"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r2")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data


class ANLIR3Reader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("anli",))

    def read_orig_dataset(self, split):
        if split == "validation":
            split = "test"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r3")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data


class WSCFixedReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "wsc.fixed"))


class RTEReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "rte"))


class HSwagReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("hellaswag",))
        if config.change_hswag_templates:
            from promptsource.templates import Template

            name_jinja = [
                ("basic", "{{ctx}}|||{{endings [label | int()]}}"),
                (
                    "prompt 1",
                    "Can you pick the correct ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
                ),
                (
                    "prompt 2",
                    "The task is to generate the ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
                ),
                ("prompt 3", "How does this sentence end? {{ctx}}|||{{answer_choices [label | int()]}}"),
                (
                    "prompt 4",
                    "From the list of endings described below, what ending makes the most sense for the sentence {{ctx}}|||{{answer_choices [label | int()]}}",
                ),
                (
                    "ctx a,b",
                    "Complete the description with an appropriate ending:\n First, {{ ctx_a.lower() }} Then, {{ ctx_b.lower() }} ...|||{{answer_choices [label | int()]}}",
                ),
                (
                    "middle",
                    "If a description of a situation begins like this: {{ ctx }}... Then how does it continue?|||{{answer_choices [label | int()]}}",
                ),
            ]

            self.templates = []
            for name, jinja in name_jinja:
                self.templates.append(
                    Template(name=name, jinja=jinja, reference="", answer_choices='{{endings | join("|||")}}')
                )

            if self.config.train_template_idx >= 0:
                self.train_template = self.templates[self.config.train_template_idx]
            else:
                self.train_template = self.templates
            if self.config.eval_template_idx >= 0:
                self.eval_template = self.templates[self.config.eval_template_idx]
            else:
                self.eval_template = self.templates

    def read_orig_dataset(self, split):
        orig_data = [example for example in super().read_orig_dataset(split)]
        for idx, example in enumerate(orig_data):
            example["label"] = int(example["label"])
            example["idx"] = idx
        return orig_data


class WiCReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "wic"))


class COPAReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "copa"))

    def get_template(self, template_idx):
        if template_idx >= 0:
            return super().get_template(template_idx)
        else:
            return super().get_template(template_idx)[:8]


class WinograndeReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("winogrande", "winogrande_xl"))

    def read_orig_dataset(self, split):
        orig_data = [example for example in super().read_orig_dataset(split)]
        for idx, example in enumerate(orig_data):
            example["label"] = int(example["answer"]) - 1
            example["idx"] = idx
        return orig_data


class CBReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "cb"))


class T0MixtureReader(object):
    """
    DatasetReader is responsible for reading and processing dataset
    """

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        datatset_subset_tuple = Tuple[str, Optional[str]]
        t0_train: Dict[str, List[datatset_subset_tuple]] = {
            "BASE": [],
            # GPT3 evaluation set
            "GPT_EVAL": [],
            # SuperGLUE (except RTE and CB)
            "SGLUE": [],
        }
        t0_eval: Dict[str, List[datatset_subset_tuple]] = {"BASE": [], "BIAS_FAIRNESS": []}
        gsheet: Dict[datatset_subset_tuple, Dict] = {}
        experiment_path = pkg_resources.resource_filename(__name__, "datasets.csv")

        with open(experiment_path) as exp_file:
            reader = csv.DictReader(exp_file)
            for row in reader:
                if row["subset"] == "":
                    row["subset"] = None  # to match promptsource.Template object
                dataset_subset = (row["HF_name"], row["subset"])
                if row["do_train"] != "":
                    do_train_source = row["do_train"]
                    # sanity checks
                    if do_train_source == "SGLUE":
                        assert dataset_subset[0] == "super_glue"
                    t0_train[do_train_source].append(dataset_subset)
                if row["do_eval"] != "":
                    do_eval_source = row["do_eval"]
                    # sanity checks
                    if do_eval_source == "BIAS_FAIRNESS":
                        assert row["task_by_convention"] == "bias_and_fairness"
                    t0_eval[do_eval_source].append(dataset_subset)
                gsheet[dataset_subset] = row

        all_datasets = sum(t0_train.values(), []) + sum(t0_eval.values(), [])
        all_templates = templates.TemplateCollection()
        all_templates.remove("anli")

        # 3 stages of training/ablation: D4 -> GPT -> SuperGLUE
        t0_train_mixture: Dict[str, List[str]] = {key: [] for key in t0_train}
        t0_eval_mixture: Dict[str, List[str]] = {key: [] for key in t0_eval}
        mixture_cap: Dict[str, int] = {}
        single_original_task: Dict[Tuple[str, str], str] = {}
        all_original_tasks: List[str] = []
        added_tasks: List[Tuple[str, str, str]] = []

        def get_task_name(dataset_name, subset_name, template_name):
            # Clean the text according to allowed characters for a task name
            task_name = dataset_name + (f"_{subset_name}_" if subset_name is not None else "_") + template_name
            return re.sub(r"[^\w\d\._]+", "_", task_name)

        for dataset_name, subset_name in all_templates.keys:

            if (dataset_name, subset_name) not in all_datasets:
                all_templates.remove(dataset_name, subset_name)
                continue
            dataset = all_templates.get_dataset(dataset_name, subset_name)
            num_templates = len(dataset.all_template_names)
            train_size = gsheet[(dataset_name, subset_name)]["train_size"]
            if train_size == "":
                train_size = 0
            else:
                train_size = int(train_size)
            if train_size > MAX_EXAMPLES_PER_DATASET // num_templates:
                cap = MAX_EXAMPLES_PER_DATASET // num_templates
            else:
                cap = train_size
            for template_name in dataset.all_template_names:
                added_tasks.append((dataset_name, subset_name, template_name))

                template = dataset[template_name]

                task_name = get_task_name(dataset_name, subset_name, template_name)

                if (dataset_name, subset_name) not in single_original_task and template.metadata.original_task:
                    single_original_task[(dataset_name, subset_name)] = task_name

                if template.metadata.original_task:
                    all_original_tasks.append(task_name)

                # Check that the dataset_subset_tuple is in t0_train
                for key, dataset_subset_tuples in t0_train.items():
                    if (dataset_name, subset_name) in dataset_subset_tuples:
                        t0_train_mixture[key].append(task_name)
                        mixture_cap[task_name] = cap

                # Check that the dataset_subset_tuple is in t0_eval
                if (dataset_name, subset_name) in t0_eval["BASE"]:
                    if template.metadata.original_task:
                        t0_eval_mixture["BASE"].append(task_name)
                    # TODO use template.metadata.answer_choices here for rank eval
                if (dataset_name, subset_name) in t0_eval["BIAS_FAIRNESS"]:
                    t0_eval_mixture["BIAS_FAIRNESS"].append(task_name)

        self.t0_base_tasks = []
        self.t0_base_templates = []
        for (dataset_name, subset_name, template_name) in added_tasks:
            task_name = get_task_name(dataset_name, subset_name, template_name)
            if task_name in t0_train_mixture["BASE"]:
                if task_name not in TASK_BLACKLIST:
                    self.t0_base_tasks.append((dataset_name, subset_name, template_name, mixture_cap[task_name]))
                    template = all_templates.get_dataset(dataset_name, subset_name)[template_name]
                    self.t0_base_templates.append(template)

    def get_template(self):
        return self.t0_base_templates

    def read_orig_dataset(self, split):
        """
        Read the original dataset

        :param split: split of data
        """
        orig_data = []
        for (dataset_name, subset_name, template_name, cap) in self.t0_base_tasks:
            if split == "train":
                split_num = f"{split}[0:{cap}]"
            else:
                split_num = split

            orig_data.append(load_dataset(dataset_name, subset_name, split=split_num))
        return orig_data


class RaftTemplate(object):
    def __init__(self, config, answer_choices):
        with open(os.path.join(os.path.dirname(__file__), "raft_prompt_construction_settings.jsonl")) as f:
            data = [json.loads(line) for line in f]
            FIELD_ORDERING = data[0]
            INSTRUCTIONS = data[1]
        self.dataset_name = config.dataset
        self.answer_choices = answer_choices
        self.instruction = INSTRUCTIONS[self.dataset_name]
        self.fields = FIELD_ORDERING[self.dataset_name]
        self.raft_labels_in_input_string = config.raft_labels_in_input_string

    def apply(self, example):
        if self.raft_labels_in_input_string == "comma":
            input_str = [
                self.instruction.strip()
                + " Possible labels: "
                + ", ".join([choice for index, choice in enumerate(self.answer_choices)])
            ]
        elif self.raft_labels_in_input_string == "newline":
            input_str = [
                self.instruction.strip()
                + "\nPossible labels:\n"
                + "\n".join([str(index + 1) + ". " + choice for index, choice in enumerate(self.answer_choices)])
            ]
        else:
            input_str = [self.instruction.strip()]

        for key in example:
            if key in self.fields:
                if example[key].strip() != "":
                    input_str.append(str(key) + ": " + example[key].strip())

        if example["label"] == -1:
            target_str = "Unlabeled"
        else:
            target_str = self.answer_choices[example["label"]]
        input_str[-1] += "\nLabel:"
        return input_str, target_str

    def get_answer_choices_list(self, example):
        return self.answer_choices


class RaftReader(object):
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.dataset
        self.orig_data = load_dataset("ought/raft", name=self.dataset_name)
        self.answer_choices = self.orig_data["train"].features["Label"].names[1:]
        if self.config.dataset == "banking_77" and config.cleaned_answer_choices_b77:
            self.answer_choices = [answer.replace("_", " ").replace(". ", " ") for answer in self.answer_choices]

        self.template = RaftTemplate(config, self.answer_choices)

    def get_train_template(self):
        return self.template

    def get_eval_template(self):
        return self.template

    def read_orig_dataset(self, split):
        """
        Read the original dataset

        :param split: split of data
        """
        if self.config.raft_cross_validation:
            orig_data = [example for example in self.orig_data["train"]]
            if split == "train":
                orig_data = (
                    orig_data[: self.config.raft_validation_start] + orig_data[self.config.raft_validation_start + 10 :]
                )
                assert len(orig_data) == 40
            elif split == "validation":
                orig_data = orig_data[self.config.raft_validation_start : self.config.raft_validation_start + 10]
                assert len(orig_data) == 10
        else:
            if split == "validation":
                split = "test"
            orig_data = [example for example in self.orig_data[split]]
        for i, example in enumerate(orig_data):
            # if self.dataset_name in ['ade_corpus_v2', 'terms_of_service','overruling']:
            #     example['input'] = example['Sentence'].strip()
            # elif self.dataset_name in ['banking_77']:
            #     example['input'] = example['Query'].strip()
            # elif self.dataset_name in ['tai_safety_research']:
            #     example['input'] = 'Title : ' + example['Title'].strip() + ' ' + \
            #         'Abstract Note : ' + example['Abstract Note'].strip() + ' '+ \
            #             'Url : ' + example['Url'].strip() + ' ' + \
            #                 'Publication Year : ' + example['Publication Year'].strip() + ' '+ \
            #                     'Item Type : ' + example['Item Type'].strip() + ' ' + \
            #                         'Author : ' + example['Author'].strip() + ' '+ \
            #                             'Publication Title : '  + example['Publication Title'].strip()
            # elif self.dataset_name in ['neurips_impact_statement_risks']:
            #     example['input'] = 'Paper title : ' + example['Paper title'].strip() + ' ' + \
            #         'Paper link : ' + example['Paper link'].strip() + ' ' + \
            #             'Impact statement : ' + example['Impact statement'].strip()
            # elif self.dataset_name in ['systematic_review_inclusion']:
            #     example['input'] = 'Title : ' + example['Title'].strip() + ' ' + \
            #         'Abstract : ' + example['Abstract'].strip() + ' ' + \
            #             'Authors : ' + example['Authors'].strip() + ' ' + \
            #                 'Journal : ' + example['Journal'].strip()
            # elif self.dataset_name in ['one_stop_english']:
            #     example['input'] = example['Article'].strip()
            # elif self.dataset_name in ['tweet_eval_hate']:
            #     example['input'] = example['Tweet'].strip()
            # elif self.dataset_name in ['twitter_complaints']:
            #     example['input'] = example['Tweet text'].strip()
            # elif self.dataset_name in ['semiconductor_org_types']:
            #     example['input'] = 'Paper title : ' + example['Paper title'].strip() + \
            #         'Organization name : ' + example['Organization name'].strip()
            example["label"] = int(example["Label"]) - 1
            example["idx"] = example["ID"]
        return orig_data

    def compute_metric(self, accumulated, *args, **kwargs):
        data = []
        idxs = accumulated["idx"]
        predictions = accumulated["prediction"]
        for idx, prediction in zip(idxs, predictions):
            data.append({"ID": idx, "Label": self.answer_choices[prediction]})
        result_df = pd.DataFrame(data=data, columns=["ID", "Label"]).astype({"ID": int, "Label": str})
        result_df.to_csv(self.config.dev_pred_file, index=False)
        matching = [a == b for a, b in zip(accumulated["prediction"], accumulated["label"])]
        accuracy = sum(matching) / len(matching)
        return {"accuracy": accuracy}


# -----------------------------------------------------------------------------
# 2022-09-05 Update
# -----------------------------------------------------------------------------
class CustomBaseReader(ABC):

    def __init__(self, config):
        self.config = config
        self.is_regression = config.use_regress
        # -------------------------------------------------------
        # Dataset name will be one of the following:
        # -------------------------------------------------------
        # - 2class: handled as 2 classes
        #   (where labels are Yes/No)
        # 
        # - 5class: handled as 5 classes 
        #   (where labels are Never/No/Maybe/Yes/Definitely)
        # 
        # - regression: handled in a token-wise fashion
        # -------------------------------------------------------
        self.dataset_name = config.dataset
        self.setting = "regression" if self.is_regression else "classification"

        self.label_col = config.label_col or "label"
        self.id_col = config.id_col or "id"
        self.sampling_col = config.sampling_col

        self.data_dir = config.data_dir
        self.suffix_path = config.filepath_suffix
    
    def get_template(self, template_idx):
        template_names = self.templates.all_template_names
        if template_idx >= 0:
            return self.templates[template_names[template_idx]]
        elif template_idx < 0:
            list_templates = []
            for name in template_names:
                list_templates.append(self.templates[name])

            print("Using templates:", template_names)
            return list_templates

    def get_train_template(self):
      return self.get_template(self.config.train_template_idx)

    def get_eval_template(self):
      return self.get_template(self.config.eval_template_idx)

    @abstractmethod
    def get_canonical_filename(self, split: str) -> str:
        raise NotImplemented("Must be overriden by subclasses")

    @abstractmethod
    def compute_metric(self, accumulated: dict, is_dev: bool=True) -> dict:
        raise NotImplemented("Must be overriden by subclasses")

    def read_orig_dataset(self, split: str):
        """
        Read the original dataset

        :param split: split of data
        """
        if os.path.exists(DATASETS_OFFLINE):
            orig_data = load_from_disk(os.path.join(DATASETS_OFFLINE))
        else:
            if self.suffix_path is None:
                files = {split: f"{self.get_canonical_filename(split)}.csv"}
            else:
                files = {split: f"{self.get_canonical_filename(split)}.{self.suffix_path}"}
            orig_data = load_dataset(self.data_dir, data_files=files, cache_dir=os.environ["HF_HOME"])

        def rename_cols(ex):
            return {"label": ex[self.label_col], "idx": ex[self.id_col]}

        data = orig_data[split].map(rename_cols, load_from_cache_file=False)
        return data

    def read_few_shot_dataset(self):
        canonical_name = self.get_canonical_filename()
        # Creates a cache directory to place the dataset with the selected fewshot examples 
        file_dir = os.path.join("data", "few_shot", self.setting, self.config.dataset, canonical_name, f"{self.config.num_shot}_shot")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_path = os.path.join(file_dir, f"{self.config.few_shot_random_seed}_seed.jsonl")

        if os.path.exists(file_path):
            with open(file_path, "r") as fin:
                data = []
                for idx, line in enumerate(fin.readlines()):
                    data.append(json.loads(line.strip("\n")))

            return data
        else:
            orig_data = self.read_orig_dataset("train")
            selected_data = self._sample_few_shot_data(orig_data)

            # Dump the selected files in the dataset dir
            with open(file_path, "w+") as fout:
                for example in selected_data:
                    fout.write(json.dumps(example) + "\n")
            return selected_data

    def _sample_few_shot_data(self, orig_data):
        """Samples balanced few-shot examples, specifying ``num_shot=16``
        means we will sample k * len(classes).
        """
        # Select balanced number of examples
        loader = BalancedSampler(orig_data[self.sampling_col])
        selected_ids = loader.get(
            self.config.num_shot,
            strict=False,
            seed=self.config.few_shot_random_seed,
        )
      
        selected_data = [orig_data[id] for id in selected_ids]
        return selected_data


class CustomClassificationReader(CustomBaseReader):
    CLASSES_2_TEXT = {
        2: ["No", "Yes"],
        5: ["Never", "No", "Maybe", "Yes", "Definitely"]
    }

    def __init__(self, config):
        super().__init__(config)

        self.num_classes = config.dataset_classes
        self.answer_choices = self.CLASSES_2_TEXT.get(self.num_classes)
        if not self.answer_choices:
            raise ValueError(f"Unsupported number of classes: {self.num_classes}")
        
        # Align batch_size and num_shot
        self.config.batch_size = min(self.config.batch_size, self.config.num_shot * len(self.answer_choices))

    def get_canonical_filename(self, split: str=None):
        if split == "validation":
            split = "dev"
        return f"{self.num_classes}class_{split}" if split else f"{self.num_classes}class"

    def compute_metric(self, accumulated, is_dev=True):
        def neg(lst):
            return [-e for e in lst]

        data = defaultdict(list)
        for info, values in accumulated.items():
            if info.startswith("log."):
                values = neg(values)
            data[info].extend(values) 

        result_df = pd.DataFrame(data)
        result_df.to_csv(self.config.dev_pred_file if is_dev else self.config.test_pred_file, index=False)

        # Compute metrics
        matching = [a == b for a, b in zip(accumulated["prediction"], accumulated["label"])]
        accuracy = sum(matching) / len(matching)
        metrics = {"accuracy": accuracy}
        
        for i in range(self.num_classes):
            predicted = (result_df["prediction"] == i).sum()
            true = (result_df["label"] == i).sum()

            metrics[f"predicted_{i}"] = float(predicted)
            metrics[f"true_{i}"] = float(true)
            
        metrics["epoch"] = float(result_df.loc[0, "current_epoch"])
        return metrics


class REALSummClassificationReader(CustomClassificationReader):
    def __init__(self, config):
        """
        """
        super().__init__(config)
        self.templates = SemanticCovTemplates(config, self.answer_choices, "ref_summ", "sys_summ")


class WMTClassificationReader(CustomClassificationReader):
    def __init__(self, config):
        super().__init__(config)
        self.templates = AdequacyTemplates(config, self.answer_choices, "mt", "ref")


class CustomRegressionReader(CustomBaseReader):
    def __init__(self, config):
        super().__init__(config)

    def get_canonical_filename(self, split: str=None):
        if split == "validation":
            split = "dev"

        return f"{split}" if split else ""

    def compute_metric(self, accumulated, is_dev=True):
        def neg(lst):
            return [-e for e in lst]

        metrics = {}
        data = defaultdict(list)
        for info, values in accumulated.items():
            if info == "_log.example_scores":
                # log.example_scores is a list of dicts
                values = [{
                    "example_id": v["example_id"],
                    "log.scores": neg(v["log.scores"]),
                    "log.target": neg(v["log.target"]) 
                } for v in values]
            elif info.startswith("log."):
                values = neg(values)
            data[info].extend(values)

        result_df = pd.DataFrame(data)
        result_df.to_csv(self.config.dev_pred_file if is_dev else self.config.test_pred_file, index=False)

        num_correct = 0
        num_digits = 0
        errs_p, errs_t, errs, mae, mse = [], [], [], [], []
        for p, t in zip(accumulated["prediction"], accumulated["label"]):
            # label is already a number
            p = p.strip()

            num_correct += (p == str(t))
            num_digits += p.isdigit()

            if not p.isdigit():
                continue

            p, t = float(p), float(t)
            err = (t - p)
            errs_p.append(p)
            errs_t.append(t)
            errs.append(err)
            mae.append(np.abs(err))
            mse.append(err * err)
    
        # Dump errs for debuging purposes
        if errs != []:
            errs_df = pd.DataFrame({"pred": errs_p, "label": errs_t, "err": errs, "abs_err": mae, "sqr_err": mse})
            errs_df.to_csv(self.config.dev_pred_file + ".errs" if is_dev else self.config.test_pred_file, index=False)
        
        metrics["accuracy"] = num_correct / len(accumulated["prediction"])
        metrics["digits_count"] = num_digits
        metrics["digits_pct"] = num_digits / len(accumulated["prediction"])

        metrics["err_len"] = len(errs)
        metrics["err_avg"] = float(np.mean(errs))
        metrics["mae_avg"] = float(np.mean(mae))
        metrics["mse_avg"] = float(np.mean(mse))
        
        metrics["err_std"] = float(np.std(errs))
        metrics["mae_std"] = float(np.std(mae))
        metrics["mse_std"] = float(np.std(mse))

        metrics["epoch"] = float(result_df.loc[0, "current_epoch"])
        metrics["num_truncated_examples"] = float((result_df["num_truncated"] != 0).sum())
        metrics["num_truncated_tokens_avg"] = float(result_df["num_truncated"].mean())

        return metrics


class REALSummRegressionReader(CustomRegressionReader):
    def __init__(self, config):
        super().__init__(config)

        if "t5" in config.origin_model.lower():
            self.templates = T5Templates(config, None, "ref_summ", "sys_summ")
        else:
            self.templates = SemanticCovTemplates(config, None, "ref_summ", "sys_summ")


class WMTRegressionReader(CustomRegressionReader):
    def __init__(self, config):
        super().__init__(config)
        if "t5" in config.origin_model.lower():
            self.templates = T5Templates(config, None, "mt", "ref")
        else:
            self.templates = AdequacyTemplates(config, None, "mt", "ref")
