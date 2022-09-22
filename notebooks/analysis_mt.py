from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, kendalltau
from glob import glob

import pandas as pd 
import numpy as np
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns

def filter_by_ids(data: pd.DataFrame, ids_list: list):
    return data[data["bartscore_doc_id"].isin(ids_list)]


def _get_system_level_correlation(data, metrics, target_col, systems, corr_method: callable) -> dict:
    # pseudo code
    # for each system
    # compute the mean score attributed by a metric m to the outputs of each system.
    # compute the mean score attributed by a target_col to the outputs of each system.
    # compute correlation
    system_level_correlation = defaultdict(list)
    for sys in systems:
        data_sys = data[data["sys_name"] == sys]
        # ^Note: since we're computing the mean, we dont need to ensure the ordering

        for m in metrics + [target_col]:
            mean_sys = data_sys[m].mean()
            system_level_correlation[m].append(mean_sys)

    # Compute the correlation now
    correlations = {}
    for m in metrics:
        corr, p_val = corr_method(system_level_correlation[m], system_level_correlation[target_col])

        correlations[m] = round(corr, 4)

    return correlations


def compute_system_level_correlations(data, metrics, target_col, dataset_name, systems, output_dir, to_persist=True, **_):

    correlations = {}
    for corr_method in (pearsonr, spearmanr, kendalltau):
        result = _get_system_level_correlation(data, metrics, target_col, systems, corr_method)
        correlations[corr_method.__name__] = result

    correlations = pd.DataFrame(correlations)
    if to_persist:
        os.makedirs(output_dir, exist_ok=True)
        correlations.reset_index().to_csv(f"{output_dir}/{dataset_name}_system_corrs.csv", index=0)
    
    return correlations


def get_t0_baselines(dataset_name: str, dataset_classes: str, parent_dir: str, output_dir="./t-few-master/exp_out", **_kwargs):
    def extract_name(path: str) -> str:
        # Given a path in the format
        # './t-few-master/exp_out/realsumm/t03b_realsumm_baseline_ft_train/<something>
        # 1. Extract the parent dir `t03b_realsumm_baseline_ft_train`
        exp_name = path.split("/")[-2]
        
        # 2. Keep all parts including baseline and afterwards
        index_baseline = exp_name.index("baseline")
        return exp_name[index_baseline:]
        
    
    output_dir = f"{output_dir}/{dataset_name.lower()}/{dataset_classes}"
    t0_baselines = "t03b_*_baseline*"
    
    dev_files = glob(os.path.join(output_dir, t0_baselines, "dev_pred.txt"))
    print(f"Find {len(dev_files)} experiments fit into {t0_baselines}")
    print("\t -->", "\n\t --> ".join(dev_files))
    print()
        
    test_files = glob(os.path.join(output_dir, t0_baselines, "test_pred.txt"))
    print(f"Find {len(test_files)} experiments fit into {t0_baselines}")
    print("\t -->", "\n\t --> ".join(test_files))
    print()
    
    # The result of the files will be:
    # ['./t-few-master/exp_out/realsumm/t03b_realsumm_baseline/dev_pred.txt',
    #  './t-few-master/exp_out/realsumm/t03b_realsumm_baseline_ft_train/dev_pred.txt',
    # './t-few-master/exp_out/realsumm/t03b_realsumm_baseline_ft_train_20k_steps/dev_pred.txt', 
    # './t-few-master/exp_out/realsumm/t03b_realsumm_baseline_ft_all/dev_pred.txt',
    # './t-few-master/exp_out/realsumm/t03b_realsumm_baseline_ft_all_20k_steps/dev_pred.txt'
    # ]
    # As such, we will extract the name of the baseline using the `extract_name` method, 
    # e.g., `baseline`, `baseline_ft_train`, `baseline_ft_train_20k_steps`, ...
    dev_data = {extract_name(path): pd.read_csv(path) for path in dev_files}
    test_data = {extract_name(path): pd.read_csv(path) for path in test_files}

    # Before returning the dataframe, we will recover the bartscore_doc_id
    # to facilitate re-use of previous correlation methods.
    dev_original_df = pd.read_csv(f"{parent_dir}/{dataset_name}/{dataset_classes}_dev.csv")[["index", "sys_name", "bartscore_doc_id"]]
    test_original_df = pd.read_csv(f"{parent_dir}/{dataset_name}/{dataset_classes}_test.csv")[["index", "sys_name", "bartscore_doc_id"]]
    
    dev_corpora = {}
    for baseline, data in dev_data.items():
        if "ft_all" in baseline: continue
        data = data.merge(dev_original_df, left_on="idx", right_on="index")
        dev_corpora[baseline] = data
    
    test_corpora = {}
    for baseline, data in test_data.items():
        if "ft_all" in baseline: continue
            
        data = data.merge(test_original_df, left_on="idx", right_on="index")
        test_corpora[baseline] = data
    
    print("Baselines (for dev):\n ->", list(dev_corpora.keys()))
    print("Baselines (for test):\n ->", list(test_corpora.keys()))
    return dev_corpora, test_corpora


def compute_split_correlations(
    split, 
    split_baselines, 
    dataset_split, 
    dataset_name, 
    dataset_classes, 
    parent_dir, 
    output_dir, 
    target_col, 
    systems, 
    tag="t0_baselines",
    **_
):
    instance_baseline_corrs = []
    system_baseline_corrs = []
        
    for baseline, data in split_baselines.items():
        print("-" * 80)
        print("Computing correlations for", baseline)
        print("-" * 80)

        d = dataset_split[["bartscore_doc_id", "sys_name", target_col]]
        ds = data.merge(d, on=["sys_name", "bartscore_doc_id", target_col], how="inner")

        # instance_corr = compute_instance_level_correlations(
        #     ds,
        #     metrics=["log.scores_class_1"],
        #     target_col=target_col,
        #     dataset_name=f"{dataset_name.lower()}_{baseline}",
        #     output_dir="",
        #     to_persist=False,
        # )

        system_corr = compute_system_level_correlations(
            ds, 
            metrics=["log.scores_class_1"],
            target_col=target_col,
            dataset_name=f"{dataset_name.lower()}_{baseline}",
            output_dir="",
            systems=systems,
            to_persist=False
        )

      #  instance_corr["index"] = baseline
        system_corr["index"] = baseline

     #   instance_baseline_corrs.append(instance_corr)
        system_baseline_corrs.append(system_corr)
        
    #instance_baseline_corrs = pd.concat(instance_baseline_corrs)
    system_baseline_corrs = pd.concat(system_baseline_corrs)

    dataset_name = f"{dataset_name}_{dataset_classes}_{tag}_{split}"
    # instance_baseline_corrs.reset_index().to_csv(f"{output_dir}/{dataset_name}_inst_corrs.csv", index=0)
    system_baseline_corrs.reset_index().to_csv(f"{output_dir}/{dataset_name}_system_corrs.csv", index=0)
    
    # return instance_baseline_corrs, system_baseline_corrs
    return None, system_baseline_corrs


def compute_metrics(df: pd.DataFrame):
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
    from scipy.stats import pearsonr, spearmanr, kendalltau
    
    metrics = {}
    
    kwargs = {"y_true": df['label'], "y_pred": df["prediction"]}
    metrics["precision"] = precision_score(**kwargs)
    metrics["recall"] = recall_score(**kwargs)
    metrics["f1_score"] = f1_score(**kwargs)
    
    tn, fp, fn, tp = confusion_matrix(**kwargs).ravel()
    
    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["tp"] = tp
    
    metrics["lp"] = metrics["tp"] + metrics["fn"]
    metrics["ln"] = metrics["fp"] + metrics["tn"]
    metrics["pp"] = metrics["tp"] + metrics["fp"]
    metrics["pn"] = metrics["tn"] + metrics["fn"]
    
    metrics["n"] = len(df)
    metrics["fpr"] = fp / (fp + tn)
    metrics["fnr"] = 1 - metrics["recall"]
    
    metrics["accuracy"] = accuracy_score(**kwargs)
    
    
    return metrics


from glob import glob
from itertools import takewhile

def get_kshot_files(dataset_name: str, dataset_classes: str, parent_dir: str, output_dir="./t-few-master/exp_out", **_kwargs):
    def extract_name(path):
        exp_name = path.split("/")[-2]
        
        model = exp_name.split("_")[0]
        
        shot_idx = exp_name.index("shots")
        shot = next(takewhile(str.isdigit, exp_name[shot_idx + len("shots"):].split("_")))
        shot = int(shot)
        
        seed_idx = exp_name.index("seed")
        seed = next(takewhile(str.isdigit, exp_name[seed_idx + len("seed"):].split("_")))
        seed = int(seed)
        
        return (model, shot, seed)
        
    def add_shots_metadata(datasets):
        for (model, shots, seed), data in datasets.items():
            data["model"] = model
            data["num_shots"] = shots
            data["seed_shots"] = seed

        return datasets
    
    output_dir = f"{output_dir}/{dataset_name.lower()}/{dataset_classes}"
    print(output_dir)
    files = f"t03b_*_shots*"
    
    dev_filepaths = glob(os.path.join(output_dir, files, "dev_pred.txt"))
    print(f"Find {len(dev_filepaths)} experiments fit into {files}")
    print("\t -->", "\n\t --> ".join(dev_filepaths))
    print()
        
    test_filepaths = glob(os.path.join(output_dir, files, "test_pred.txt"))
    print(f"Find {len(test_filepaths)} experiments fit into {files}")
    print("\t -->", "\n\t --> ".join(test_filepaths))
    print()
    
    # The result of the files will be:
    # ['./t-few-master/exp_out/realsumm/t03b_realsumm_2class_seed9284_shots2_ia3_pretrained100k/dev_pred.txt',
    # ]
    # As such, we will extract the name of the baseline using the `extract_name` method, 
    # e.g., `baseline`, `baseline_ft_train`, `baseline_ft_train_20k_steps`, ...
    dev_data = {extract_name(path): pd.read_csv(path) for path in dev_filepaths}
    test_data = {extract_name(path): pd.read_csv(path) for path in test_filepaths}
     
    dev_data = add_shots_metadata(dev_data)
    test_data = add_shots_metadata(test_data)

    # Before returning the dataframe, we will recover the bartscore_doc_id
    # to facilitate re-use of previous correlation methods.
    dev_original_df = pd.read_csv(f"{parent_dir}/{dataset_name}/{dataset_classes}_dev.csv")[["index", "sys_name", "bartscore_doc_id", "human_score"]]
    test_original_df = pd.read_csv(f"{parent_dir}/{dataset_name}/{dataset_classes}_test.csv")[["index", "sys_name", "bartscore_doc_id", "human_score"]]
    
    print("\n\n")
    print(dev_original_df.columns)
    
    dev_corpora = {}
    for (model, shots, seed), data in dev_data.items():
        data = data.merge(dev_original_df, left_on="idx", right_on="index")
        baseline = f"{model}_shots{shots}_seed{seed}"

        dev_corpora[baseline] = data
    
    test_corpora = {}
    for (model, shots, seed), data in test_data.items():
        data = data.merge(test_original_df, left_on="idx", right_on="index")
        
        baseline = f"{model}_shots{shots}_seed{seed}"
        test_corpora[baseline] = data
        
    print("Baselines (for dev):\n ->", list(dev_corpora.keys()))
    print("Baselines (for test):\n ->", list(test_corpora.keys()))
    return dev_corpora, test_corpora


if __name__ == "__main__":
    SOURCE_LANG = "de"
    WMT_PREPROC_DIR = f"./experiments/mt_data/preproc/{SOURCE_LANG}-en"
    DATASET_CLASSES = "2class"
    TARGET_COL = "raw_score"
    METRICS = [
    'bleu',
    'chrf',
    'bleurt', 'prism', 'comet', 'bert_score', 'bart_score',
    'bart_score_cnn', 'bart_score_para', 'bart_score_para_en_Such as',
    'bart_score_para_de_Such as'
    ]

    # Load the scores file
    WMT = pd.read_csv(f"{WMT_PREPROC_DIR}/WMT19DARR_w_all_scores.csv")
    WMT["human_score"] = WMT[TARGET_COL].apply(lambda ex: round(ex/100, 4))
    WMT.head()

    # Traceback the dataset to the evaluation splits used for evaluation
    wmt_eval_splits = {}
    for split in ("dev", "test"):
        filepath = f"{WMT_PREPROC_DIR}/{DATASET_CLASSES}_{split}.csv"
        _split = pd.read_csv(filepath)

        ids = sorted(_split["bartscore_doc_id"].unique())
        wmt_eval_splits[split] = ids
        
    WMT_DEV_SCORES = filter_by_ids(WMT, wmt_eval_splits["dev"])
    WMT_TEST_SCORES = filter_by_ids(WMT, wmt_eval_splits["test"])
    WMT_DEV_SCORES.head()

    BASE_WMT_DATASET_NAME = f"{SOURCE_LANG}_{DATASET_CLASSES}"

    metadata = {
        "metrics": METRICS,
        "target_col": "human_score",
        "output_dir": "./results/mt",
        "to_persist": True,
        "systems": sorted(WMT["sys_name"].unique()),
        "dataset_classes": DATASET_CLASSES,
        "parent_dir": "./experiments/mt_data/preproc",
    }
    metadata["dataset_name"] = f"{BASE_WMT_DATASET_NAME}_dev"
    BASE_WMT_DATASET_NAME = f"{SOURCE_LANG}_{DATASET_CLASSES}"

    # In some cases, due to the averaging, we dont get enough information to compute the correlation scores.
    # compute_instance_level_correlations(data=WMT_DEV_SCORES, **metadata)
    compute_system_level_correlations(data=WMT_DEV_SCORES, **metadata)

    dev_shots, test_shots = get_kshot_files(
        f"{SOURCE_LANG}-en",
        "2class", 
        f"./experiments/mt_data/preproc",
        output_dir="./t-few-master/exp_out_2022_sep_10"
    )

    print("=" * 80)
    print(" " *30,"DEV SET", " "*30)
    print("=" * 80)
    compute_split_correlations(
        split="dev",
        split_baselines=dev_shots,
        dataset_split=WMT_DEV_SCORES, 
        tag = "kshots",
        **metadata
    )