import pandas as pd 
import numpy as np
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns


def train_test_split(data_dir, dataset_name, test_fraction, dev_fraction=None, seed=42):
    dev_ids, test_ids = [], []
    data: dict = joblib.load(f"{data_dir}/{dataset_name}_scores.pkl")
    print("Loaded", len(data), "examples.")

    # Select the indices for the holdout set
    rand = np.random.default_rng(seed)
    ids = np.arange(len(data))
    rand.shuffle(ids)
    
    assert 0 < test_fraction <= 1, f"Invalid test_fraction: {test_fraction}"
    test_num = round(test_fraction * len(data))
    
    test_ids = ids[:test_num]    
    print("Collecting", len(test_ids), "examples for the holdout set.")
    
    if dev_fraction is not None:
        assert 0 < dev_fraction <= 1, f"Invalid dev_fraction: {dev_fraction}"
        dev_num = round(dev_fraction * len(data))
    
        dev_ids = ids[test_num:test_num+dev_num]    
        print("Collecting", len(dev_ids), "examples for the dev set.")
    
    train_docs = data
    test_docs = {}
    dev_docs = {}
    
    data_ids = list(data.keys())

    for test_doc_id in test_ids:
        data_id = data_ids[test_doc_id] 
        test_docs[test_doc_id] = train_docs.pop(data_id)
    
    for dev_doc_id in dev_ids:
        data_id = data_ids[dev_doc_id] 
        dev_docs[dev_doc_id] = train_docs.pop(data_id)
        
    print("Dataset", dataset_name, "train-dev-test split:", len(train_docs), len(dev_docs), len(test_docs))
    return train_docs, dev_docs, test_docs


def flatten(dataset: dict) -> list:
    """
    {
        "doc_id": {
            "src": "This is the source text.",
            "ref_summ": "This is the reference summary",
            "sys_summs": {
                "sys_name1": {
                    "sys_summ": "This is the system summary.",
                    "scores": {
                        "human_metric1": 0.3,
                        "human_metric2": 0.5
    }}}}}
    """
    flattened = []
    
    for doc_id, doc_data in dataset.items():
        for sys_name, sys_data in doc_data["sys_summs"].items():
            data = {
                "bartscore_doc_id": doc_id,
                "src": doc_data["src"],
                "ref_summ": doc_data["ref_summ"],
                "sys_name": sys_name,
                "sys_summ": sys_data["sys_summ"],
            }
            
            data.update({col: score for col, score in sys_data["scores"].items()})
            flattened.append(data)
    return flattened


def plot_distributions(data: list, labels: list, figsize=(10, 4), **kwargs):
    n = len(data)
    fig, axes = plt.subplots(1, n, sharex=True, sharey=True, tight_layout=True, figsize=figsize)

    hist_kwargs = {"bins": 10}
    hist_kwargs.update(**kwargs)
    
    for i, (datum, label) in enumerate(zip(data, labels)):
        sns.histplot(ax=axes[i], data=datum, **hist_kwargs)
        axes[i].set_title(label)
    plt.show()
    
    
def get_distribution(data: pd.DataFrame, col: str) -> pd.DataFrame:
    return pd.DataFrame(data[col].value_counts() / len(data)).sort_index().reset_index()


def discretize_dist(data: pd.DataFrame, col: str, binrange: tuple, bins: int, **kwargs) -> pd.DataFrame:
    """Discretizes distributions in terms of equal sized bins."""
    data = data.copy()

    out, bins = pd.cut(binrange, bins=bins, retbins=True) 
    intervals = pd.cut(data[col], bins=bins)
    data["bin"] = intervals
    data["label"] = intervals.cat.codes
    data["discretization_type"] = f"equal-bins-{bins}-{binrange}"

    return data


def discretize_qdist(data: pd.DataFrame, col: str, q: int, precision:int=2, **kwargs) -> pd.DataFrame:
    """Discretizes the distribution in terms of quantile. Less prone to get imbalanced distributions."""
    data = data.copy()
    
    out, bins = pd.qcut(data[col], q=q, retbins=True, precision=precision)
    data["bin"] = out
    data["label"] = out.cat.codes
    data["discretization_type"] = f"{q}-quantile"
    return data


def get_quantile_dists(dfs, col, q, precision):
    results = []
    print("woho")
    data = dfs[0].copy()
    out, bins = pd.qcut(data[col], q=q, retbins=True, precision=precision)
    if bins[0] == 0:
        print(bins)
        bins[0] = -0.01
        print(bins)
    data["bin"] = out
    data["label"] = out.cat.codes
    data["discretization_type"] = f"{q}-quantile"
    results.append(data)
    
    # Apply bins
    for df in dfs[1:]:
        data = df.copy()
        out = pd.cut(data[col], bins=bins, precision=2)
        
        data["bin"] = out
        data["label"] = out.cat.codes
        data["discretization_type"] = f"{q}-quantile"
        results.append(data)

    return results


def create_dataset(dataset_name, col, dev_fraction, test_fraction, seed, raw_dataset_dir, output_dir, scale=None, use_quantiles=True):
    os.makedirs(output_dir, exist_ok=True)
    
    # ------------------------------------------------------
    # Step 1. Create the splits
    # ------------------------------------------------------
    # Train test split
    data = train_test_split(raw_dataset_dir, dataset_name, test_fraction, dev_fraction, seed)

    # Flatten the nested dictionary
    data = [flatten(d) for d in data]

    # Transform into dataframe to be more manageable and visual
    dfs = [pd.DataFrame(d).reset_index() for d in data]
    dfs_names = ["train", "dev", "test"]
    
    # ------------------------------------------------------
    # Step 2. Plot distributions
    # ------------------------------------------------------
    print("=" * 80)
    print(f"Train, dev, test set `{col}` distribution")
    plot_distributions(data=[d[col] for d in dfs], labels=dfs_names)
    
    if scale is not None:
        print("Scale!")
        def min_max_scale(df, min_r, max_r):
            df[col] = df[col].apply(lambda e: (e-min_r)/(max_r-min_r))
            return df
        
        dfs = [min_max_scale(df, *scale) for df in dfs]
        plot_distributions(data=[d[col] for d in dfs], labels=dfs_names)

    # Step 2.1. Plot *binary* distribution
    if use_quantiles:
        dist_fn = get_quantile_dists 
        binary_configs = {"q": 2, "precision": 2}
        fivary_configs = {"q": 5, "precision": 2}
    else:
        def get_discrete(dfs, col, **configs):
            return [discretize_dist(d, col, **configs) for d in dfs]
        dist_fn = get_discrete
        binary_configs = {"binrange": (0, 1), "bins": 2}
        fivary_configs = {"binrange": (0, 1), "bins": 5}
        

    print("-" * 80)
    print("2-class distribution using configs:", binary_configs)
    print("-" * 80)

    dfs_class2 = dist_fn(dfs, col, **binary_configs)
    plot_distributions(data=[d["label"] for d in dfs_class2], labels=dfs_names)

    for d, d_name in zip(dfs_class2, dfs_names):
        if len(d) != 0:
            print(d_name)
            print(get_distribution(d, "bin"))
            d.to_csv(f"{output_dir}/2class_{d_name}.csv", index=False)
    
    pd.concat(dfs_class2).to_csv(f"{output_dir}/2class_all.csv", index=False)

    # Step 2.2. Plot *5-class* distribution
    print("-" * 80)
    print("5-Class distribution using configs:", fivary_configs)
    print("-" * 80)
    
    dfs_class5 = dist_fn(dfs, col, **fivary_configs)
    plot_distributions(data=[d["label"] for d in dfs_class5], labels=dfs_names)

    for d, d_name in zip(dfs_class5, dfs_names):
        if len(d) != 0:
            print(d_name)
            print(get_distribution(d, "bin"))
            d.to_csv(f"{output_dir}/5class_{d_name}.csv", index=False)
            
    pd.concat(dfs_class5).to_csv(f"{output_dir}/5class_all.csv", index=False)
    
    return dfs