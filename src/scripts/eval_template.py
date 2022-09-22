import subprocess
import argparse

dict_dataset_2_template_idx = {
    "copa": list(range(12)),
    "h-swag": [3, 4, 8, 10],
    "storycloze": [0, 1, 3, 4, 5],
    "winogrande": [0, 2, 3, 4, 5],
    "wsc": list(range(10)),
    "wic": list(range(10)),
    "rte": list(range(10)),
    "cb": list(range(15)),
    "anli-r1": list(range(15)),
    "anli-r2": list(range(15)),
    "anli-r3": list(range(15)),
    # Update 2022-09-20
    "realsumm": list(range(9)),
}


def eval_random_template(model, dataset, is_regression):
    dataset = dataset.lower()

    if is_regression:
        dataset = dataset + "_reg"
        exp_dir = f"experiments_balanced/{dataset}"
    else:
        exp_dir = f"experiments_balanced/{dataset}_2class"

    command = f"bash bin/GEM-eval-template.sh {model} {dataset} -1 {exp_dir}"
    subprocess.run([command], stdout=subprocess.PIPE, shell=True)


def eval_all_templates(model, dataset, is_regression=True):
    dataset = dataset.lower()
    templates = dict_dataset_2_template_idx[dataset]

    if is_regression:
        dataset = dataset + "_reg"
        exp_dir = f"experiments_balanced/{dataset}"
    else:
        exp_dir = f"experiments_balanced/{dataset}_2class"
    
    for template_idx in templates:
        print("\n" * 8)
        print("=======================")
        print("TEMPLATE", template_idx)
        print("=======================")
        print("\n" * 8)
        command = f"bash bin/GEM-eval-template.sh {model} {dataset} {template_idx} {exp_dir}"
        subprocess.run([command], stdout=subprocess.PIPE, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all_template_or_random_template", required=True, choices=["all", "random"])
    parser.add_argument("-r", "--reg_or_class", required=True, choices=["reg", "class"])
    parser.add_argument("-model", "--model", default="t03b")
    parser.add_argument("-dataset", "--dataset", default="realsumm")
    args = parser.parse_args()

    if args.all_template_or_random_template == "all":
        eval_all_templates(args.model, args.dataset, is_regression=args.reg_or_class == "reg")
    else:
        eval_random_template(args.model, args.dataset, is_regression=args.reg_or_class == "reg")

