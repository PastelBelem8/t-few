# T-Few 

This repository contains an adapted version of the official code for the paper: "[Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638)". Check their official repository for more information on how to run the original T-few.

## Setup

First, create a virtual environment for the project and install all the requirments.
(We use conda to manage environments. Be sure to install and initialize conda first.)

1. Create a virtual environment with python 3.7 `conda create -n tfew python==3.7`, then activate the environment `conda activate tfew`.
2. Install other dependencies. `pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html`

The steps above only needs to be done once. In addition, every time you start a new session, you will need to run `. bin/start.sh`

### Setting up in the cluster 

1. Create a folder for your project named `~/my_project` in your home dir at the cluster.
2. Retrieve the data from `/srv/disk00/ucinlp/GEM-workshop-2022/data-experiments-20220910.tar.gz` and extract it to the folder `~/my_project/experiments`. You can use the command `$ mkdir -p ~/my_project && tar -xvzf /srv/disk00/ucinlp/GEM-workshop-2022/data-experiments-20220910.tar.gz -C ~/my_project`.
3. Retrieve the code source either by downloading it from this repository (for the most up-to-date version) or by extracting the code from `/srv/disk00/ucinlp/GEM-workshop-2022/t-few-master.tar.gz`. You can use the command `tar -xvzf /srv/disk00/ucinlp/GEM-workshop-2022/t-few-master.tar.gz -C ~/my_project` to extract the folder into your main project folder.
4. If you use VS Code, we suggest you connect your open the `~/my_project`.

### Running in the cluster

Assuming you have setup the project in the cluster, a standard way of running the experiments is through the scripts available at `~/my_project/t-few-master/bins`. All the scripts mentioned in this section refer to the fine-tuning of T0 3B models. 


#### Training / Running an experiment

The following scripts are the ones responsible for creating the experiments. 

- `GEM-MT-few-shot-pretrained-3b-100k.sh` (and `GEM-SUMM-few-shot-pretrained-3b-100k`): the former runs the experiments for the MT task, whereas the latter runs the experiments for the summarization task. In order to run the MT script, you'll have to update the value of a configuration in the [MT script](https://github.com/PastelBelem8/t-few/blob/master/bin/GEM-MT-few-shot-pretrained-3b-100k.sh#L32), thus reflecting your own directory. In this case, we sugest you replace with `CONFIG_DATA_DIR=\"~/my_project/experiments/mt_data/preproc/${dataset}\"` (or the absolute version of the path if you prefer). Make sure that you keep the escaped quotes, so you don't get any errors in the parsing of the configurations. Similarly, if you run the SUMM script, you will have to update the `data_dir` configuration (either on the [config file](https://github.com/PastelBelem8/t-few/blob/master/configs/realsumm.json#L3) directly) or by overriding its value in the [script](https://github.com/PastelBelem8/t-few/blob/master/bin/GEM-SUMM-few-shot-pretrained-3b-100k.sh#L20) by adding the argument `data_dir=\"~/my_project/experiments/mt_data/preproc/${dataset}\"`.

By default these scripts will run the experiments for $k=[128, 100, 64, 32, 16, 4, 2]$. You can adapt the script to run the number of shots you need. 
To run the script we use the command: `conda activate <CONDA_ENV> && cd ~/my_project/t-few-master && source ./bin/GEM-MT-few-shot-pretrained-3b-100k.sh <GPU_DEVICE> <SEED>`. The seed is used for setting training the model and for selecting the few-shot samples. However, you can specify different seeds if you so desire by changing the value of the configs: `few_shot_random_seed=${seed} seed=${seed}`


**Note**: When running an experiment for the first time, a directory  `~/my_project/t-few-master/data/few_shot/<DATASET_NAME>/class_{N}` is created. There you'll find for each number of shot K and each seed S the json files `<K>_shot/<S>_seed.jsonl` with the information about the fewshot examples being used (and re-used if you re-run the same exact experiment) for fine-tuning!


#### Evaluation

We create the following evaluation scripts. 

- `GEM-MT-eval-baselines-3b.sh` (and [`GEM-SUMM-eval-baselines-3b.sh`](https://github.com/PastelBelem8/t-few/blob/master/bin/GEM-SUMM-eval-baselines-3b.sh)): which runs the [0-shot evaluation of T0_3B model](https://github.com/PastelBelem8/t-few/blob/master/bin/GEM-MT-eval-baselines-3b.sh#L46-L49) (dubbed **baseline**) and two other variants of T0 that were fine tuned on the training set of the corresponding datasets. The first variant is the continual pretraining of a [fine-tuned version of T0_3B with (IA)3](https://github.com/PastelBelem8/t-few/blob/master/bin/GEM-MT-eval-baselines-3b.sh#L57-L59), whereas the [second variant](https://github.com/PastelBelem8/t-few/blob/master/bin/GEM-MT-eval-baselines-3b.sh#L62-L64) differs from the former just on the number of steps -- we train it for 10k steps instead of 1k step. 

We use the following command to run the evaluation script: `$ conda activate <CONDA_ENV_NAME> && cd ~/my_project/t-few-master && source ./bin/GEM-MT-eval-baselines-3b.sh <GPU_DEVICE>`, where <CONDA_ENV_NAME> is the name of your CONDA ENVIRONMENT and <GPU_DEVICE> is a single integer referring to the GPU device to use for the evaluation.

**note**: You can configure the output directory by changing the variable `OUTPUT_PATH`.




## Run your first experiment (should work for the original T-few experiments)

Once you finished setting up the environment, you can try running
`CUDA_VISIBLE_DEVICES=3 python -m src.pl_train -c t0.json+rte.json -k save_model=False exp_name=first_exp`
The outputs of this run will be saved to `${OUTPUT_PATH}/first_exp/`, which is usually `/t-few/exp_out/first_exp/`. Here, `first_exp` is the experiment name, you can run more experiments with different expeirment names. The code will automatically skip finished experiments. (However, if you wish to rerun a finished experiment under the same experiment name, you will need to manually remove the corresponding files in the output directory.)

There are two ways to control an experiment.

1. You can specify **config files** with `-c`. Multiple config files can be combined with `+`. (When there are conflits, config terms from the config file on the right will have greater power.) This will be convinient when you have multiple terms that forms a fixed group.
2. You can **override values** with `-k`. This will be convenient when you need to change a small number of terms.

It is recommended to use GPUs with 40GB to train T0(3B) and 80GB to train T0.
