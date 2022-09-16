# Experiments Schedule 

## REALSumm 

In this section, we plan and describe the experiments conducted using the REALSumm dataset. 
We separate this experiment into a few stages: 
(1) balanced vs non-balanced; 
(2) numerical vs word;
(3) selection of few-shot examples.

We'll use the data available in `./datasets/summ_data/REALSumm/quantile` with `dataset_classes=2`. 
### Balanced, word, random sampling (stratified by `bin`)

We collect 15 different seeds for different numbers of shots: $\{100, 64, 16, 4 , 2\}$. 
We set `add_special_tokens=False` to avoid length normalization problems. 

#### Baselines 

| Name               | Machine  | Seed | Started | Command | Results |
| ------------------ | -------- | ---- | ------- | ------- | ------- |
| T0                 | s5, 3    | 100  |
| T0-FT-TRAIN        | s5, 4    | 100  |
| T0-FT-TRAIN-PEFT   | s5, 5    | 100  | 
| T0-FT-10k          | s5, 6    | 100  | 
| T0-FT-10k          | s5, 7    | 100  |


#### Experiments

| Machine  | Seed | Started | Command | Results |
| -------- | ---- | ------- | ------- | ------- |
| s5, 3 | 
| s5, 4 | 
| s5, 5 |
| s5, 6 |
| s5, 7 |
