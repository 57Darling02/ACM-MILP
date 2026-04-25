# ACM-MILP: Adaptive Constraint Modification via Grouping and Selection for Hardness-Preserving MILP Instance Generation

This is the code of paper **"ACM-MILP: Adaptive Constraint Modification via Grouping and Selection for Hardness-Preserving MILP Instance Generation"**. *Ziao Guo, Yang Li, Chang Liu, Wenli Ouyang, Junchi Yan.* ICML 2024. 

## Environment
- Python environment
    - python 3.7
    - pytorch 1.13
    - torch-geometric 2.3
    - ecole 0.7.3
    - pyscipopt 3.5.0
    - community 0.16
    - networkx
    - pandas
    - tensorboardX
    - gurobipy

- MILP Solver
    - [Gurobi](https://www.gurobi.com/) 10.0.1. Academic License.

- Hydra
    - [Hydra](https://hydra.cc/docs/intro/) for managing hyperparameters and experiments.


In order to build the environment, you can follow commands in `scripts/environment.sh`.

Or alternatively, to build the environment from a file,
```
conda env create -f scripts/environment.yaml
```

## Usage

Go to the root directory. Put the datasets under the `./data` directory. Below is an illustration of the directory structure.
```
ACM-MILP
├── conf
├── data
│   ├── ca
│   │   ├── train/
│   │   └── test/
│   ├── mis
│   │   ├── train/
│   │   └── test/
│   └── setcover
│       ├── train/
│       └── test/
├── scripts/
├── src/
├── README.md
├── generate.py
├── preprocess.py
└── train.py
```

The hyperparameter configurations are in `./conf/`.
The commands to run for all datasets are in `./scripts/`.
The main part of the code is in `./src/`.
The workflow of ACM-MILP (using MIS as an example) is as following.

### 1. Preprocessing

To preprocess a dataset,
```
python preprocess.py dataset=mis num_workers=10
```
This will produce graph data for instances and the statistics of the dataset to be used for training. The preprocessed results are saved under `./preprocess/mis/`. 

### 2. Training **ACM-MILP**

To train ACM-MILP with default parameters,
```
python train.py dataset=mis cuda=0 num_workers=10 job_name=mis-default
```
The training log is saved under `TRAIN DIR=./outputs/train/${DATE}/${TIME}-${JOB NAME}/`. The model ckpts are saved under `${TRAIN DIR}/model/`. The generated instances and benchmarking results are saved under `${TRAIN DIR}/eta-${eta}/`.

### 3. Generating new instances

To generate new instances with a trained model,
```
python generate.py dataset=mis \
    generator.mask_ratio=0.01 \
    cuda=0 num_workers=10 \
    dir=${TRAIN DIR}
```
The generated instances and benchmarking results are saved under `${TRAIN DIR}/generate/${DATE}/${TIME}`.

### 4. Rerunning benchmark for existing samples

To rerun benchmark evaluation for generated samples without retraining,
```
python scripts/benchmark.py \
    --samples-dir ${TRAIN DIR}/eta-0.1/samples_step_500 \
    --benchmark-dir ${TRAIN DIR}/eta-0.1/benchmark_step_500_rerun \
    --dataset-stats-dir preprocess/mis/stats \
    --num-workers 10 \
    --num-samples 10000
```
This reads existing generated instances from `samples_step_500`, recomputes `features.csv` and `solving_results.csv`, and writes the benchmark summary to `info.json`.

If `--dataset-stats-dir` is omitted, it defaults to `preprocess/${dataset}/stats`, where `--dataset` defaults to `mis`.

### 5. Visualizing preprocess, train, and generate outputs

The repository also provides standalone visualization scripts under `./scripts/`. These scripts are designed to work directly on exported CSV files, TensorBoard event files, or training logs, so they can be run independently on a server even if the full training environment is not available locally.

Install the plotting dependencies first:
```
pip install matplotlib pandas scipy
```
If you want `visualize_train.py` to read TensorBoard event files directly, also install:
```
pip install tensorboard
```

To visualize preprocessing statistics with KDE curves for graph features and hardness plots from `solving_results.csv`,
```
python scripts/visualize_preprocess.py \
    --stats-dir preprocess/mis/stats
```
This reads `features.csv` and `solving_results.csv` under the stats directory and saves figures under `preprocess/mis/stats/visualizations/`.

To visualize the training loss curve,
```
python scripts/visualize_train.py \
    --train-dir outputs/train/${DATE}/${TIME}-${JOB NAME}
```
The script first tries to read `Train/total_loss` from TensorBoard event files. If TensorBoard parsing is unavailable, it automatically falls back to parsing the training log. The generated figure and extracted CSV are saved under `${TRAIN DIR}/visualizations/`.

To visualize generated benchmark outputs with KDE curves and hardness plots,
```
python scripts/visualize_generate.py \
    --benchmark-dir ${TRAIN DIR}/eta-0.1/benchmark_step_500 \
    --reference-stats-dir preprocess/mis/stats
```
This overlays generated distributions with the reference distributions from preprocessing and saves figures under `${TRAIN DIR}/eta-0.1/benchmark_step_500/visualizations/`.

Useful optional arguments:
```
--output-dir <DIR>      # custom output directory
--columns col1 col2     # only plot selected feature columns
--bins 40               # histogram bin count used with KDE plots
--dpi 300               # output image resolution
```

## Citation

If you find this code useful, please consider citing the following paper.

```
@inproceedings{
guo2024acmmilp,
title={ACM-MILP: Adaptive Constraint Modification via Grouping and Selection for Hardness-Preserving MILP Instance Generation},
author={Ziao Guo, Yang Li, Chang Liu, Wenli Ouyang, Junchi Yan},
booktitle={Forty-first International Conference on Machine Learning},
year={2024}
}
```
