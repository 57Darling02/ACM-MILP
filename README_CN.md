# ACM-MILP：面向保难度MILP实例生成的自适应分组选择约束修改方法

本文档是论文《ACM-MILP: Adaptive Constraint Modification via Grouping and Selection for Hardness-Preserving MILP Instance Generation》（作者：郭子傲、李阳、刘畅、欧阳文利、严骏驰，发表于ICML 2024）的配套代码说明。

## 环境配置

### Python 环境
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

### MILP 求解器
- [Gurobi](https://www.gurobi.com/) 10.0.1（需学术许可）

### 配置管理工具
- [Hydra](https://hydra.cc/docs/intro/)：用于超参数管理和实验调度

你可通过执行 `scripts/environment.sh` 脚本中的命令搭建环境，也可通过环境配置文件构建：
```
conda env create -f scripts/environment.yaml
```

## 使用方法
进入代码根目录，将数据集放置在 `./data` 目录下。目录结构示例如下：
```
ACM-MILP
├── conf                 # 配置文件目录
├── data                 # 数据集目录
│   ├── ca               # ca数据集
│   │   ├── train/       # 训练集
│   │   └── test/        # 测试集
│   ├── mis              # mis数据集
│   │   ├── train/
│   │   └── test/
│   └── setcover         # setcover数据集
│       ├── train/
│       └── test/
├── scripts/             # 脚本目录
├── src/                 # 核心代码目录
├── README.md            # 说明文档
├── generate.py          # 实例生成脚本
├── preprocess.py        # 数据预处理脚本
└── train.py             # 模型训练脚本
```

超参数配置文件位于 `./conf/` 目录，各数据集运行命令在 `./scripts/` 目录，核心代码逻辑在 `./src/` 目录。以下以MIS数据集为例，说明ACM-MILP的完整使用流程：

### 1. 数据预处理
执行以下命令预处理数据集：
```
python preprocess.py dataset=mis num_workers=10
```
该命令会生成实例的图结构数据及数据集统计信息（用于模型训练），预处理结果将保存至 `./preprocess/mis/` 目录。

### 2. 训练ACM-MILP模型
使用默认参数训练模型：
```
python train.py dataset=mis cuda=0 num_workers=10 job_name=mis-default
```
训练日志保存路径：`TRAIN DIR=./outputs/train/${DATE}/${TIME}-${JOB NAME}/`（`${DATE}`为日期、`${TIME}`为时间、`${JOB NAME}`为任务名）；
模型权重文件保存至 `${TRAIN DIR}/model/` 目录；
生成的实例及基准测试结果保存至 `${TRAIN DIR}/eta-${eta}/` 目录（`${eta}`为对应超参数）。

### 3. 生成新的MILP实例
基于训练好的模型生成新实例：
```
python generate.py dataset=mis \
    generator.mask_ratio=0.01 \
    cuda=0 num_workers=10 \
    dir=${TRAIN DIR}
```
生成的实例及基准测试结果将保存至 `${TRAIN DIR}/generate/${DATE}/${TIME}` 目录。

## 引用说明
如果该代码对你的研究有帮助，请引用下述论文：
```
@inproceedings{
guo2024acmmilp,
title={ACM-MILP: Adaptive Constraint Modification via Grouping and Selection for Hardness-Preserving MILP Instance Generation},
author={Ziao Guo, Yang Li, Chang Liu, Wenli Ouyang, Junchi Yan},
booktitle={Forty-first International Conference on Machine Learning},
year={2024}
}
```
