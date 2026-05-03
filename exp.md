# 铁路 MILP 三步实验指南

本文档面向当前仓库中的铁路 MILP 迁移实验。目标不是一次性把所有实验都跑满，而是分三步逐层推进：

1. 第一步：快速调参，先排除明显失败的配置。
2. 第二步：中期筛选，在更可信的规模上比较候选模型。
3. 第三步：最终训练与论文结果导出。

文档中的命令均为 Linux 命令，可直接在服务器终端执行。默认工作目录为仓库根目录。

## 0. 重要约定

- 你的铁路数据目前仍复用 `mis` 目录：
  - 原始样本目录：`data/mis/train/`
  - 预处理目录：`preprocess/mis/`
- 本文新增的 `rail_*` 配置不会改动目录名。
  - 它们通过 `dataset.name: mis` 继续复用现有路径。
- `.gitignore` 会忽略 `data/`、`preprocess/`、`outputs/`。
  - 这次新增的 `conf/` 配置和本文档 `exp.md` 会被 Git 跟踪，可以直接同步到服务器。
- 当前代码已补上两点：
  - `train.py` 现在会真正使用 `pretrained_model_path`，可从上一步最佳权重继续训练。
  - benchmark 现在支持配置 `mip_gap` 和 `time_limit`，不再固定用 `60s / 0 gap`。

## 1. 第一步：快速调参

### 1.1 目标

- 用小样本生成和低频 benchmark 快速判断配置是否有效。
- 这一阶段只看趋势，不追求最终论文指标。

### 1.2 适用配置

- `dataset=rail_dev`
- `model=rail`
- `trainer=rail_dev`
- `generator=rail_dev`
- `benchmarking=rail_dev`

对应含义：

- 最多使用 `120` 个训练实例；如果你的 `data/mis/train/` 实际少于 120 个，就按实际数量加载。
- 训练总步数 `2400`
- 训练中只在 `1200 / 1800 / 2400` 左右做中间生成评估
- 每次只生成 `20` 个样本
- benchmark 求解参数使用 `mip_gap=0.1`、`time_limit=120`

### 1.3 运行命令

如果你还没有在服务器上做预处理，先执行：

```bash
python preprocess.py dataset=rail_dev num_workers=10
```

开始第一步训练：

```bash
python train.py \
  dataset=rail_dev \
  model=rail \
  trainer=rail_dev \
  generator=rail_dev \
  benchmarking=rail_dev \
  cuda=0 \
  num_workers=10 \
  job_name=rail-step1
```

### 1.4 你要检查什么

训练完成后，重点看以下文件：

- `outputs/train/<DATE>/<TIME>-rail-step1/train.log`
- `outputs/train/<DATE>/<TIME>-rail-step1/eta-0.1/benchmark_step_1200/info.json`
- `outputs/train/<DATE>/<TIME>-rail-step1/eta-0.1/benchmark_step_2400/info.json`

重点判断：

- `distribution.score` 是否明显高于你之前的 `0.10`
- `coef_dens / var_degree_mean / cons_degree_mean` 是否不再接近 `0`
- `var_degree_min / cons_degree_min` 是否仍出现大量 `0`

### 1.5 进入下一步的标准

建议至少满足下面两条再进入第二步：

- `distribution.score` 明显优于旧结果，最好先达到 `0.3` 以上
- 生成结构不再系统性出现孤立变量、孤立约束

如果第一步仍然很差，不要扩大规模，先继续调：

- `conf/model/rail.yaml`
- `conf/generator/rail_dev.yaml`

## 2. 第二步：中期筛选

### 2.1 目标

- 从第一步中表现最好的权重配置继续训练。
- 用更长训练和更多生成样本筛选最终候选模型。

### 2.2 适用配置

- `dataset=rail_select`
- `model=rail`
- `trainer=rail_select`
- `generator=rail_select`
- `benchmarking=rail_select`

对应含义：

- 最多使用 `300` 个训练实例
- 训练总步数 `4000`
- 每次生成 `50` 个样本
- benchmark 求解参数使用 `mip_gap=0.1`、`time_limit=300`

### 2.3 从第一步继续训练

假设第一步训练目录记为：

```bash
STEP1_DIR=outputs/train/<DATE>/<TIME>-rail-step1
```

从第一步最佳权重继续训练：

```bash
python train.py \
  dataset=rail_select \
  model=rail \
  trainer=rail_select \
  generator=rail_select \
  benchmarking=rail_select \
  pretrained_model_path=${STEP1_DIR}/model/model_best.ckpt \
  cuda=0 \
  num_workers=10 \
  job_name=rail-step2
```

如果你想从第一步的嵌入模型起步，也可以改成：

```bash
pretrained_model_path=${STEP1_DIR}/model/emb_model_best.ckpt
```

但一般优先建议从 `model_best.ckpt` 开始。

### 2.4 你要检查什么

训练完成后，重点比较：

- `eta-0.1/benchmark_step_1600/info.json`
- `eta-0.1/benchmark_step_2400/info.json`
- `eta-0.1/benchmark_step_3200/info.json`
- `eta-0.1/benchmark_step_4000/info.json`

关注三类指标：

- `distribution.score`
- `solving.solving_time.mean_error`
- `solving.num_nodes.mean_error`

你的目标不是让三者同时最优，而是找到“结构相似性明显提升，同时难度误差开始下降”的 checkpoint。

## 3. 第三步：最终训练与论文结果导出

### 3.1 目标

- 用最终配置做完整训练
- 导出论文需要的最终生成结果与 benchmark
- 对 `eta=0.05 / 0.1 / 0.2` 做统一评估

### 3.2 适用配置

- `dataset=rail_final`
- `model=rail`
- `trainer=rail_final`
- `generator=rail_final`
- `benchmarking=rail_final`

对应含义：

- 最多使用 `1000` 个训练实例
- 训练总步数 `6000`
- 每次生成 `150` 个样本
- benchmark 求解参数使用 `mip_gap=0.1`、`time_limit=500`
  - 这一步和你的预处理求解设置保持一致，更适合论文汇报

### 3.3 从第二步最佳权重继续训练

假设第二步训练目录记为：

```bash
STEP2_DIR=outputs/train/<DATE>/<TIME>-rail-step2
```

执行最终训练：

```bash
python train.py \
  dataset=rail_final \
  model=rail \
  trainer=rail_final \
  generator=rail_final \
  benchmarking=rail_final \
  pretrained_model_path=${STEP2_DIR}/model/model_best.ckpt \
  cuda=0 \
  num_workers=10 \
  job_name=rail-step3
```

### 3.4 训练完成后的最终生成

假设第三步训练目录记为：

```bash
STEP3_DIR=outputs/train/<DATE>/<TIME>-rail-step3
```

分别对三个修改比例生成最终结果：

```bash
python generate.py \
  dataset=rail_final \
  model=rail \
  generator=rail_final \
  benchmarking=rail_final \
  cuda=0 \
  num_workers=10 \
  dir=${STEP3_DIR} \
  generator.mask_ratio=0.05
```

```bash
python generate.py \
  dataset=rail_final \
  model=rail \
  generator=rail_final \
  benchmarking=rail_final \
  cuda=0 \
  num_workers=10 \
  dir=${STEP3_DIR} \
  generator.mask_ratio=0.1
```

```bash
python generate.py \
  dataset=rail_final \
  model=rail \
  generator=rail_final \
  benchmarking=rail_final \
  cuda=0 \
  num_workers=10 \
  dir=${STEP3_DIR} \
  generator.mask_ratio=0.2
```

这些命令会把结果写到：

- `${STEP3_DIR}/generate/<DATE>/<TIME>-eta-0.05/`
- `${STEP3_DIR}/generate/<DATE>/<TIME>-eta-0.1/`
- `${STEP3_DIR}/generate/<DATE>/<TIME>-eta-0.2/`

### 3.5 如需重跑 benchmark

如果你已经有生成样本，只想重跑评估：

```bash
python scripts/benchmark.py \
  --samples-dir ${STEP3_DIR}/eta-0.1/samples \
  --benchmark-dir ${STEP3_DIR}/eta-0.1/benchmark_rerun \
  --dataset-stats-dir preprocess/mis/stats \
  --num-workers 10 \
  --num-samples 10000 \
  --mip-gap 0.1 \
  --time-limit 500
```

### 3.6 导出可视化

训练损失曲线：

```bash
python scripts/visualize_train.py \
  --train-dir ${STEP3_DIR}
```

生成结果可视化，以 `eta=0.1` 为例：

```bash
python scripts/visualize_generate.py \
  --benchmark-dir ${STEP3_DIR}/eta-0.1/benchmark \
  --reference-stats-dir preprocess/mis/stats
```

## 4. 建议的论文口径

建议你把三步实验分别表述为：

- 第一步：迁移调参实验，用于验证 ACM-MILP 在铁路 MILP 场景中的可行性
- 第二步：中等规模筛选实验，用于确定最终训练配置
- 第三步：正式实验，用于生成论文中的结果表和图

这样写的好处是：

- 不会把早期失败实验误写成最终结论
- 能合理解释为什么前期使用了较小的样本数和较短的训练周期
- 能把“算力受限下的分阶段实验设计”写成方法上的审慎选择

## 5. 当前新增配置清单

本次新增并建议纳入 Git 的文件有：

- `conf/model/rail.yaml`
- `conf/dataset/rail_dev.yaml`
- `conf/dataset/rail_select.yaml`
- `conf/dataset/rail_final.yaml`
- `conf/trainer/rail_dev.yaml`
- `conf/trainer/rail_select.yaml`
- `conf/trainer/rail_final.yaml`
- `conf/generator/rail_dev.yaml`
- `conf/generator/rail_select.yaml`
- `conf/generator/rail_final.yaml`
- `conf/benchmarking/rail_dev.yaml`
- `conf/benchmarking/rail_select.yaml`
- `conf/benchmarking/rail_final.yaml`
- `exp.md`

同步到服务器前，可在本地检查：

```bash
git status
git add exp.md conf/model/rail.yaml conf/dataset/rail_dev.yaml conf/dataset/rail_select.yaml conf/dataset/rail_final.yaml conf/trainer/rail_dev.yaml conf/trainer/rail_select.yaml conf/trainer/rail_final.yaml conf/generator/rail_dev.yaml conf/generator/rail_select.yaml conf/generator/rail_final.yaml conf/benchmarking/rail_dev.yaml conf/benchmarking/rail_select.yaml conf/benchmarking/rail_final.yaml train.py src/benchmarks/benchmarks.py src/benchmarks/utils.py scripts/benchmark.py conf/benchmarking/default.yaml
git status
```
