# 基于 Baseline 的 DNC-ICA 剩余时间预测实验

本项目用于业务流程事件日志的剩余时间预测。原始基线实验保留在 `baseline.py` 中；新增创新实验单独放在 `baseline_dnc_ica.py` 中，避免污染 baseline 代码，便于做公平对比、消融实验和论文写作。

## 项目目标

在 `baseline.py` 的数据读取、前缀样本构建、多随机种子评估和结果导出流程基础上，新增一个面向剩余时间预测的创新模型 `DNCICA`。该模型融合了三类创新：

1. 数据层创新：异常值处理、缺失值标记、滚动统计、EWMA、偏度、峰度等时序特征构建。
2. 模型层创新：局部-全局 DNC 降噪卷积、ICA 自适应通道混合、LSTM 序列骨干。
3. 训练层创新：动态加权 DAL 风格损失，针对稀有变体、长剩余时间样本和高波动样本自适应加权。

`baseline.py` 不做修改；所有新增方法都集中在 `baseline_dnc_ica.py`。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `baseline.py` | 原始基线实验脚本，包含 VanillaTransformer、LSTM、RandomForest，多 seed 训练与评估。 |
| `baseline_dnc_ica.py` | 新增创新脚本，在 baseline 流程上加入 DNCICA 模型、特征构建、特征门控和动态加权损失。 |
| `model.py` | 原有模型定义，包含 VanillaTransformerBaseline 等。 |
| `prefix.py` | 将事件日志转成前缀预测样本，包括 activity/resource/time/mask/target 等字段。 |
| `training_utils.py` | 公共训练辅助函数，包括数据划分、分桶评估等。 |
| `dataset/` | 预处理后的数据集目录。 |
| `results/` | 实验结果输出目录。 |

## 环境要求

建议使用 Python 3.10 或相近版本。当前代码依赖：

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`

可以先验证核心依赖：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
python --version
python -c "import torch, pandas, numpy, sklearn; print(torch.__version__)"
```

## 数据格式

默认运行的数据集为：

```text
dataset/processed_Sepsis.csv
```

数据文件需要包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `CaseID` | 流程实例 ID。 |
| `Activity` | 当前事件活动名称。 |
| `Resource` | 当前事件资源/执行者。 |
| `Timestamp` | 事件时间戳，用于按 case 时间划分。 |
| `TimeSinceLast` | 当前事件距离上一事件的时间间隔。 |
| `TimeSinceStart` | 当前事件距离 case 开始的累计时间。 |
| `Next_Activity` | 下一活动标签，当前脚本主要用于构造前缀。 |
| `Next_Event_Time` | 下一事件时间间隔标签，当前脚本主要用于构造前缀。 |
| `Remaining_Time` | 剩余时间预测目标。 |

如果要更换数据集，修改 `baseline_dnc_ica.py` 中的：

```python
DATASET_PATH = "dataset/processed_Sepsis.csv"
```

## 快速运行

运行创新实验：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
python baseline_dnc_ica.py
```

运行原始 baseline 对照：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
python baseline.py
```

## 实验配置

`baseline_dnc_ica.py` 默认配置：

```python
DATASET_PATH = "dataset/processed_Sepsis.csv"
REPORT_DIR = "results"
TRAIN_SPLIT_RATIO = 0.8
SPLIT_STRATEGY = "case"
MAX_SEQ_LENGTH = 50
MAX_PREFIXES = 100

BATCH_SIZE = 128
LEARNING_RATE = 3e-4
EPOCHS = 30
FEATURE_GATE_LAMBDA = 1e-4

D_MODEL = 128
LSTM_NUM_LAYERS = 2
DNC_LAYERS = 2

SEEDS = [42, 67, 80, 89]
```

说明：

- `SPLIT_STRATEGY = "case"` 表示按流程实例划分训练集和验证集，避免同一个 case 同时出现在训练集与验证集。
- `MAX_SEQ_LENGTH` 控制前缀最大长度。
- `MAX_PREFIXES` 控制单个 case 最多生成多少个前缀样本。
- `FEATURE_GATE_LAMBDA` 控制特征门控稀疏约束强度。
- `SEEDS` 使用 4 个随机种子，输出均值和标准差。

## 创新方法说明

### 1. 数据层创新：因果时序特征构建

实现位置：

```text
baseline_dnc_ica.py -> CausalTemporalFeatureBuilder
```

该模块从原始时间特征 `TimeSinceLast` 和 `TimeSinceStart` 中构造额外变量：

| 特征组 | 含义 |
| --- | --- |
| `clean` | 对 NaN、Inf、负值和极端异常值处理后的时间特征。 |
| `delta` | 当前时间特征相对上一事件的变化量。 |
| `roll3_mean` | 长度为 3 的因果滚动均值。 |
| `roll3_std` | 长度为 3 的因果滚动标准差。 |
| `roll3_median` | 长度为 3 的因果滚动中位数。 |
| `roll5_mean` | 长度为 5 的因果滚动均值。 |
| `roll5_std` | 长度为 5 的因果滚动标准差。 |
| `ewma` | 指数加权移动平均，强调近期变化。 |
| `robust_z` | 基于样本内均值和标准差的鲁棒标准化偏离程度。 |
| `anomaly_flag` | 极端异常值标记。 |
| `missing_flag` | 缺失或非有限值标记。 |
| `seq_skew` | 前缀时间序列偏度，描述不对称性。 |
| `seq_kurtosis` | 前缀时间序列峰度，描述尖锐程度或尾部厚度。 |

这些特征都是因果构造，只使用当前位置及之前的信息，不泄露未来。

### 2. 特征选择创新：可学习 Feature Gate

实现位置：

```text
baseline_dnc_ica.py -> FeatureGate
```

构造出的时间特征先经过一个可学习门控：

```python
selected_time = engineered_time * sigmoid(feature_gate_logits)
```

作用：

- 自动学习哪些时序构造特征更重要。
- 给模型提供一种可解释的软特征选择机制。
- 训练结束后导出每个特征的 gate 值，用于解释和论文分析。

输出文件：

```text
results/baseline_dnc_ica_feature_gates.csv
```

gate 越大，说明该特征被模型使用得越强；gate 越小，说明该特征被弱化。

### 3. 模型层创新：DNCICA

实现位置：

```text
baseline_dnc_ica.py -> DNCICABaseline
```

整体结构：

```text
Activity Embedding
Resource Embedding
Constructed Time Features
        ↓
Feature Gate
        ↓
Input Projection + LayerNorm
        ↓
Local-Global DNC Conv
        ↓
LSTM Backbone
        ↓
ICA Mixer Readout
        ↓
Regression Head
```

#### Local-Global DNC Conv

实现位置：

```text
baseline_dnc_ica.py -> LocalGlobalDNCConv
```

核心思想：

- 局部分支：用 depthwise temporal convolution 提取短期局部模式，并降低局部噪声影响。
- 全局分支：维护可学习的全局时间记忆 `global_memory`。
- 融合分支：将局部特征和全局记忆堆叠，用 2D 卷积融合。
- 门控分支：用 gate 控制全局信息进入局部表示的强度。

这对应“局部-全局 DNC 卷积”方向，也借鉴了 GTR 中全局可学习表示与局部输入融合的思路。

#### ICA Mixer

实现位置：

```text
baseline_dnc_ica.py -> ICAMixer
```

核心思想：

- 不使用标准 self-attention 的二次复杂度结构。
- 用实例级上下文生成 channel gate。
- 用 token gate 控制每个事件位置的信息保留。
- 通过轻量局部卷积和门控混合实现自适应读出。

这个模块用于替代传统注意力读出，降低计算复杂度，同时保留自适应建模能力。

### 4. 训练层创新：动态加权 DAL 风格损失

实现位置：

```text
baseline_dnc_ica.py -> adaptive_dal_loss
```

损失由三类动态权重组成：

| 权重 | 作用 |
| --- | --- |
| 稀有变体权重 | 对低频流程变体加权，提升长尾样本表现。 |
| 峰值权重 | 对剩余时间较长、预测难度更高的样本加权。 |
| 波动权重 | 对时间特征波动更强的样本加权。 |

基础误差项使用：

```text
0.7 * MAE + 0.3 * SmoothL1
```

并加入特征门控稀疏惩罚：

```text
FEATURE_GATE_LAMBDA * feature_gate_l1_penalty
```

这样做的目标是让模型不仅追求整体 MAE，还更关注稀有、高峰、高波动样本。

## 对比模型

`baseline_dnc_ica.py` 默认同时运行：

| 模型 | 说明 |
| --- | --- |
| `DNCICA` | 新增创新模型。 |
| `VanillaTransformer` | 原始 Transformer baseline。 |
| `LSTM` | 原始 LSTM baseline。 |
| `RandomForest` | 传统机器学习 baseline。 |

所有模型使用同一份训练/验证划分和同一组 seed。

## 输出文件

运行结束后会生成：

| 文件 | 内容 |
| --- | --- |
| `results/baseline_dnc_ica_per_seed.csv` | 每个 seed、每个模型的 MAE/RMSE/Tail MAE/Score。 |
| `results/baseline_dnc_ica_summary.csv` | 各模型跨 seed 的均值和标准差。 |
| `results/baseline_dnc_ica_bucket_metrics.csv` | head/torso/tail 分桶评估结果。 |
| `results/baseline_dnc_ica_feature_gates.csv` | DNCICA 的特征门控权重。 |

## 指标解释

| 指标 | 含义 |
| --- | --- |
| `MAE` | 平均绝对误差，越低越好。 |
| `RMSE` | 均方根误差，对大误差更敏感，越低越好。 |
| `Tail MAE` | 长尾/低频变体样本上的 MAE，越低越好。 |
| `Score` | 当前脚本中使用 `0.5 * MAE + 0.5 * Tail MAE`，越低越好。 |

## 如何做消融实验

建议在 `baseline_dnc_ica.py` 中复制一个模型配置或增加开关，做以下消融：

| 实验 | 目的 |
| --- | --- |
| 去掉 DNC 卷积 | 验证局部-全局卷积是否有效。 |
| 去掉 ICA Mixer | 验证自适应读出是否有效。 |
| 去掉 Feature Gate | 验证特征选择是否有效。 |
| 去掉动态 DAL 损失 | 验证动态样本加权是否有效。 |
| 只保留基础时间特征 | 验证特征构建是否有效。 |

推荐命名：

```text
DNCICA
DNCICA_no_dnc
DNCICA_no_ica
DNCICA_no_gate
DNCICA_no_dal
```

## 论文写作建议

可以将方法组织成三层贡献：

1. 数据层：提出面向事件日志剩余时间预测的因果时序特征构建与异常标记方法。
2. 模型层：提出局部-全局 DNC 卷积与 ICA 自适应读出结合的轻量序列模型。
3. 优化层：提出考虑稀有变体、峰值剩余时间和时间波动的动态加权损失。

推荐表述：

```text
We propose DNCICA, a baseline-compatible remaining-time prediction framework that integrates causal temporal feature construction, learnable feature gating, local-global denoising convolution, instance-channel adaptive sequence readout, and dynamically weighted regression loss.
```

中文表述：

```text
本文提出 DNCICA，一种兼容基线流程的业务过程剩余时间预测框架。该方法从数据层、模型层和训练层联合改进：首先构建因果时序统计特征并进行可学习特征选择；随后利用局部-全局 DNC 卷积提取去噪后的序列表示；最后通过 ICA 自适应读出和动态加权损失提升模型对长尾、高波动和长剩余时间样本的预测能力。
```

## 常见问题

### 为什么不直接改 `baseline.py`？

`baseline.py` 是原始对照实验。保持它不变，可以保证创新模型与原始 baseline 的差异清晰、可复现、可解释。

### 为什么 seed 是 4 个？

当前设置为：

```python
SEEDS = [42, 67, 80, 89]
```

这样既能体现多随机种子稳定性，又不会像 5 个或更多 seed 那样显著增加训练时间。

### 为什么使用 case 划分？

默认：

```python
SPLIT_STRATEGY = "case"
```

按 case 划分可以避免同一个流程实例的不同前缀同时进入训练集和验证集，评估更严格。

### 如何缩短测试时间？

可以临时调小：

```python
EPOCHS = 1
BATCH_SIZE = 32
SEEDS = [42]
MAX_PREFIXES = 20
```

确认代码能跑通后，再恢复完整实验配置。

### 如何查看哪些特征最重要？

运行后查看：

```text
results/baseline_dnc_ica_feature_gates.csv
```

可以按 `gate` 从大到小排序，gate 越高表示该构造特征越被模型保留。

## 开发验证

已进行以下基础验证：

```powershell
python -m py_compile baseline_dnc_ica.py
python -c "import baseline_dnc_ica; print('import ok')"
```

同时完成过小批量前向传播、反向传播和 1 个 epoch 的 tiny DataLoader 训练闭环验证。

