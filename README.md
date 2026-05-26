# 基于 Baseline 的业务流程剩余时间预测实验

本项目用于业务流程事件日志的剩余时间预测。原始基线实验保留在 `baseline.py` 中；已有 DNC-ICA 实验放在 `baseline_dnc_ica.py` 中；本次新增 PGR-TC 实验放在 `baseline_pgr_tc.py` 中。这样可以避免污染 baseline 代码，便于做公平对比、消融实验和论文写作。

## 当前实验脚本

| 文件 | 作用 |
| --- | --- |
| `baseline.py` | 原始基线实验脚本，包含 VanillaTransformer、LSTM、RandomForest，多 seed 训练与评估。 |
| `baseline_dnc_ica.py` | DNC-ICA 创新脚本，在 baseline 流程上加入 DNCICA 模型、特征构建、特征门控和动态加权损失。 |
| `baseline_pgr_tc.py` | 本次新增 PGR-TC 脚本，加入 prefix 统计特征、LocalTCN、事件对齐 GTR 和 Tail-weighted Huber Loss。 |
| `docs/PGR_TC_IMPLEMENTATION_REPORT.md` | PGR-TC 详细实现报告，解释每个模块为什么做、做了什么、有什么好处。 |
| `model.py` | 原有模型定义，包含 VanillaTransformerBaseline 等。 |
| `prefix.py` | 将事件日志转成前缀预测样本，包括 activity/resource/time/mask/target 等字段。 |
| `training_utils.py` | 公共训练辅助函数，包括数据划分、分桶评估等。 |
| `dataset/` | 预处理后的数据集目录。 |
| `results/` | 实验结果输出目录。 |

---

# 一、PGR-TC 实验说明

## 1. PGR-TC 是什么

PGR-TC 全称暂定为：

```text
Prefix-aware Global Retrieval Network with Temporal Consistency-friendly training
```

中文可以理解为：

```text
融合前缀统计特征、局部时序卷积、全局时间检索和长尾加权损失的业务流程剩余时间预测模型
```

它不是普通时间序列预测模型，而是面向业务流程事件日志的剩余时间预测模型。输入是当前 case 的 trace prefix，输出是该 prefix 到 case 完成的 remaining time。

## 2. 为什么新增 PGR-TC

原始 `baseline.py` 中的 Transformer、LSTM 和 RandomForest 主要基于：

```text
activity sequence
resource sequence
TimeSinceLast / TimeSinceStart
```

这些信息可以建模事件前缀，但还存在三个不足：

```text
1. 没有显式刻画 prefix 内部的统计状态，比如等待时间波动、活动重复、资源切换。
2. 没有显式利用全局业务时间模式，比如周几、几点、流程阶段。
3. 普通 MAE 损失对 long remaining time 和长尾变体关注不足。
```

因此新增 `baseline_pgr_tc.py`，从数据层、模型层和损失层同时改进。

## 3. PGR-TC 总体结构

```text
act_seq, res_seq, time_seq
        ↓
EventEmbedding
        ↓
LocalTCN
        ↓
Dual Event-Aligned GTR
    ├── Calendar-GTR
    └── Progress-GTR
        ↓
TransformerEncoder
        ↓
Last Valid Hidden State
        ↓
Concat prefix_stat_feats
        ↓
Regression Head
        ↓
Remaining Time Prediction
```

---

## 4. 数据层改进：Prefix-level Statistical Features

`baseline_pgr_tc.py` 中新增：

```python
class EnhancedProcessPrefixDataset(ProcessPrefixDataset)
```

它继承原来的 `ProcessPrefixDataset`，不破坏旧数据管线，只额外加入：

```text
prefix_stat_feats
calendar_bucket_ids
progress_bucket_ids
prefix_len
```

新增 prefix 统计特征包括：

```text
prefix_delta_mean
prefix_delta_std
prefix_delta_max
prefix_delta_min
prefix_delta_median
prefix_delta_ewma
prefix_delta_skewness
prefix_delta_kurtosis
recent_delta_mean_3
recent_delta_std_3
recent_delta_mean_5
recent_delta_std_5
unique_activity_count
unique_activity_ratio
activity_repeat_count
activity_loop_count
recent_activity_repeat_count_3
activity_transition_entropy
unique_resource_count
unique_resource_ratio
resource_switch_count
resource_switch_ratio
recent_resource_switch_count_3
prefix_len
normalized_prefix_pos
last_hour
last_weekday
is_weekend
```

这些特征全部只使用当前 prefix 内已经发生的事件，不使用未来信息，因此不会造成标签泄露。

### 好处

| 特征类型 | 解决的问题 | 好处 |
| --- | --- | --- |
| delta 均值/方差/中位数 | 当前 case 的处理速度 | 更好表达时间状态 |
| EWMA | 最近事件更重要 | 对近期变化更敏感 |
| 偏度/峰度 | 极端等待、长尾 | 有助于识别异常拖延 case |
| 活动重复/loop | 返工、循环 | 对复杂流程变体更友好 |
| 转移熵 | 前缀结构复杂度 | 表达流程路径不确定性 |
| 资源切换 | 人员交接、资源变化 | 捕捉组织层面的延迟风险 |
| prefix_len/progress | 流程阶段 | 区分早期、中期、后期预测 |
| hour/weekday/weekend | 日历周期 | 捕捉业务时间节律 |

---

## 5. 模型层改进

### 5.1 EventEmbedding

融合：

```text
activity embedding + resource embedding + time feature projection
```

作用：把 activity、resource、time 三类事件信息映射到统一的 `d_model` 维度。

### 5.2 LocalTCN

使用轻量 Conv1d 做局部事件模式提取：

```text
Conv1d → GELU → Dropout → Conv1d → Residual → LayerNorm
```

作用：提取最近事件之间的局部模式，降低异常时间间隔和噪声事件对后续 Transformer 的干扰。

### 5.3 Event-Aligned GTR

原始 GTR 面向规则时间序列，而事件日志是不规则序列。因此这里改成事件对齐版本：

```text
每个事件根据 bucket_id 检索一个可学习全局原型向量。
```

当前有两类 GTR：

```text
Calendar-GTR：根据 weekday * 24 + hour 检索日历原型
Progress-GTR：根据 position / max_seq_len 检索流程阶段原型
```

### 5.4 DualRetrieverFusion

同时使用 Calendar-GTR 和 Progress-GTR，然后门控融合：

```text
gate * Calendar-GTR + (1 - gate) * Progress-GTR
```

这样模型可以自动判断当前事件更依赖日历周期信息，还是流程阶段信息。

### 5.5 TransformerEncoder

PGR-TC 仍然保留 TransformerEncoder 作为全局序列编码器。LocalTCN 和 GTR 负责增强事件表示，Transformer 负责建模整个 prefix 内的长距离依赖。

---

## 6. 损失层改进：Tail-weighted Huber Loss

新增：

```python
class TailWeightedHuberLoss(nn.Module)
```

默认使用：

```text
log1p(target) + Huber Loss + long-tail weight
```

权重形式：

```text
weight = 1 + alpha * log1p(target) / mean(log1p(target))
```

这样做的原因是 remaining time 往往长尾严重：大部分 case 较快完成，少部分 case 持续时间特别长。普通 MAE 容易偏向多数普通 case；Tail-weighted Huber 可以让模型更关注长剩余时间样本，同时保持训练稳定。

---

## 7. PGR-TC 支持的消融模型

`baseline_pgr_tc.py` 当前支持：

| 模型 | 含义 |
| --- | --- |
| `VanillaTransformer` | 原项目 Transformer baseline |
| `LSTM` | 原项目 LSTM baseline |
| `RandomForest` | 原项目传统机器学习 baseline |
| `PGR_Transformer` | 使用新训练逻辑的 Transformer 版本 |
| `PGR_PrefixStat` | 只加 prefix 统计特征 |
| `PGR_LocalTCN` | 只加局部卷积 |
| `PGR_CalendarGTR` | 只加日历全局检索 |
| `PGR_ProgressGTR` | 只加流程进度检索 |
| `PGR_DualGTR` | Calendar-GTR + Progress-GTR |
| `PGR_TC_MVP` | PrefixStat + LocalTCN + DualGTR + TailWeightedHuber |

这组消融可以回答：

```text
1. prefix 统计特征是否有效？
2. LocalTCN 是否有效？
3. Calendar-GTR 是否有效？
4. Progress-GTR 是否有效？
5. DualGTR 是否比单一 GTR 更稳？
6. PGR-TC 完整模型是否优于 baseline？
```

---

## 8. PGR-TC 快速运行

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
python baseline_pgr_tc.py
```

默认数据集：

```python
DATASET_PATH = "dataset/processed_BPIC2015_1.csv"
```

如果要换数据集，修改 `baseline_pgr_tc.py` 中：

```python
DATASET_PATH = "dataset/processed_BPIC2015_2.csv"
DATASET_PATH = "dataset/processed_Sepsis.csv"
DATASET_PATH = "dataset/processed_Helpdesk.csv"
```

第一次建议小规模测试：

```python
EPOCHS = 1
SEEDS = [42]
MAX_PREFIXES = 20
BATCH_SIZE = 32
```

确认能跑通后，再恢复完整配置。

---

## 9. PGR-TC 输出文件

运行结束后会生成：

| 文件 | 内容 |
| --- | --- |
| `results/pgr_tc_per_seed.csv` | 每个 seed、每个模型的 MAE/RMSE/Tail MAE/Score。 |
| `results/pgr_tc_summary.csv` | 各模型跨 seed 的均值和标准差。 |
| `results/pgr_tc_bucket_metrics.csv` | head/torso/tail 分桶评估结果。 |
| `results/pgr_tc_prefix_stat_feature_names.csv` | prefix 统计特征名称。 |

---

## 10. PGR-TC 当前还没做的内容

当前版本是 MVP，还没有实现：

```text
1. Prefix Temporal Consistency Loss
2. Quantile Loss
3. Variant-aware MoE
4. 独立 models/process_gtr/ 模块化目录
5. argparse 命令行参数
6. 自动保存 best_model.pt
7. test_predictions.csv
```

建议先验证当前 MVP 是否能跑通，再逐步加入这些功能。

---

# 二、DNC-ICA 实验说明

## 项目目标

在 `baseline.py` 的数据读取、前缀样本构建、多随机种子评估和结果导出流程基础上，`baseline_dnc_ica.py` 新增一个面向剩余时间预测的创新模型 `DNCICA`。该模型融合了三类创新：

1. 数据层创新：异常值处理、缺失值标记、滚动统计、EWMA、偏度、峰度等时序特征构建。
2. 模型层创新：局部-全局 DNC 降噪卷积、ICA 自适应通道混合、LSTM 序列骨干。
3. 训练层创新：动态加权 DAL 风格损失，针对稀有变体、长剩余时间样本和高波动样本自适应加权。

## DNC-ICA 快速运行

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

## DNC-ICA 输出文件

运行结束后会生成：

| 文件 | 内容 |
| --- | --- |
| `results/baseline_dnc_ica_per_seed.csv` | 每个 seed、每个模型的 MAE/RMSE/Tail MAE/Score。 |
| `results/baseline_dnc_ica_summary.csv` | 各模型跨 seed 的均值和标准差。 |
| `results/baseline_dnc_ica_bucket_metrics.csv` | head/torso/tail 分桶评估结果。 |
| `results/baseline_dnc_ica_feature_gates.csv` | DNCICA 的特征门控权重。 |

---

# 三、环境要求

建议使用 Python 3.10 或相近版本。当前代码依赖：

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`

验证核心依赖：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
python --version
python -c "import torch, pandas, numpy, sklearn; print(torch.__version__)"
```

---

# 四、数据格式

数据文件需要包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `CaseID` | 流程实例 ID。 |
| `Activity` | 当前事件活动名称。 |
| `Resource` | 当前事件资源/执行者。 |
| `Timestamp` | 事件时间戳，用于按 case 时间划分和构造 calendar bucket。 |
| `TimeSinceLast` | 当前事件距离上一事件的时间间隔。 |
| `TimeSinceStart` | 当前事件距离 case 开始的累计时间。 |
| `Next_Activity` | 下一活动标签，当前脚本主要用于构造前缀。 |
| `Next_Event_Time` | 下一事件时间间隔标签，当前脚本主要用于构造前缀。 |
| `Remaining_Time` | 剩余时间预测目标。 |

---

# 五、论文写作建议

PGR-TC 可以组织成三层贡献：

```text
1. 数据层：提出 prefix-level statistical features，增强对当前流程状态的刻画。
2. 模型层：提出事件对齐全局检索模块，用 Calendar-GTR 和 Progress-GTR 建模全局日历周期与流程阶段原型。
3. 优化层：提出 Tail-weighted Huber Loss，增强模型对长剩余时间和长尾变体样本的关注。
```

推荐中文表述：

```text
本文提出 PGR-TC，一种面向业务流程剩余时间预测的前缀感知全局检索模型。该方法首先从事件前缀中构建等待时间波动、活动重复、资源切换、转移熵和日历周期等统计特征，以刻画当前流程实例的局部执行状态；随后利用 LocalTCN 提取局部事件模式，并通过 Calendar-GTR 与 Progress-GTR 分别检索日历时间原型和流程进度原型；最后采用 Tail-weighted Huber Loss 提升模型对长尾样本和长剩余时间 case 的预测能力。
```

---

# 六、常见问题

## 为什么不直接改 `baseline.py`？

`baseline.py` 是原始对照实验。保持它不变，可以保证创新模型与原始 baseline 的差异清晰、可复现、可解释。

## 为什么默认使用 case 划分？

默认：

```python
SPLIT_STRATEGY = "case"
```

按 case 划分可以避免同一个流程实例的不同前缀同时进入训练集和验证集，评估更严格。

## 如何缩短测试时间？

可以临时调小：

```python
EPOCHS = 1
BATCH_SIZE = 32
SEEDS = [42]
MAX_PREFIXES = 20
```

确认代码能跑通后，再恢复完整实验配置。

## 如果 PGR-TC 跑不通优先检查什么？

```text
1. 数据集文件是否存在。
2. 数据列是否完整。
3. calendar_bucket_ids 和 progress_bucket_ids 是否为 LongTensor。
4. prefix_stat_feats 维度是否等于模型中的 num_prefix_stat_features。
5. GPU 显存是否足够。
```
