# PGR-TC 实现报告：基于事件日志的剩余时间预测改进方案

## 1. 本次新增了什么

本次没有修改原始 `baseline.py`、`model.py`、`prefix.py` 的核心逻辑，而是新增了一个独立实验脚本：

```text
baseline_pgr_tc.py
```

这样做的目的是保留原始 baseline 的可复现性，避免新方法和旧方法混在一起，后续做公平对比和消融实验更清楚。

当前新增脚本实现了一个新的实验方向：

```text
PGR-TC: Prefix-aware Global Retrieval Network with Temporal Consistency-friendly training
```

中文可以理解为：

```text
融合前缀统计特征、局部时序卷积、全局时间检索和长尾加权损失的业务流程剩余时间预测模型
```

它对应三层改进：

```text
1. 数据层：Prefix-level Statistical Features
2. 模型层：LocalTCN + Event-Aligned GTR + Transformer
3. 损失层：log1p + Tail-weighted Huber Loss
```

---

## 2. 为什么要新建 `baseline_pgr_tc.py`

原项目中已经有：

```text
baseline.py
baseline_dnc_ica.py
model.py
prefix.py
training_utils.py
```

其中：

- `baseline.py` 是原始基线实验，包含 LSTM、VanillaTransformer、RandomForest 等对照模型。
- `prefix.py` 负责把事件日志转成 prefix 样本，输出 `act_seq`、`res_seq`、`time_seq`、`mask`、`target_rem_time`、`variant_freq` 等字段。
- `baseline_dnc_ica.py` 是之前的 DNC-ICA 方向实验。

如果直接改 `baseline.py`，后面就很难判断结果提升是来自新模型，还是来自数据处理、训练方式、损失函数的变化。因此本次采用独立脚本：

```text
baseline_pgr_tc.py
```

好处：

```text
1. 原 baseline 保持不变，便于公平对比。
2. 新方法所有代码集中在一个文件里，方便调试和回滚。
3. 后续可以把有效模块再拆分到 models/process_gtr/ 目录。
4. 消融实验更清晰。
```

---

## 3. 数据层改进：EnhancedProcessPrefixDataset

### 3.1 做了什么

新增类：

```python
class EnhancedProcessPrefixDataset(ProcessPrefixDataset)
```

它继承原来的 `ProcessPrefixDataset`，保留原始输出字段，同时增加：

```text
prefix_stat_feats
calendar_bucket_ids
progress_bucket_ids
prefix_len
```

原来的 `ProcessPrefixDataset` 已经能生成：

```text
act_seq
res_seq
time_seq
time_matrix
mask
variant_id
variant_freq
target_act
target_next_time
target_rem_time
```

新 dataset 在此基础上扩展，不破坏旧模型。

### 3.2 为什么要做 prefix 统计特征

普通剩余时间预测模型通常只使用单事件级别的时间特征，例如：

```text
TimeSinceLast
TimeSinceStart
```

但是业务流程剩余时间不仅由最后一个事件决定，还和当前 prefix 的整体状态有关。例如：

```text
1. 当前 case 是否已经出现长时间等待？
2. 当前 case 的等待时间是否波动很大？
3. 当前 case 是否出现活动重复或返工？
4. 当前 case 是否频繁切换资源？
5. 当前 case 是否处于流程早期、中期或后期？
```

这些信息不是单个事件能表达的，所以新增了 prefix-level statistical features。

### 3.3 新增的 prefix 统计特征

当前实现中 `prefix_stat_feats` 包含以下特征：

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

这些特征全部只使用当前 prefix 内已经发生的事件，不使用未来事件，因此不会泄露标签。

### 3.4 这些特征的好处

| 特征类型 | 解决的问题 | 预期好处 |
|---|---|---|
| delta 均值/方差/中位数 | case 当前处理速度 | 增强时间状态表达 |
| EWMA | 最近事件更重要 | 更敏感地反映近期变化 |
| 偏度/峰度 | 极端等待、长尾 | 辅助识别异常拖延 case |
| 活动重复/loop | 返工、循环 | 对流程变体复杂场景更友好 |
| 转移熵 | 当前前缀结构复杂度 | 辅助判断流程是否稳定 |
| 资源切换 | 资源交接、人员变化 | 捕捉组织层面的延迟风险 |
| prefix_len / progress | 流程阶段 | 区分早期、中期、后期预测难度 |
| hour / weekday / weekend | 日历周期 | 捕捉业务时间节律 |

---

## 4. 模型层改进：PGR-TC 模型结构

主模型类：

```python
class PGRTCModel(nn.Module)
```

整体结构：

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

## 5. EventEmbedding：事件多模态嵌入

### 5.1 做了什么

新增：

```python
class EventEmbedding(nn.Module)
```

输入：

```text
act_seq: 活动序列
res_seq: 资源序列
time_seq: TimeSinceLast + TimeSinceStart
```

输出：

```text
[B, L, d_model]
```

内部融合方式：

```text
activity embedding + resource embedding + time projection
```

### 5.2 为什么这么做

事件日志不是单纯数值时间序列，而是多属性事件序列。一个事件至少包含：

```text
activity
resource
time information
```

将三类信息映射到同一维度再融合，可以让后续 LocalTCN、GTR 和 Transformer 在统一表示空间中建模。

### 5.3 好处

```text
1. 保留活动、资源和时间三类信息。
2. 和原有 baseline 输入兼容。
3. 为后续 LocalTCN 和 GTR 提供统一事件表示。
```

---

## 6. LocalTCN：局部时序卷积降噪

### 6.1 做了什么

新增：

```python
class LocalTCN(nn.Module)
```

它使用轻量 Conv1d 在事件前缀上做局部模式提取：

```text
Conv1d → GELU → Dropout → Conv1d → Residual → LayerNorm
```

### 6.2 为什么这么做

业务流程前缀中，最近几个事件往往对剩余时间预测很重要。例如：

```text
1. 最近是否连续等待很久？
2. 最近是否出现返工？
3. 最近是否资源频繁切换？
4. 最近的活动组合是否对应某种局部状态？
```

Transformer 能捕捉全局依赖，但对局部连续模式不一定高效；LocalTCN 可以先提取局部模式、平滑噪声，再交给后续模块处理。

### 6.3 好处

```text
1. 提高局部事件模式建模能力。
2. 降低异常单点时间间隔对模型的干扰。
3. 对短 prefix 更友好。
4. 计算代价低，容易做消融实验。
```

---

## 7. Event-Aligned GTR：事件对齐全局检索

### 7.1 做了什么

新增：

```python
class EventAlignedGTR(nn.Module)
```

它不是照搬规则时间序列 GTR，而是改成事件日志版本：

```text
每个事件根据 bucket_id 检索一个可学习全局原型向量。
```

输入：

```text
h: 当前事件表示 [B, L, D]
bucket_ids: 每个事件的 bucket [B, L]
```

输出：

```text
增强后的事件表示 [B, L, D]
```

内部逻辑：

```text
1. 用 bucket_ids 从 global_memory 中取全局原型 q。
2. 拼接当前事件表示 h 和全局原型 q。
3. 用 Conv1d 融合局部事件表示和全局原型。
4. 用 gate 控制全局信息进入当前表示的强度。
5. 残差连接 + LayerNorm。
```

### 7.2 为什么这么做

原始 GTR 适合规则时间序列，例如电力负荷、风电、光伏等，因为这些数据有固定采样周期。但事件日志是不规则的：

```text
1. case 长度不同；
2. 事件间隔不规则；
3. 不同 case 的活动路径不同；
4. 不能直接用固定周期索引。
```

所以这里把 GTR 思想改成事件对齐版本：

```text
规则时间序列 GTR：时间位置 → 检索全局周期片段
PGR-TC：事件 bucket → 检索全局事件原型
```

### 7.3 好处

```text
1. 保留 GTR 的“全局模式检索”思想。
2. 避免直接照搬规则时间序列假设。
3. 能建模业务流程中的全局时间节律和流程阶段模式。
4. 模块可插拔，方便 Calendar-GTR、Progress-GTR 分别消融。
```

---

## 8. Calendar-GTR：日历周期检索

### 8.1 做了什么

Calendar-GTR 使用：

```text
calendar_bucket = weekday * 24 + hour
```

bucket 范围：

```text
0 ~ 167
```

每个事件根据它发生的星期几和小时检索一个全局日历原型。

### 8.2 为什么这么做

业务流程执行时间常常受日历周期影响：

```text
1. 工作日和周末处理速度不同。
2. 上午、下午、夜间处理速度不同。
3. 节假日前后可能出现积压。
4. 不同时间段资源可用性不同。
```

这些信息不一定能通过活动序列直接学到，因此引入 Calendar-GTR。

### 8.3 好处

```text
1. 显式利用业务时间节律。
2. 对存在工作时间规律的数据集更有帮助。
3. 可以解释模型是否利用了日历周期。
```

---

## 9. Progress-GTR：流程阶段检索

### 9.1 做了什么

Progress-GTR 使用：

```text
progress_bucket = floor(position_index / max_seq_len * 20)
```

bucket 范围：

```text
0 ~ 19
```

注意：这里没有使用真实 case 总长度，也没有使用真实总耗时，避免未来信息泄露。

### 9.2 为什么这么做

剩余时间预测在不同 prefix 阶段难度不同：

```text
1. 早期 prefix 信息少，预测不确定性高。
2. 中期 prefix 可以观察到更多活动路径。
3. 后期 prefix 更接近完成，剩余时间更短。
```

Progress-GTR 让模型学习不同流程阶段的全局原型。

### 9.3 好处

```text
1. 区分早期、中期、后期前缀。
2. 不依赖真实 case 总长度，不泄露未来。
3. 对长流程和复杂流程变体更友好。
```

---

## 10. DualRetrieverFusion：双检索门控融合

### 10.1 做了什么

新增：

```python
class DualRetrieverFusion(nn.Module)
```

它同时使用：

```text
Calendar-GTR
Progress-GTR
```

然后通过 gate 融合：

```text
gate * Calendar-GTR + (1 - gate) * Progress-GTR
```

### 10.2 为什么这么做

不同数据集依赖的信息不同：

```text
1. 有些日志日历周期强，Calendar-GTR 更有用。
2. 有些日志流程阶段明显，Progress-GTR 更有用。
3. 有些日志两者都有用。
```

固定相加会强迫模型使用两种信息，可能引入噪声。门控融合可以让模型自动决定当前事件更依赖哪类全局原型。

### 10.3 好处

```text
1. 比简单相加更灵活。
2. 可以降低无效全局信息的干扰。
3. 后续可以分析 gate，增强可解释性。
```

---

## 11. TransformerEncoder：全局序列建模

### 11.1 做了什么

PGR-TC 仍然保留 TransformerEncoder 作为主干序列编码器：

```text
LocalTCN / GTR 负责增强事件表示
TransformerEncoder 负责建模 prefix 内全局依赖
```

### 11.2 为什么保留 Transformer

因为业务流程前缀中存在长距离依赖：

```text
1. 早期活动可能决定后续路径。
2. 某些关键活动出现后，剩余时间分布会明显变化。
3. 长流程中的远距离活动关系需要全局建模。
```

LocalTCN 只负责局部模式，不能完全替代全局建模。

### 11.3 好处

```text
1. 保留强序列建模能力。
2. 方便和 VanillaTransformer 做公平对比。
3. 新模块的提升可以通过消融清晰体现。
```

---

## 12. PrefixStat 拼接到回归头

### 12.1 做了什么

Transformer 输出最后一个有效事件 hidden state 后，将其与 `prefix_stat_feats` 拼接：

```text
pooled_hidden + prefix_stat_feats → regression head
```

### 12.2 为什么这么做

prefix 统计特征是整个前缀级别的信息，不是单个 token 信息。直接把它拼到回归头前，比强行复制到每个 token 更简单、更稳定。

### 12.3 好处

```text
1. 不改变序列长度。
2. 不污染 token-level 表示。
3. 对 tabular-style prefix 特征更友好。
4. 方便做 PrefixStat 消融。
```

---

## 13. 损失层改进：TailWeightedHuberLoss

### 13.1 做了什么

新增：

```python
class TailWeightedHuberLoss(nn.Module)
```

训练时默认使用：

```text
log1p(target) + Huber Loss + long-tail weight
```

权重形式：

```text
weight = 1 + alpha * log1p(target) / mean(log1p(target))
```

### 13.2 为什么这么做

业务流程剩余时间通常具有长尾分布：

```text
1. 大部分 case 较快完成。
2. 少部分 case 持续时间特别长。
3. 普通 MAE 容易偏向多数普通 case。
4. RMSE 又容易被极端样本过度影响。
```

所以使用 Huber Loss 平衡 MAE/MSE，同时对长剩余时间样本加权。

### 13.3 好处

```text
1. 对极端误差比 MSE 更稳。
2. 比普通 MAE 更关注长尾 case。
3. log1p 压缩目标范围，训练更稳定。
4. 有助于降低长流程、低频变体上的误差。
```

---

## 14. 当前支持的模型消融

`baseline_pgr_tc.py` 当前支持以下模型：

```text
VanillaTransformer
LSTM
RandomForest
PGR_Transformer
PGR_PrefixStat
PGR_LocalTCN
PGR_CalendarGTR
PGR_ProgressGTR
PGR_DualGTR
PGR_TC_MVP
```

含义如下：

| 模型 | 含义 |
|---|---|
| VanillaTransformer | 原项目 Transformer baseline |
| LSTM | 原项目 LSTM baseline |
| RandomForest | 原项目传统机器学习 baseline |
| PGR_Transformer | 使用新训练逻辑的 Transformer 版本 |
| PGR_PrefixStat | 只加 prefix 统计特征 |
| PGR_LocalTCN | 只加局部卷积 |
| PGR_CalendarGTR | 只加日历全局检索 |
| PGR_ProgressGTR | 只加流程进度检索 |
| PGR_DualGTR | Calendar-GTR + Progress-GTR |
| PGR_TC_MVP | PrefixStat + LocalTCN + DualGTR + TailWeightedHuber |

这样可以回答论文里的几个关键问题：

```text
1. prefix 统计特征是否有效？
2. 局部卷积是否有效？
3. Calendar-GTR 是否有效？
4. Progress-GTR 是否有效？
5. DualGTR 是否比单一 GTR 更稳？
6. PGR-TC 完整模型是否优于 baseline？
```

---

## 15. 输出文件

运行后输出：

```text
results/pgr_tc_per_seed.csv
results/pgr_tc_summary.csv
results/pgr_tc_bucket_metrics.csv
results/pgr_tc_prefix_stat_feature_names.csv
```

### 15.1 `pgr_tc_per_seed.csv`

每个 seed、每个模型的结果：

```text
seed
model
mae
rmse
tail_mae
score
mae_short
mae_middle
mae_long
```

### 15.2 `pgr_tc_summary.csv`

跨 seed 均值和标准差。

### 15.3 `pgr_tc_bucket_metrics.csv`

按照 variant frequency 分桶的评估结果：

```text
overall
head
torso
tail
```

### 15.4 `pgr_tc_prefix_stat_feature_names.csv`

记录 prefix 统计特征名称，方便后续解释和写论文。

---

## 16. 如何运行

默认运行：

```bash
python baseline_pgr_tc.py
```

默认数据集：

```python
DATASET_PATH = "dataset/processed_BPIC2015_1.csv"
```

如果要换数据集，修改：

```python
DATASET_PATH = "dataset/processed_BPIC2015_2.csv"
```

或者：

```python
DATASET_PATH = "dataset/processed_Sepsis.csv"
DATASET_PATH = "dataset/processed_Helpdesk.csv"
```

第一次建议先小规模测试：

```python
EPOCHS = 1
SEEDS = [42]
MAX_PREFIXES = 20
BATCH_SIZE = 32
```

确认能跑通后，再恢复完整配置。

---

## 17. 当前还没做什么

当前版本还没有实现：

```text
1. Prefix Temporal Consistency Loss
2. Quantile Loss
3. Variant-aware MoE
4. 独立 models/process_gtr/ 模块化目录
5. 命令行参数 argparse
6. 自动保存 best_model.pt
7. test_predictions.csv
```

原因是当前阶段先做 MVP，优先验证：

```text
PrefixStat + LocalTCN + DualGTR + TailWeightedHuber
```

是否能在现有数据管线上跑通并产生稳定对比结果。

---

## 18. 后续建议

### 18.1 第一轮

先跑：

```text
EPOCHS = 1
SEEDS = [42]
MAX_PREFIXES = 20
```

检查是否报错。

### 18.2 第二轮

跑完整 Helpdesk 或 BPIC2015_2：

```text
SEEDS = [42, 67, 80, 89]
EPOCHS = 30
```

观察：

```text
PGR_TC_MVP 是否优于 VanillaTransformer、LSTM、RandomForest。
```

### 18.3 第三轮

扩展多个数据集：

```text
BPIC2015_1
BPIC2015_2
BPIC2015_3
BPIC2015_4
BPIC2015_5
Sepsis
Helpdesk
```

### 18.4 第四轮

增加 Prefix Temporal Consistency Loss：

```text
pred_{k-1} - pred_k ≈ delta_t_k
```

这是后续最有论文味道的训练层创新。

---

## 19. 论文中可以怎么表述

本文针对业务流程剩余时间预测中前缀局部状态刻画不足、全局业务时间模式利用不足以及长尾样本预测不稳定的问题，提出 PGR-TC。首先，从事件前缀中构建统计特征，包括等待时间均值、波动性、EWMA、偏度、峰度、活动重复、资源切换和转移熵，以增强对当前流程状态的表示。其次，设计 LocalTCN 提取局部事件模式，并提出事件对齐全局检索模块，通过 Calendar-GTR 和 Progress-GTR 分别建模日历周期原型和流程阶段原型。最后，采用 log1p Tail-weighted Huber Loss，提高模型对长剩余时间样本和长尾流程变体的关注。实验通过多模型消融验证各模块对剩余时间预测性能的贡献。

---

## 20. 注意事项

当前代码已经提交，但尚未在本地或服务器完成实际训练验证。第一次运行后如果出现报错，应优先检查：

```text
1. 数据集文件名是否存在。
2. 数据集中是否包含 CaseID、Activity、Resource、Timestamp、TimeSinceLast、TimeSinceStart、Next_Activity、Next_Event_Time、Remaining_Time。
3. GPU 显存是否足够。
4. PGR-TC 中 prefix_stat_feats 维度是否和 head_in 一致。
5. calendar_bucket_ids 和 progress_bucket_ids 是否为 long tensor。
```

如果能跑通，再根据实验结果判断是否保留所有模块，或进一步简化模型。
