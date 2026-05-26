"""
数据集特性分析脚本：
对 dataset/ 目录下的所有预处理数据集进行全面特性剖析，
输出统计指标用于数据集分类和模型选择。
"""
import pandas as pd
import numpy as np
import os
import json
from collections import Counter
from scipy.stats import entropy as sp_entropy

DATASET_DIR = "dataset"


def analyze_dataset(file_path):
    """对单个数据集进行全面特性分析"""
    name = os.path.basename(file_path).replace("processed_", "").replace(".csv", "")
    df = pd.read_csv(file_path)

    result = {"name": name, "file": os.path.basename(file_path)}

    # ======== 1. 基础规模指标 ========
    num_events = len(df)
    num_cases = df['CaseID'].nunique()
    num_activities = df['Activity'].nunique()
    num_resources = df['Resource'].nunique() if 'Resource' in df.columns else 0

    result["num_events"] = num_events
    result["num_cases"] = num_cases
    result["num_activities"] = num_activities
    result["num_resources"] = num_resources
    result["events_per_case_ratio"] = round(num_events / num_cases, 2)

    # ======== 2. Trace 长度分布 ========
    trace_lengths = df.groupby('CaseID').size()
    result["trace_len_min"] = int(trace_lengths.min())
    result["trace_len_max"] = int(trace_lengths.max())
    result["trace_len_mean"] = round(trace_lengths.mean(), 2)
    result["trace_len_median"] = round(trace_lengths.median(), 2)
    result["trace_len_std"] = round(trace_lengths.std(), 2)
    result["trace_len_cv"] = round(trace_lengths.std() / trace_lengths.mean(), 4) if trace_lengths.mean() > 0 else 0

    # 分位数
    result["trace_len_p25"] = round(trace_lengths.quantile(0.25), 2)
    result["trace_len_p75"] = round(trace_lengths.quantile(0.75), 2)
    result["trace_len_p90"] = round(trace_lengths.quantile(0.90), 2)
    result["trace_len_p95"] = round(trace_lengths.quantile(0.95), 2)

    # ======== 3. 剩余时间分布 (预测目标) ========
    rt = df['Remaining_Time']
    result["rt_min_hours"] = round(rt.min(), 4)
    result["rt_max_hours"] = round(rt.max(), 4)
    result["rt_mean_hours"] = round(rt.mean(), 4)
    result["rt_median_hours"] = round(rt.median(), 4)
    result["rt_std_hours"] = round(rt.std(), 4)
    result["rt_cv"] = round(rt.std() / rt.mean(), 4) if rt.mean() > 0 else 0
    result["rt_skewness"] = round(rt.skew(), 4)
    result["rt_kurtosis"] = round(rt.kurtosis(), 4)

    # 转换为天
    result["rt_mean_days"] = round(rt.mean() / 24.0, 2)
    result["rt_median_days"] = round(rt.median() / 24.0, 2)
    result["rt_max_days"] = round(rt.max() / 24.0, 2)

    # RT = 0 的比例 (最后一个事件)
    result["rt_zero_ratio"] = round((rt == 0).sum() / len(rt), 4)

    # RT 分位数
    result["rt_p25_hours"] = round(rt.quantile(0.25), 2)
    result["rt_p75_hours"] = round(rt.quantile(0.75), 2)
    result["rt_p90_hours"] = round(rt.quantile(0.90), 2)

    # ======== 4. 时间间隔分析 ========
    tsl = df['TimeSinceLast']
    result["time_since_last_mean_hours"] = round(tsl.mean(), 4)
    result["time_since_last_median_hours"] = round(tsl.median(), 4)
    result["time_since_last_max_hours"] = round(tsl.max(), 4)
    result["time_since_last_std_hours"] = round(tsl.std(), 4)

    # ======== 5. 活动分布特性 ========
    act_counts = df['Activity'].value_counts(normalize=True)
    result["activity_entropy"] = round(sp_entropy(act_counts.values, base=2), 4)
    result["activity_max_entropy"] = round(np.log2(num_activities), 4) if num_activities > 1 else 0
    result["activity_normalized_entropy"] = round(
        sp_entropy(act_counts.values, base=2) / np.log2(num_activities), 4
    ) if num_activities > 1 else 0

    # Top-5 活动占比
    top5_ratio = act_counts.head(5).sum()
    result["top5_activity_ratio"] = round(top5_ratio, 4)

    # 活动的唯一 trace variant 数量
    trace_variants = df.groupby('CaseID')['Activity'].apply(lambda x: '->'.join(x.astype(str)))
    num_variants = trace_variants.nunique()
    result["num_trace_variants"] = num_variants
    result["variant_case_ratio"] = round(num_variants / num_cases, 4)

    # Top-5 变体覆盖率
    variant_counts = trace_variants.value_counts(normalize=True)
    top5_variant_ratio = variant_counts.head(5).sum()
    result["top5_variant_coverage"] = round(top5_variant_ratio, 4)

    # ======== 6. 控制流复杂度 ========
    # 直接跟随关系数量 (DFG edge count)
    transitions = set()
    for _, group in df.groupby('CaseID'):
        acts = group['Activity'].tolist()
        for i in range(len(acts) - 1):
            transitions.add((acts[i], acts[i + 1]))
    result["dfg_edge_count"] = len(transitions)
    result["dfg_density"] = round(len(transitions) / (num_activities * num_activities), 4) if num_activities > 0 else 0

    # 自循环活动数量 (同一活动连续出现)
    self_loops = sum(1 for (a, b) in transitions if a == b)
    result["self_loop_count"] = self_loops

    # ======== 7. 资源分布 ========
    if 'Resource' in df.columns:
        res_counts = df['Resource'].value_counts(normalize=True)
        result["resource_entropy"] = round(sp_entropy(res_counts.values, base=2), 4)
        result["resource_per_case"] = round(
            df.groupby('CaseID')['Resource'].nunique().mean(), 2
        )
        unknown_ratio = (df['Resource'] == 'UNKNOWN').sum() / len(df)
        result["resource_unknown_ratio"] = round(unknown_ratio, 4)
    else:
        result["resource_entropy"] = 0
        result["resource_per_case"] = 0
        result["resource_unknown_ratio"] = 1.0

    # ======== 8. 时间跨度 ========
    if 'Timestamp' in df.columns:
        df['_ts'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        valid_ts = df['_ts'].dropna()
        if len(valid_ts) > 0:
            result["time_span_days"] = round(
                (valid_ts.max() - valid_ts.min()).total_seconds() / 86400.0, 2
            )
            # 检查是否有明显的周期性 (工作日/休息日)
            dow_counts = valid_ts.dt.dayofweek.value_counts(normalize=True)
            result["weekday_ratio"] = round(dow_counts[dow_counts.index < 5].sum(), 4) if len(dow_counts) > 0 else 0

            # 工作时间分布 (9-17点)
            hour_counts = valid_ts.dt.hour.value_counts(normalize=True).sort_index()
            business_hour_ratio = hour_counts[(hour_counts.index >= 9) & (hour_counts.index < 17)].sum()
            result["business_hour_ratio"] = round(business_hour_ratio, 4)
        else:
            result["time_span_days"] = 0
            result["weekday_ratio"] = 0
            result["business_hour_ratio"] = 0
        df.drop(columns=['_ts'], inplace=True)
    else:
        result["time_span_days"] = 0
        result["weekday_ratio"] = 0
        result["business_hour_ratio"] = 0

    # ======== 9. 目标分布形态 (用于模型选择) ========
    # log1p 后的偏度 (判断是否适合 log 变换)
    rt_log = np.log1p(rt.values)
    result["rt_log_skewness"] = round(float(pd.Series(rt_log).skew()), 4)
    result["rt_log_kurtosis"] = round(float(pd.Series(rt_log).kurtosis()), 4)

    # 零值膨胀程度
    result["rt_near_zero_ratio"] = round((rt < 1.0).sum() / len(rt), 4)  # < 1小时

    return result


def classify_datasets(all_stats):
    """根据分析结果对数据集进行分类 (已优化评分与平局打破机制)"""
    for ds in all_stats:
        tags = []

        # --- 规模分类 ---
        if ds["num_events"] < 5000:
            tags.append("SMALL_SCALE")
        elif ds["num_events"] < 50000:
            tags.append("MEDIUM_SCALE")
        else:
            tags.append("LARGE_SCALE")

        # --- Trace 长度分类 ---
        if ds["trace_len_mean"] <= 5:
            tags.append("SHORT_TRACE")
        elif ds["trace_len_mean"] <= 15:
            tags.append("MEDIUM_TRACE")
        else:
            tags.append("LONG_TRACE")

        # --- Trace 长度变异性 ---
        if ds["trace_len_cv"] > 0.6:
            tags.append("HIGH_LENGTH_VARIANCE")
        else:
            tags.append("LOW_LENGTH_VARIANCE")

        # --- 控制流复杂度 ---
        if ds["num_activities"] <= 10:
            tags.append("SIMPLE_FLOW")
        elif ds["num_activities"] <= 30:
            tags.append("MODERATE_FLOW")
        else:
            tags.append("COMPLEX_FLOW")

        # --- 变体多样性 (新增极端变异检测) ---
        # 变体比例极高说明流程高度非结构化
        if ds["variant_case_ratio"] > 0.9:
            tags.append("EXTREME_VARIABILITY")
        elif ds["variant_case_ratio"] > 0.5:
            tags.append("HIGH_VARIABILITY")
        elif ds["variant_case_ratio"] > 0.1:
            tags.append("MODERATE_VARIABILITY")
        else:
            tags.append("LOW_VARIABILITY")

        # --- 目标分布特性 ---
        if ds["rt_cv"] > 1.5:
            tags.append("HEAVY_TAIL_RT")
        else:
            tags.append("MODERATE_TAIL_RT")

        if ds["rt_skewness"] > 2.0:
            tags.append("RIGHT_SKEWED_RT")

        if ds["rt_near_zero_ratio"] > 0.3:
            tags.append("ZERO_INFLATED_RT")

        # --- 时间粒度 ---
        if ds["rt_max_days"] < 1:
            tags.append("INTRADAY_PROCESS")
        elif ds["rt_max_days"] < 30:
            tags.append("SHORT_DURATION_PROCESS")
        else:
            tags.append("LONG_DURATION_PROCESS")

        # --- 周期性与资源 ---
        if ds["business_hour_ratio"] > 0.7:
            tags.append("BUSINESS_HOUR_DOMINANT")

        if ds["resource_unknown_ratio"] > 0.5:
            tags.append("WEAK_RESOURCE_INFO")
        else:
            tags.append("RICH_RESOURCE_INFO")

        ds["tags"] = tags

        # === 综合分类计算 ===
        # 使用浮点数可更精细地控制特征权重
        score_a = 0.0
        score_b = 0.0
        score_c = 0.0

        # 规模：大规模本身不决定必须用复杂模型，需结合变异性
        if "SMALL_SCALE" in tags: score_a += 1.0
        if "LARGE_SCALE" in tags: score_c += 1.5

        # Trace长度
        if "SHORT_TRACE" in tags: score_a += 2.0
        if "LONG_TRACE" in tags: score_b += 2.0
        if "MEDIUM_TRACE" in tags: score_b += 1.0

        # 控制流复杂度：图结构越复杂，越倾向于Type C
        if "SIMPLE_FLOW" in tags: score_a += 2.0
        if "COMPLEX_FLOW" in tags:
            score_b += 1.0
            score_c += 1.5
        if "MODERATE_FLOW" in tags: score_b += 1.0

        # 变异性 (核心区分特征)
        if "LOW_VARIABILITY" in tags: score_a += 2.0
        if "MODERATE_VARIABILITY" in tags: score_b += 1.5
        if "HIGH_VARIABILITY" in tags:
            score_b += 1.0
            score_c += 1.0
        if "EXTREME_VARIABILITY" in tags:
            score_c += 3.0  # 给予极高的惩罚分，直接推向 Type C

        # 目标分布：长尾分布需要更强的模型泛化能力
        if "HEAVY_TAIL_RT" in tags:
            score_b += 0.5
            score_c += 1.0

        scores = {"Type_A": score_a, "Type_B": score_b, "Type_C": score_c}
        ds["classification_scores"] = scores

        # === 平局打破机制 (Tie-breaker) ===
        # 找出最高分
        max_score = max(scores.values())
        # 获取所有获得最高分的候选分类
        candidate_types = [k for k, v in scores.items() if v == max_score]

        # 优先级路由：遇到模棱两可的数据集，优先归类到更高复杂度的类别
        if "Type_C" in candidate_types:
            ds["primary_type"] = "Type_C"
        elif "Type_B" in candidate_types:
            ds["primary_type"] = "Type_B"
        else:
            ds["primary_type"] = "Type_A"

    return all_stats


def main():
    csv_files = sorted([
        os.path.join(DATASET_DIR, f)
        for f in os.listdir(DATASET_DIR)
        if f.endswith('.csv') and f.startswith('processed_')
    ])

    print(f"找到 {len(csv_files)} 个数据集，开始分析...\n")

    all_stats = []
    for fpath in csv_files:
        print(f"  分析中: {os.path.basename(fpath)} ...", end=" ")
        try:
            stats = analyze_dataset(fpath)
            all_stats.append(stats)
            print(f"OK (events={stats['num_events']}, cases={stats['num_cases']})")
        except Exception as e:
            print(f"ERROR: {e}")

    # 分类
    all_stats = classify_datasets(all_stats)

    # 保存 JSON
    output_json = "dataset_analysis_results.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\n分析结果已保存至: {output_json}")

    # 打印摘要表
    print("\n" + "=" * 150)
    print(f"{'Dataset':<22} | {'Events':>7} | {'Cases':>6} | {'Acts':>4} | {'AvgLen':>6} | {'MaxLen':>6} | "
          f"{'RT Mean(d)':>10} | {'RT Max(d)':>9} | {'Variants':>8} | {'VarRatio':>8} | {'Type':>6}")
    print("-" * 150)
    for ds in all_stats:
        print(f"{ds['name']:<22} | {ds['num_events']:>7} | {ds['num_cases']:>6} | {ds['num_activities']:>4} | "
              f"{ds['trace_len_mean']:>6} | {ds['trace_len_max']:>6} | {ds['rt_mean_days']:>10} | {ds['rt_max_days']:>9} | "
              f"{ds['num_trace_variants']:>8} | {ds['variant_case_ratio']:>8} | {ds['primary_type']:>6}")

    # 按分类分组打印
    type_groups = {}
    for ds in all_stats:
        t = ds['primary_type']
        if t not in type_groups:
            type_groups[t] = []
        type_groups[t].append(ds['name'])

    print("\n数据集分类结果:")
    type_names = {
        "Type_A": "结构化短流程 (短trace/少活动/低变异 → 适合轻量模型)",
        "Type_B": "复杂长流程 (长trace/多活动/高复杂度 → 需要强序列建模能力)",
        "Type_C": "大规模高变异流程 (数据量大/变体多 → 需要强泛化+图建模能力)"
    }
    for t, datasets in sorted(type_groups.items()):
        print(f"\n  {t} - {type_names.get(t, t)}:")
        for d in datasets:
            print(f"    - {d}")


if __name__ == '__main__':
    main()
