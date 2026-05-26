# -*- coding: utf-8 -*-
import math
import os
import random
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
except Exception:
    torch = None
    nn = None

    class Dataset:  # type: ignore[override]
        pass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors.*")

try:
    import lightgbm as lgb
except Exception:
    lgb = None


PAPER_DATASET_CONFIGS = {
    "bpic2015": {
        "repo_columns": [
            ("case", ["case", "case:concept:name"]),
            ("startTime", ["startTime", "completeTime", "time:timestamp"]),
            ("completeTime", ["completeTime", "time:timestamp"]),
            ("event", ["event", "concept:name", "Activity"]),
            ("caseStatus", ["caseStatus", "case:caseStatus"]),
            ("SUMleges", ["SUMleges", "case:SUMleges"]),
            ("last_phase", ["last_phase", "case:last_phase"]),
            ("Includes_subCases", ["Includes_subCases", "case:Includes_subCases"]),
            ("Responsible_actor", ["Responsible_actor", "case:Responsible_actor"]),
            ("landRegisterID", ["landRegisterID", "case:landRegisterID"]),
            ("caseProcedure", ["caseProcedure", "case:caseProcedure"]),
            ("parts", ["parts", "case:parts"]),
            ("termName", ["termName", "case:termName"]),
            ("requestComplete", ["requestComplete", "case:requestComplete"]),
            ("IDofConceptCase", ["IDofConceptCase", "case:IDofConceptCase"]),
            ("activityNameEN", ["activityNameEN"]),
            ("monitoringResource", ["monitoringResource"]),
            ("activityNameNL", ["activityNameNL"]),
            ("resource", ["resource", "org:resource", "Resource"]),
        ],
        "categorical_original_indices": list(range(3, 19)),
    },
    "hd": {
        "repo_columns": [
            ("case", ["case", "Case ID"]),
            ("startTime", ["startTime", "Complete Timestamp", "Complete Timestamp.1"]),
            ("completeTime", ["completeTime", "Complete Timestamp.1", "Complete Timestamp"]),
            ("event", ["event", "Activity"]),
            ("Resource", ["Resource", "resource"]),
            ("seriousness", ["seriousness"]),
            ("customer", ["customer"]),
            ("product", ["product"]),
            ("responsible_section", ["responsible_section"]),
            ("seriousness_2", ["seriousness_2"]),
            ("service_level", ["service_level"]),
            ("service_type", ["service_type"]),
            ("support_section", ["support_section"]),
            ("workgroup", ["workgroup"]),
        ],
        "categorical_original_indices": list(range(3, 14)),
    },
    "sepsis": {
        "repo_columns": [
            ("case", ["case", "Case ID"]),
            ("startTime", ["startTime", "Complete Timestamp", "time:timestamp"]),
            ("completeTime", ["completeTime", "Complete Timestamp.1", "Complete Timestamp", "time:timestamp"]),
            ("event", ["event", "Activity", "concept:name"]),
            ("CRP", ["CRP"]),
            ("InfectionSuspected", ["InfectionSuspected"]),
            ("org:group", ["org:group"]),
            ("DiagnosticBlood", ["DiagnosticBlood"]),
            ("SIRSCritTachypnea", ["SIRSCritTachypnea"]),
            ("DisfuncOrg", ["DisfuncOrg"]),
            ("Hypotensie", ["Hypotensie"]),
            ("SIRSCritHeartRate", ["SIRSCritHeartRate"]),
            ("Infusion", ["Infusion"]),
            ("Leucocytes", ["Leucocytes"]),
            ("DiagnosticArtAstrup", ["DiagnosticArtAstrup"]),
            ("LacticAcid", ["LacticAcid"]),
            ("DiagnosticIC", ["DiagnosticIC"]),
            ("Age", ["Age"]),
            ("DiagnosticSputum", ["DiagnosticSputum"]),
            ("DiagnosticLiquor", ["DiagnosticLiquor"]),
            ("DiagnosticOther", ["DiagnosticOther"]),
            ("SIRSCriteria2OrMore", ["SIRSCriteria2OrMore"]),
            ("DiagnosticXthorax", ["DiagnosticXthorax"]),
            ("SIRSCritTemperature", ["SIRSCritTemperature"]),
            ("DiagnosticUrinaryCulture", ["DiagnosticUrinaryCulture"]),
            ("SIRSCritLeucos", ["SIRSCritLeucos"]),
            ("Oligurie", ["Oligurie"]),
            ("DiagnosticLacticAcid", ["DiagnosticLacticAcid"]),
            ("Hypoxie", ["Hypoxie"]),
            ("Diagnose", ["Diagnose"]),
            ("DiagnosticECG", ["DiagnosticECG"]),
            ("DiagnosticUrinarySediment", ["DiagnosticUrinarySediment"]),
        ],
        "categorical_original_indices": [
            3,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            14,
            16,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
        ],
    },
}


@dataclass
class FPMFeatureArtifact:
    train_dataset: Dataset
    test_dataset: Dataset
    selected_features: List[str]
    selected_feature_indices: List[int]
    selected_feature_states: List[int]
    selector_valid_mae: float
    selected_feature_path: str
    input_dim: int
    collate_fn: object
    model_metadata: Dict[str, object]
    split_summary: Dict[str, object]


class PaperTraceDataset(Dataset):
    def __init__(self, traces: Sequence[Sequence[Sequence[float]]], feature_indices: Sequence[int]):
        self.traces = traces
        self.feature_indices = list(feature_indices)
        self.variant_keys = [tuple(int(event[0]) for event in trace) for trace in self.traces]
        variant_counts = Counter(self.variant_keys)
        self.variant_freqs = [float(variant_counts[key]) for key in self.variant_keys]

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        if torch is None:
            raise RuntimeError("FPM 训练数据集需要 torch，但当前环境未安装。")

        trace = self.traces[idx]
        feature_rows = []
        target_rows = []
        for event in trace:
            feature_rows.append([float(event[i]) for i in self.feature_indices])
            target_rows.append(float(event[-1]))

        return {
            "feature_seq": torch.tensor(feature_rows, dtype=torch.float32),
            "target_seq": torch.tensor(target_rows, dtype=torch.float32),
            "variant_freq": torch.tensor(self.variant_freqs[idx], dtype=torch.float32),
            "length": len(feature_rows),
        }


def require_lightgbm():
    if lgb is None:
        raise RuntimeError(
            "FPM 特征选择需要 lightgbm，但当前环境未安装。请先执行：python -m pip install lightgbm"
        )


def require_torch():
    if torch is None or nn is None:
        raise RuntimeError(
            "FPM 训练需要 torch，但当前环境未安装。请先在你的训练环境中安装 PyTorch。"
        )


def resolve_raw_dataset_path(processed_dataset_path):
    name = os.path.basename(processed_dataset_path)
    if name.startswith("processed_"):
        name = name[len("processed_"):]
    processed_dir = os.path.dirname(os.path.abspath(processed_dataset_path)) or os.getcwd()
    candidates = [
        os.path.join("data", name),
        os.path.join(processed_dir, name),
        os.path.join(os.path.dirname(processed_dir), "data", name),
        os.path.join(os.getcwd(), "data", name),
    ]
    for guess in candidates:
        if os.path.exists(guess):
            return guess
    return None


def detect_paper_dataset_key(dataset_name: str) -> str:
    name = str(dataset_name).lower()
    if "bpic2015" in name:
        return "bpic2015"
    if name in {"hd", "helpdesk"} or "helpdesk" in name:
        return "hd"
    if "sepsis" in name:
        return "sepsis"
    raise ValueError(
        f"当前仅实现了论文源码可对齐的数据布局映射，暂不支持数据集: {dataset_name}"
    )


def _pick_first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _normalize_categorical_value(value) -> str:
    if pd.isna(value):
        return "null"
    value = str(value).strip()
    return value if value else "null"


def _to_repo_style_dataframe(raw_df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    config = PAPER_DATASET_CONFIGS[dataset_key]
    repo_cols = []
    for target_name, candidates in config["repo_columns"]:
        source_name = _pick_first_existing(raw_df.columns, candidates)
        if source_name is None:
            repo_cols.append(pd.Series(["null"] * len(raw_df), name=target_name))
            continue
        repo_cols.append(raw_df[source_name].rename(target_name))

    df = pd.concat(repo_cols, axis=1)
    df["case"] = df["case"].astype(str)
    df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce", utc=True)
    df["completeTime"] = pd.to_datetime(df["completeTime"], errors="coerce", utc=True)
    df = df.dropna(subset=["case", "completeTime"]).copy()
    df["startTime"] = df["startTime"].fillna(df["completeTime"])
    df = df.sort_values(["case", "completeTime"]).reset_index(drop=True)
    return df


def _is_processed_event_log(df: pd.DataFrame) -> bool:
    required = {
        "CaseID",
        "Activity",
        "Timestamp",
        "Resource",
        "TimeSinceLast",
        "TimeSinceStart",
        "Next_Activity",
        "Next_Event_Time",
        "Remaining_Time",
    }
    return required.issubset(set(df.columns))


def _build_categorical_vocab(values: Sequence[object]):
    normalized = [_normalize_categorical_value(value) for value in values]
    uniques = sorted({value for value in normalized if value != "null"})
    vocab = {value: idx + 1 for idx, value in enumerate(uniques)}
    vocab["null"] = 0
    return vocab


def _load_processed_event_log(processed_df: pd.DataFrame, dataset_name: str):
    df = processed_df.copy()
    df["CaseID"] = df["CaseID"].astype(str)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["CaseID", "Activity", "Timestamp"]).copy()
    df = df.sort_values(["CaseID", "Timestamp"]).reset_index(drop=True)

    activity_vocab = _build_categorical_vocab(
        pd.concat([df["Activity"], df["Next_Activity"]], ignore_index=True).tolist()
    )
    resource_vocab = _build_categorical_vocab(df["Resource"].tolist())

    encoded_df = df.copy()
    encoded_df["Activity"] = encoded_df["Activity"].map(_normalize_categorical_value).map(activity_vocab).fillna(0).astype(float)
    encoded_df["Resource"] = encoded_df["Resource"].map(_normalize_categorical_value).map(resource_vocab).fillna(0).astype(float)
    encoded_df["nextEvent"] = encoded_df["Next_Activity"].map(_normalize_categorical_value).map(activity_vocab).fillna(0).astype(float)
    encoded_df["nextDuration"] = pd.to_numeric(encoded_df["Next_Event_Time"], errors="coerce").fillna(0.0).astype(float)
    encoded_df["remaining"] = pd.to_numeric(encoded_df["Remaining_Time"], errors="coerce").fillna(0.0).astype(float)
    encoded_df["TimeSinceLast"] = pd.to_numeric(encoded_df["TimeSinceLast"], errors="coerce").fillna(0.0).astype(float)
    encoded_df["TimeSinceStart"] = pd.to_numeric(encoded_df["TimeSinceStart"], errors="coerce").fillna(0.0).astype(float)
    encoded_df["month"] = encoded_df["Timestamp"].dt.month.astype(float)
    encoded_df["day"] = encoded_df["Timestamp"].dt.day.astype(float)
    encoded_df["week"] = encoded_df["Timestamp"].dt.dayofweek.astype(float)
    encoded_df["hour"] = encoded_df["Timestamp"].dt.hour.astype(float)
    encoded_df["year"] = (encoded_df["Timestamp"].dt.year - 2000).astype(float)

    feature_names = [
        "Activity",
        "Resource",
        "TimeSinceLast",
        "TimeSinceStart",
        "month",
        "day",
        "week",
        "hour",
        "year",
    ]
    header = feature_names + ["nextEvent", "nextDuration", "remaining"]
    categorical_indices = [0, 1]
    vocabularies = [
        {idx: value for value, idx in activity_vocab.items()},
        {idx: value for value, idx in resource_vocab.items()},
    ] + [{} for _ in range(len(feature_names) - 2)]

    traces = []
    trace_times = []
    for _, group in encoded_df.groupby("CaseID", sort=False):
        trace = []
        for _, row in group.iterrows():
            event = [row[name] for name in feature_names]
            event.extend([row["nextEvent"], row["nextDuration"], row["remaining"]])
            trace.append(event)
        if trace:
            traces.append(trace)
            trace_times.append(group["Timestamp"].iloc[-1])

    state = _build_feature_state(feature_names, feature_names, categorical_indices, traces)
    return {
        "dataset_key": "processed",
        "source_schema": "processed",
        "dataset_name": dataset_name,
        "header": header,
        "feature_names": feature_names,
        "traces": traces,
        "trace_times": trace_times,
        "state": state,
        "vocabularies": vocabularies,
        "categorical_indices": categorical_indices,
        "target_scale": 1.0,
        "time_unit": "hour",
    }


def load_paper_event_log(raw_path: str, dataset_name: str):
    raw_df = pd.read_csv(raw_path, low_memory=False)
    if _is_processed_event_log(raw_df):
        return _load_processed_event_log(raw_df, dataset_name=dataset_name)

    dataset_key = detect_paper_dataset_key(dataset_name)
    paper_df = _to_repo_style_dataframe(raw_df, dataset_key)
    config = PAPER_DATASET_CONFIGS[dataset_key]

    repo_column_names = [column for column, _ in config["repo_columns"]]
    original_feature_names = repo_column_names[3:]
    categorical_indices = {
        index - 3
        for index in config["categorical_original_indices"]
        if index >= 3
    }

    vocabularies = []
    encoded_df = paper_df.copy()
    for feature_pos, column_name in enumerate(original_feature_names):
        if feature_pos in categorical_indices:
            values = encoded_df[column_name].map(_normalize_categorical_value)
            uniques = sorted({value for value in values.tolist() if value != "null"})
            vocab = {value: idx + 1 for idx, value in enumerate(uniques)}
            vocab["null"] = 0
            vocabularies.append({idx: value for value, idx in vocab.items()})
            encoded_df[column_name] = values.map(vocab).astype(float)
        else:
            numeric = pd.to_numeric(encoded_df[column_name], errors="coerce").fillna(0.0)
            max_value = float(numeric.max()) if not numeric.empty else 0.0
            if max_value > 0:
                numeric = numeric / max_value
            encoded_df[column_name] = numeric.astype(float)

    time_features = _build_time_features(encoded_df)
    encoded_df = pd.concat([encoded_df, time_features], axis=1)
    encoded_df = _append_paper_labels(encoded_df)

    feature_names = original_feature_names + [
        "duration",
        "allDuration",
        "month",
        "day",
        "week",
        "hour",
        "year",
    ]
    header = feature_names + ["nextEvent", "nextDuration", "remaining"]

    traces = []
    for _, group in encoded_df.groupby("case", sort=False):
        trace = []
        for _, row in group.iterrows():
            event = [row[name] for name in feature_names]
            event.extend([row["nextEvent"], row["nextDuration"], row["remaining"]])
            trace.append(event)
        traces.append(trace)

    trace_times = [group["completeTime"].iloc[-1] for _, group in encoded_df.groupby("case", sort=False)]

    state = _build_feature_state(feature_names, original_feature_names, categorical_indices, traces)
    return {
        "dataset_key": dataset_key,
        "header": header,
        "feature_names": feature_names,
        "traces": traces,
        "trace_times": trace_times,
        "state": state,
        "vocabularies": vocabularies,
        "categorical_indices": sorted(categorical_indices),
        "target_scale": 24.0,
        "time_unit": "day",
    }


def _build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    temp = df[["case", "completeTime"]].copy()
    temp["prev_complete_time"] = temp.groupby("case")["completeTime"].shift(1)
    temp["duration"] = (
        (temp["completeTime"] - temp["prev_complete_time"]).dt.total_seconds().fillna(0.0) / 86400.0
    )
    temp["allDuration"] = temp.groupby("case")["duration"].cumsum()
    temp["month"] = temp["completeTime"].dt.month.astype(float)
    temp["day"] = temp["completeTime"].dt.day.astype(float)
    temp["week"] = temp["completeTime"].dt.dayofweek.astype(float)
    temp["hour"] = temp["completeTime"].dt.hour.astype(float)
    temp["year"] = (temp["completeTime"].dt.year - 2000).astype(float)
    return temp[["duration", "allDuration", "month", "day", "week", "hour", "year"]]


def _append_paper_labels(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["nextEvent"] = result.groupby("case")["event"].shift(-1).fillna(0).astype(float)
    result["nextDuration"] = result.groupby("case")["duration"].shift(-1).fillna(0.0).astype(float)
    result["remaining"] = (
        result.groupby("case")["duration"].transform(lambda series: series.iloc[::-1].cumsum().iloc[::-1].shift(-1).fillna(0.0))
    )
    result["remaining"] = result["remaining"].astype(float)
    return result


def _build_feature_state(
    feature_names: Sequence[str],
    original_feature_names: Sequence[str],
    categorical_indices: Sequence[int],
    traces: Sequence[Sequence[Sequence[float]]],
) -> List[int]:
    state = [0]
    original_feature_count = len(original_feature_names)
    for feature_idx in range(1, original_feature_count):
        state.append(1 if feature_idx in categorical_indices else 3)
    state.extend([3] * (len(feature_names) - original_feature_count))

    for trace in traces:
        for feature_idx in range(1, len(state)):
            first_value = trace[0][feature_idx]
            for event in trace[1:]:
                if event[feature_idx] != first_value:
                    if state[feature_idx] in (1, 3):
                        state[feature_idx] += 1
                    break
    return state


def split_traces_with_paper_strategy(
    traces: Sequence[Sequence[Sequence[float]]],
    trace_times: Sequence[pd.Timestamp],
):
    paired = list(zip(traces, trace_times))
    paired.sort(key=lambda item: item[1])
    sorted_traces = [trace for trace, _ in paired]

    trace_num = len(sorted_traces)
    if trace_num < 2:
        raise ValueError(f"FPM 至少需要 2 条 trace 才能划分训练/测试集，当前 trace_count={trace_num}。")

    sec_num = int(trace_num / 25)
    if sec_num == 0:
        split_idx = max(1, int(trace_num * 0.8))
        split_idx = min(split_idx, trace_num - 1)
        train_traces = sorted_traces[:split_idx]
        test_traces = sorted_traces[split_idx:]
        return train_traces, test_traces, {
            "trace_count": trace_num,
            "sec_num": sec_num,
            "train_trace_count": len(train_traces),
            "test_trace_count": len(test_traces),
            "ignored_trace_count": 0,
            "split_strategy": "chronological_80_20_fallback",
            "fallback_reason": "trace_count_less_than_25",
        }

    train_traces = []
    test_traces = []
    for section_id in range(5):
        start = section_id * sec_num * 5
        test_end = start + sec_num
        section_end = (section_id + 1) * sec_num * 5
        test_traces.extend(sorted_traces[start:test_end])
        train_traces.extend(sorted_traces[test_end:section_end])

    return train_traces, test_traces, {
        "trace_count": trace_num,
        "sec_num": sec_num,
        "train_trace_count": len(train_traces),
        "test_trace_count": len(test_traces),
        "ignored_trace_count": trace_num - len(train_traces) - len(test_traces),
        "split_strategy": "paper_5x5_sections",
        "fallback_reason": "",
    }


def _flatten_traces(traces: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
    return np.asarray([event for trace in traces for event in trace], dtype=np.float32)


def make_lgbm_regressor():
    require_lightgbm()
    return lgb.LGBMRegressor(verbose=-1)


def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def calculate_regression_metrics(y_true, y_pred, variant_freq=None, tail_threshold=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)

    abs_error = np.abs(y_true - y_pred)
    sq_error = (y_true - y_pred) ** 2
    mae = float(abs_error.mean()) if abs_error.size else 0.0
    rmse = float(np.sqrt(sq_error.mean())) if sq_error.size else 0.0

    if variant_freq is None or len(variant_freq) == 0:
        tail_mae = mae
        q1 = 1.0 if tail_threshold is None else float(tail_threshold)
    else:
        variant_freq = np.asarray(variant_freq, dtype=np.float64)
        q1 = float(np.quantile(variant_freq, 0.33)) if tail_threshold is None else float(tail_threshold)
        tail_mask = variant_freq <= q1
        tail_mae = float(abs_error[tail_mask].mean()) if tail_mask.any() else mae

    return {
        "mae": mae,
        "rmse": rmse,
        "tail_mae": tail_mae,
        "score": 0.5 * mae + 0.5 * tail_mae,
        "tail_q1": q1,
    }


def paper_show_local_tree(X_train, y_train, X_test, y_test, candidate_indices, categorical_indices, task):
    ai = sorted(candidate_indices)
    cid = set(categorical_indices)
    if not ai:
        raise ValueError("特征选择候选集为空。")

    if task == 0:
        raise NotImplementedError("当前仅复现论文中的剩余时间回归分支。")

    model = make_lgbm_regressor()
    selected = [ai[0]]
    selected_mae = []
    remaining = ai[1:]

    model.fit(
        X_train[:, selected],
        y_train[:, task],
        feature_name=[str(i) for i in selected],
        categorical_feature=[str(i) for i in selected if i in cid],
    )
    pred = model.predict(X_test[:, selected])
    selected_mae.append(calculate_mae(y_test[:, task], pred))

    while remaining:
        best_mae = float("inf")
        best_feature = remaining[0]
        for feature_idx in remaining:
            candidate = selected + [feature_idx]
            model.fit(
                X_train[:, candidate],
                y_train[:, task],
                feature_name=[str(i) for i in candidate],
                categorical_feature=[str(i) for i in candidate if i in cid],
            )
            pred = model.predict(X_test[:, candidate])
            mae = calculate_mae(y_test[:, task], pred)
            if feature_idx == remaining[0] or mae < best_mae:
                best_mae = mae
                best_feature = feature_idx
        selected.append(best_feature)
        selected_mae.append(best_mae)
        remaining.remove(best_feature)

    return selected, selected_mae


def select_features_with_paper_strategy(train_traces, test_traces, header, categorical_indices):
    require_lightgbm()
    attrib_num = len(header) - 3
    train_rows = _flatten_traces(train_traces)
    test_rows = _flatten_traces(test_traces)

    all_feature_indices = list(range(attrib_num))
    X_train = train_rows[:, all_feature_indices]
    y_train = train_rows[:, attrib_num:attrib_num + 3]
    X_val = test_rows[:, all_feature_indices]
    y_val = test_rows[:, attrib_num:attrib_num + 3]

    model = make_lgbm_regressor()
    selected = list(all_feature_indices)
    priority = {feature_idx: 0 for feature_idx in selected}
    priority[0] = 30
    history = []
    removed_feature = None
    min_priority = 0
    selected_count = len(selected)

    while True:
        model.fit(
            X_train[:, selected],
            y_train[:, 2],
            feature_name=[str(i) for i in selected],
            categorical_feature=[str(i) for i in selected if i in categorical_indices],
        )
        pred = model.predict(X_val[:, selected])
        mae = calculate_mae(y_val[:, 2], pred)

        if history:
            if mae > history[-1][0]:
                history.append((mae, selected.copy(), removed_feature))
                priority[removed_feature] += 1
                selected.append(removed_feature)
                model.fit(
                    X_train[:, selected],
                    y_train[:, 2],
                    feature_name=[str(i) for i in selected],
                    categorical_feature=[str(i) for i in selected if i in categorical_indices],
                )
                pred = model.predict(X_val[:, selected])
                mae = calculate_mae(y_val[:, 2], pred)
            else:
                priority.pop(removed_feature, None)

        current_importances = model.feature_importances_
        min_importance = max(current_importances) if len(current_importances) else 0
        remove_position = 0
        current_min_priority = min(priority.values())
        for feature_idx, position in zip(selected, range(len(selected))):
            if priority[feature_idx] == current_min_priority and min_importance >= current_importances[position]:
                min_importance = current_importances[position]
                remove_position = position

        history.append((mae, selected.copy(), selected[remove_position]))
        if min(priority.values()) > min_priority:
            if selected_count == len(selected):
                break
            selected_count = len(selected)
            min_priority = min(priority.values())
        if len(selected) == 1:
            break

        removed_feature = selected[remove_position]
        selected.remove(removed_feature)

    ordered_indices, mae_history = paper_show_local_tree(
        X_train,
        y_train,
        X_val,
        y_val,
        sorted(selected),
        categorical_indices,
        task=2,
    )
    best_index = int(np.argmin(np.asarray(mae_history)))
    for index in range(best_index):
        if mae_history[index] - mae_history[best_index] < 0.2:
            best_index = index
            break

    selected_indices = ordered_indices[:best_index + 1]
    selector_mae = float(mae_history[best_index])

    selected_names = [header[index] for index in selected_indices]
    return selected_indices, selected_names, selector_mae


def _split_selector_train_val(traces, validation_ratio=0.2):
    if len(traces) < 2:
        return list(traces), list(traces)

    split_idx = int(len(traces) * (1.0 - validation_ratio))
    split_idx = min(max(split_idx, 1), len(traces) - 1)
    return list(traces[:split_idx]), list(traces[split_idx:])


def _flatten_trace_rows_with_freq(traces):
    variant_keys = [tuple(int(event[0]) for event in trace) for trace in traces]
    variant_counts = Counter(variant_keys)
    rows = []
    freqs = []
    for trace, key in zip(traces, variant_keys):
        freq = float(variant_counts[key])
        for event in trace:
            rows.append(event)
            freqs.append(freq)
    return np.asarray(rows, dtype=np.float32), np.asarray(freqs, dtype=np.float32)


def evaluate_fpm_lightgbm(
    raw_dataset_path,
    dataset_name="dataset",
    output_dir="results",
    seed=42,
    target_scale=None,
):
    require_lightgbm()
    random.seed(seed)
    np.random.seed(seed)

    paper_log = load_paper_event_log(raw_dataset_path, dataset_name=dataset_name)
    train_traces, test_traces, split_summary = split_traces_with_paper_strategy(
        paper_log["traces"],
        paper_log["trace_times"],
    )
    selector_train, selector_val = _split_selector_train_val(train_traces)
    selected_indices, selected_names, selector_mae = select_features_with_paper_strategy(
        train_traces=selector_train,
        test_traces=selector_val,
        header=paper_log["header"],
        categorical_indices=paper_log["categorical_indices"],
    )
    selected_states = [paper_log["state"][feature_idx] for feature_idx in selected_indices]
    metric_scale = float(paper_log.get("target_scale", 24.0) if target_scale is None else target_scale)

    train_rows, _ = _flatten_trace_rows_with_freq(train_traces)
    test_rows, test_variant_freq = _flatten_trace_rows_with_freq(test_traces)
    attrib_num = len(paper_log["header"]) - 3
    y_train = train_rows[:, attrib_num + 2]
    y_test = test_rows[:, attrib_num + 2]

    model = make_lgbm_regressor()
    categorical_positions = [
        position
        for position, feature_idx in enumerate(selected_indices)
        if feature_idx in set(paper_log["categorical_indices"])
    ]
    model.fit(
        train_rows[:, selected_indices],
        y_train,
        feature_name=[str(feature_idx) for feature_idx in selected_indices],
        categorical_feature=categorical_positions,
    )
    y_pred = model.predict(test_rows[:, selected_indices])

    metrics = calculate_regression_metrics(
        y_true=y_test * metric_scale,
        y_pred=y_pred * metric_scale,
        variant_freq=test_variant_freq,
    )
    selected_feature_path = save_selected_features(
        output_dir=output_dir,
        dataset_name=dataset_name,
        selected_indices=selected_indices,
        selected_names=selected_names,
        selected_states=selected_states,
        selector_mae=selector_mae * metric_scale,
    )

    return {
        "dataset": dataset_name,
        "raw_dataset_path": raw_dataset_path,
        "selected_feature_path": selected_feature_path,
        "selected_feature_count": len(selected_indices),
        "selected_feature_indices": selected_indices,
        "selected_features": selected_names,
        "selected_feature_states": selected_states,
        "selector_valid_mae": selector_mae * metric_scale,
        "time_unit": paper_log.get("time_unit", "day"),
        "source_schema": paper_log.get("source_schema", "paper_raw"),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "tail_mae": metrics["tail_mae"],
        "score": metrics["score"],
        "tail_q1": metrics["tail_q1"],
        "split_summary": split_summary,
    }


PAPER_TABLE3_REFERENCE_DAY = {
    "BPIC2015_1": {"Activity": 39.96, "All": 28.10, "EFC": 32.47, "Prefix5": 27.47},
    "BPIC2015_2": {"Activity": 78.94, "All": 73.76, "EFC": 68.73, "Prefix5": 67.79},
    "BPIC2015_3": {"Activity": 22.93, "All": 20.83, "EFC": 18.48, "Prefix5": 16.73},
    "BPIC2015_4": {"Activity": 63.65, "All": 52.07, "EFC": 49.98, "Prefix5": 52.30},
    "BPIC2015_5": {"Activity": 49.98, "All": 43.14, "EFC": 38.94, "Prefix5": 38.90},
    "Sepsis": {"Activity": 39.78, "All": 50.09, "EFC": 42.45, "Prefix5": 45.73},
    "Helpdesk": {"Activity": 6.57, "All": 5.01, "EFC": 4.78, "Prefix5": 3.12},
}


PAPER_TABLE4_REFERENCE_DAY = {
    "BPIC2015_1": {"LSTM": 29.26, "FPM_LSTM": 26.44, "PT": 24.71, "FPM_Transformer": 26.52, "AETS": 37.88, "FPM_AETS": 36.33},
    "BPIC2015_2": {"LSTM": 71.94, "FPM_LSTM": 66.69, "PT": 68.89, "FPM_Transformer": 68.21, "AETS": 57.89, "FPM_AETS": 56.85},
    "BPIC2015_3": {"LSTM": 19.19, "FPM_LSTM": 18.40, "PT": 17.71, "FPM_Transformer": 16.80, "AETS": 19.06, "FPM_AETS": 17.70},
    "BPIC2015_4": {"LSTM": 52.63, "FPM_LSTM": 49.02, "PT": 49.97, "FPM_Transformer": 54.58, "AETS": 38.31, "FPM_AETS": 38.14},
    "BPIC2015_5": {"LSTM": 36.86, "FPM_LSTM": 35.85, "PT": 36.04, "FPM_Transformer": 36.45, "AETS": 45.14, "FPM_AETS": 44.01},
    "Sepsis": {"LSTM": 31.78, "FPM_LSTM": 30.64, "PT": 26.07, "FPM_Transformer": 27.46, "AETS": 35.39, "FPM_AETS": 35.07},
    "Helpdesk": {"LSTM": 6.29, "FPM_LSTM": 4.59, "PT": 5.50, "FPM_Transformer": 1.18, "AETS": 8.63, "FPM_AETS": 6.37},
}


def _canonical_dataset_name(dataset_name):
    name = str(dataset_name)
    lower = name.lower()
    if lower in {"hd", "helpdesk"}:
        return "Helpdesk"
    if lower == "sepsis":
        return "Sepsis"
    return name


def _paper_table4_reference_day(dataset_name, model_name):
    return PAPER_TABLE4_REFERENCE_DAY.get(_canonical_dataset_name(dataset_name), {}).get(model_name, "")


def _metric_rows_from_prediction(dataset_name, feature_set, y_true_hour, y_pred_hour, variant_freq, selected_count):
    metrics_hour = calculate_regression_metrics(y_true_hour, y_pred_hour, variant_freq=variant_freq)
    ref = PAPER_TABLE3_REFERENCE_DAY.get(_canonical_dataset_name(dataset_name), {}).get(feature_set, "")
    mae_day = metrics_hour["mae"] / 24.0
    return {
        "dataset": dataset_name,
        "feature_set": feature_set,
        "selected_feature_count": selected_count,
        "mae_hour": metrics_hour["mae"],
        "tail_mae_hour": metrics_hour["tail_mae"],
        "rmse_hour": metrics_hour["rmse"],
        "score_hour": metrics_hour["score"],
        "mae_day": mae_day,
        "tail_mae_day": metrics_hour["tail_mae"] / 24.0,
        "rmse_day": metrics_hour["rmse"] / 24.0,
        "score_day": metrics_hour["score"] / 24.0,
        "paper_reference_mae_day": ref,
        "paper_delta_mae_day": (mae_day - ref) if isinstance(ref, (int, float)) else "",
        "tail_q1": metrics_hour["tail_q1"],
    }


def _fit_predict_lgbm(train_x, train_y, test_x, feature_indices, categorical_original_indices, seed=42):
    model = lgb.LGBMRegressor(random_state=seed, n_estimators=100, verbose=-1)
    categorical_positions = [
        pos for pos, feature_idx in enumerate(feature_indices)
        if feature_idx in set(categorical_original_indices)
    ]
    model.fit(
        train_x,
        train_y,
        categorical_feature=categorical_positions,
    )
    return model.predict(test_x)


def _evaluate_event_feature_set(
    dataset_name,
    feature_set,
    train_rows,
    test_rows,
    test_variant_freq,
    attrib_num,
    feature_indices,
    categorical_indices,
    metric_scale,
    seed,
):
    print(f"[FPM:Table3] Train LightGBM | feature_set={feature_set} | features={len(feature_indices)}", flush=True)
    y_train = train_rows[:, attrib_num + 2]
    y_test_hour = test_rows[:, attrib_num + 2] * metric_scale
    y_pred = _fit_predict_lgbm(
        train_rows[:, feature_indices],
        y_train,
        test_rows[:, feature_indices],
        feature_indices=feature_indices,
        categorical_original_indices=categorical_indices,
        seed=seed,
    )
    return _metric_rows_from_prediction(
        dataset_name=dataset_name,
        feature_set=feature_set,
        y_true_hour=y_test_hour,
        y_pred_hour=y_pred * metric_scale,
        variant_freq=test_variant_freq,
        selected_count=len(feature_indices),
    )


def _flatten_prefix5_rows(traces, attrib_num, pf=5):
    variant_keys = [tuple(int(event[0]) for event in trace) for trace in traces]
    variant_counts = Counter(variant_keys)
    rows = []
    freqs = []
    for trace, key in zip(traces, variant_keys):
        if len(trace) < pf:
            continue
        for end in range(pf, len(trace) + 1):
            window = trace[end - pf:end]
            row = []
            for event in window:
                row.extend(event[:attrib_num])
            row.extend(window[-1][attrib_num:attrib_num + 3])
            rows.append(row)
            freqs.append(float(variant_counts[key]))
    return np.asarray(rows, dtype=np.float32), np.asarray(freqs, dtype=np.float32)


def _evaluate_prefix5_feature_set(
    dataset_name,
    train_traces,
    test_traces,
    attrib_num,
    selected_indices,
    categorical_indices,
    metric_scale,
    seed,
    pf=5,
):
    print(f"[FPM:Table3] Build Prefix{pf} matrix | selected_base_features={len(selected_indices)}", flush=True)
    train_rows = _flatten_prefix5_rows(train_traces, attrib_num=attrib_num, pf=pf)[0]
    test_rows, test_freq = _flatten_prefix5_rows(test_traces, attrib_num=attrib_num, pf=pf)
    if train_rows.size == 0 or test_rows.size == 0:
        raise ValueError(f"Prefix{pf} 构造失败：训练或测试 prefix 样本为空。")

    label_start = attrib_num * pf
    feature_indices = [
        step * attrib_num + feature_idx
        for step in range(pf)
        for feature_idx in selected_indices
    ]
    categorical_expanded = {
        step * attrib_num + feature_idx
        for step in range(pf)
        for feature_idx in categorical_indices
    }
    print(f"[FPM:Table3] Train LightGBM | feature_set=Prefix{pf} | features={len(feature_indices)}", flush=True)
    y_train = train_rows[:, label_start + 2]
    y_test_hour = test_rows[:, label_start + 2] * metric_scale
    y_pred = _fit_predict_lgbm(
        train_rows[:, feature_indices],
        y_train,
        test_rows[:, feature_indices],
        feature_indices=feature_indices,
        categorical_original_indices=categorical_expanded,
        seed=seed,
    )
    return _metric_rows_from_prediction(
        dataset_name=dataset_name,
        feature_set=f"Prefix{pf}",
        y_true_hour=y_test_hour,
        y_pred_hour=y_pred * metric_scale,
        variant_freq=test_freq,
        selected_count=len(selected_indices),
    )


def run_fpm_table3_experiment(raw_dataset_path, dataset_name="dataset", output_dir="results", seed=42):
    require_lightgbm()
    random.seed(seed)
    np.random.seed(seed)
    print(f"[FPM:Table3] Load data | path={raw_dataset_path}", flush=True)
    paper_log = load_paper_event_log(raw_dataset_path, dataset_name=dataset_name)
    print(
        f"[FPM:Table3] Loaded | schema={paper_log.get('source_schema', 'paper_raw')} | "
        f"traces={len(paper_log['traces'])} | features={len(paper_log['feature_names'])} | "
        f"unit={paper_log.get('time_unit', 'day')}",
        flush=True,
    )
    train_traces, test_traces, split_summary = split_traces_with_paper_strategy(
        paper_log["traces"],
        paper_log["trace_times"],
    )
    print(f"[FPM:Table3] Split | {split_summary}", flush=True)

    selector_train, selector_val = _split_selector_train_val(train_traces)
    print("[FPM:Table3] Feature selection Step1/Step2 start", flush=True)
    selected_indices, selected_names, selector_mae = select_features_with_paper_strategy(
        train_traces=selector_train,
        test_traces=selector_val,
        header=paper_log["header"],
        categorical_indices=paper_log["categorical_indices"],
    )
    selected_states = [paper_log["state"][feature_idx] for feature_idx in selected_indices]
    print(
        f"[FPM:Table3] Feature selection done | selected={len(selected_indices)} | "
        f"features={selected_names}",
        flush=True,
    )

    selected_feature_path = save_selected_features(
        output_dir=output_dir,
        dataset_name=dataset_name,
        selected_indices=selected_indices,
        selected_names=selected_names,
        selected_states=selected_states,
        selector_mae=selector_mae * float(paper_log.get("target_scale", 24.0)),
    )

    train_rows, _ = _flatten_trace_rows_with_freq(train_traces)
    test_rows, test_variant_freq = _flatten_trace_rows_with_freq(test_traces)
    attrib_num = len(paper_log["header"]) - 3
    metric_scale = float(paper_log.get("target_scale", 24.0))
    all_indices = list(range(attrib_num))

    rows = [
        _evaluate_event_feature_set(
            dataset_name,
            "Activity",
            train_rows,
            test_rows,
            test_variant_freq,
            attrib_num,
            [0],
            paper_log["categorical_indices"],
            metric_scale,
            seed,
        ),
        _evaluate_event_feature_set(
            dataset_name,
            "All",
            train_rows,
            test_rows,
            test_variant_freq,
            attrib_num,
            all_indices,
            paper_log["categorical_indices"],
            metric_scale,
            seed,
        ),
        _evaluate_event_feature_set(
            dataset_name,
            "EFC",
            train_rows,
            test_rows,
            test_variant_freq,
            attrib_num,
            selected_indices,
            paper_log["categorical_indices"],
            metric_scale,
            seed,
        ),
        _evaluate_prefix5_feature_set(
            dataset_name,
            train_traces,
            test_traces,
            attrib_num,
            selected_indices,
            paper_log["categorical_indices"],
            metric_scale,
            seed,
            pf=5,
        ),
    ]

    return {
        "dataset": dataset_name,
        "selected_feature_path": selected_feature_path,
        "selected_feature_indices": selected_indices,
        "selected_features": selected_names,
        "selected_feature_states": selected_states,
        "selector_valid_mae_hour": selector_mae * metric_scale,
        "split_summary": split_summary,
        "paper_log": paper_log,
        "train_traces": train_traces,
        "test_traces": test_traces,
        "table3_rows": rows,
    }


def prepare_fpm_table4_data(raw_dataset_path, dataset_name="dataset", output_dir="results", seed=42):
    """Prepare the paper Table 4 FPM data path: LightGBMNew-style feature selection only."""
    require_lightgbm()
    random.seed(seed)
    np.random.seed(seed)
    print(f"[FPM:Table4] Load data | path={raw_dataset_path}", flush=True)
    paper_log = load_paper_event_log(raw_dataset_path, dataset_name=dataset_name)
    print(
        f"[FPM:Table4] Loaded | schema={paper_log.get('source_schema', 'paper_raw')} | "
        f"traces={len(paper_log['traces'])} | features={len(paper_log['feature_names'])} | "
        f"unit={paper_log.get('time_unit', 'day')}",
        flush=True,
    )
    train_traces, test_traces, split_summary = split_traces_with_paper_strategy(
        paper_log["traces"],
        paper_log["trace_times"],
    )
    print(f"[FPM:Table4] Split | {split_summary}", flush=True)

    selector_train, selector_val = _split_selector_train_val(train_traces)
    print("[FPM:Table4] FPM feature selection start | LightGBM backward + forward tree", flush=True)
    selected_indices, selected_names, selector_mae = select_features_with_paper_strategy(
        train_traces=selector_train,
        test_traces=selector_val,
        header=paper_log["header"],
        categorical_indices=paper_log["categorical_indices"],
    )
    selected_states = [paper_log["state"][feature_idx] for feature_idx in selected_indices]
    print(
        f"[FPM:Table4] FPM feature selection done | selected={len(selected_indices)} | "
        f"features={selected_names}",
        flush=True,
    )

    metric_scale = float(paper_log.get("target_scale", 24.0))
    selected_feature_path = save_selected_features(
        output_dir=output_dir,
        dataset_name=dataset_name,
        selected_indices=selected_indices,
        selected_names=selected_names,
        selected_states=selected_states,
        selector_mae=selector_mae * metric_scale,
    )

    return {
        "dataset": dataset_name,
        "selected_feature_path": selected_feature_path,
        "selected_feature_indices": selected_indices,
        "selected_features": selected_names,
        "selected_feature_states": selected_states,
        "selector_valid_mae_hour": selector_mae * metric_scale,
        "split_summary": split_summary,
        "paper_log": paper_log,
        "train_traces": train_traces,
        "test_traces": test_traces,
        "table3_rows": [],
    }


class _FPMSequenceDataset(Dataset):
    def __init__(self, traces, selected_indices, metric_scale=1.0):
        self.items = []
        self.selected_indices = list(selected_indices)
        self.metric_scale = float(metric_scale)
        for trace in traces:
            if not trace:
                continue
            x = [[float(event[i]) for i in self.selected_indices] for event in trace]
            y = [float(event[-1]) * self.metric_scale for event in trace]
            self.items.append((x, y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "length": len(x),
        }


def _fpm_sequence_collate(batch):
    batch.sort(key=lambda item: item["length"], reverse=True)
    x = nn.utils.rnn.pad_sequence([item["x"] for item in batch], batch_first=True, padding_value=0.0)
    y = nn.utils.rnn.pad_sequence([item["y"] for item in batch], batch_first=True, padding_value=0.0)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    mask = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return {"x": x, "y": y, "mask": mask}


class _FpmLSTM(nn.Module if nn is not None else object):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Softplus())

    def forward(self, x, mask=None):
        y, _ = self.rnn(x)
        return self.out(y).squeeze(-1), None


class _FpmTransformer(nn.Module if nn is not None else object):
    def __init__(self, input_dim, d_model=64, num_heads=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Softplus())

    def forward(self, x, mask=None):
        z = self.proj(x)
        pad_mask = ~mask if mask is not None else None
        z = self.encoder(z, src_key_padding_mask=pad_mask)
        return self.out(z).squeeze(-1), None


class _FpmAETS(nn.Module if nn is not None else object):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.ReLU())
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.out = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1), nn.Softplus())

    def forward(self, x, mask=None):
        z = self.encoder(x)
        recon = self.decoder(z)
        return self.out(z).squeeze(-1), recon


def _evaluate_sequence_model(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            pred, _ = model(x, mask)
            preds.append(pred[mask].detach().cpu().numpy())
            trues.append(y[mask].detach().cpu().numpy())
    if not preds:
        return {"mae": 0.0, "rmse": 0.0}
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    metrics = calculate_regression_metrics(y_true, y_pred)
    return {"mae": metrics["mae"], "rmse": metrics["rmse"], "score": metrics["score"]}


def _train_one_sequence_model(model_name, model, train_loader, test_loader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    best = None
    for epoch in range(int(epochs)):
        start = time.time()
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad()
            pred, recon = model(x, mask)
            pred_loss = l1(pred[mask], y[mask])
            loss = pred_loss
            if recon is not None:
                loss = loss + 0.1 * mse(recon[mask], x[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(pred_loss.item()) * int(mask.sum().item())
            n += int(mask.sum().item())
        val = _evaluate_sequence_model(model, test_loader, device)
        train_mae = total_loss / max(n, 1)
        elapsed = time.time() - start
        print(
            f"    Epoch {epoch + 1:02d}/{epochs} | Model={model_name} | "
            f"TrainMAE={train_mae:.4f} | ValMAE={val['mae']:.4f} | "
            f"RMSE={val['rmse']:.4f} | Time={elapsed:.1f}s",
            flush=True,
        )
        if best is None or val["mae"] < best["mae"]:
            best = val
            best["best_epoch"] = epoch + 1
    return best or {"mae": 0.0, "rmse": 0.0, "score": 0.0, "best_epoch": 0}


def run_fpm_framework_experiment(
    table3_result,
    epochs=1,
    batch_size=64,
    lr=1e-3,
    device=None,
    model_names=None,
):
    require_torch()
    dataset_name = table3_result["dataset"]
    selected_indices = table3_result["selected_feature_indices"]
    metric_scale = float(table3_result["paper_log"].get("target_scale", 24.0))
    train_dataset = _FPMSequenceDataset(table3_result["train_traces"], selected_indices, metric_scale=metric_scale)
    test_dataset = _FPMSequenceDataset(table3_result["test_traces"], selected_indices, metric_scale=metric_scale)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_fpm_sequence_collate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=_fpm_sequence_collate)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(selected_indices)
    model_names = list(model_names or ["FPM_LSTM", "FPM_Transformer"])
    model_factories = {
        "FPM_LSTM": lambda: _FpmLSTM(input_dim=input_dim),
        "FPM_Transformer": lambda: _FpmTransformer(input_dim=input_dim),
        "FPM_AETS": lambda: _FpmAETS(input_dim=input_dim),
    }
    rows = []
    for model_name in model_names:
        if model_name not in model_factories:
            raise ValueError(f"Unsupported FPM Table4 model: {model_name}")
        model = model_factories[model_name]()
        print(f"[FPM:Table4] Train {model_name} | epochs={epochs} | device={device}", flush=True)
        metrics = _train_one_sequence_model(model_name, model, train_loader, test_loader, epochs, lr, device)
        ref = _paper_table4_reference_day(dataset_name, model_name)
        mae_day = metrics["mae"] / 24.0
        rows.append({
            "dataset": dataset_name,
            "model": model_name,
            "mae_hour": metrics["mae"],
            "rmse_hour": metrics["rmse"],
            "score_hour": metrics.get("score", metrics["mae"]),
            "mae_day": mae_day,
            "rmse_day": metrics["rmse"] / 24.0,
            "score_day": metrics.get("score", metrics["mae"]) / 24.0,
            "paper_reference_mae_day": ref,
            "paper_delta_mae_day": (mae_day - ref) if isinstance(ref, (int, float)) else "",
            "best_epoch": metrics.get("best_epoch", 0),
            "incremental_period_month": "not_run_placeholder",
            "incremental_quantity_100": "not_run_placeholder",
            "concept_drift": "placeholder",
        })
    return rows


def _build_activity_windows(traces: Sequence[Sequence[Sequence[float]]], batch_size=20, window_len=3):
    windows = []
    targets = []
    for trace in traces:
        activity_seq = [int(event[0]) for event in trace]
        next_event_seq = [int(event[-3]) for event in trace]
        if len(activity_seq) < window_len:
            continue
        for start in range(len(activity_seq) - window_len + 1):
            windows.append(activity_seq[start:start + window_len])
            targets.append(next_event_seq[start + window_len - 1])
    if not windows:
        raise ValueError("无法构建论文源码所需的活动 CBoW 训练窗口。")
    while len(windows) % batch_size != 0:
        rand_idx = random.randint(0, len(windows) - 1)
        windows.append(windows[rand_idx])
        targets.append(targets[rand_idx])
    return windows, targets


def train_activity_cbow_embedding(train_traces, activity_vocab_size):
    require_torch()
    windows, targets = _build_activity_windows(train_traces, batch_size=20, window_len=3)

    embed_dim = 16
    vocab_len = max(activity_vocab_size, 1)
    while vocab_len > 16:
        vocab_len /= 4
        embed_dim += 4

    class CBOW(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_size):
            super().__init__()
            self.embed_layer = nn.Embedding(vocab_size, embed_dim)
            self.linear_1 = nn.Linear(embed_dim, hidden_size)
            self.linear_2 = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_data):
            embeds = self.embed_layer(input_data)
            embed_sum = embeds.sum(dim=1)
            hidden = torch.relu(self.linear_1(embed_sum))
            return torch.log_softmax(self.linear_2(hidden), dim=1)

    model = CBOW(activity_vocab_size, embed_dim, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    batch_size = 20
    for _ in range(20):
        for start in range(0, len(windows), batch_size):
            batch_x = torch.tensor(windows[start:start + batch_size], dtype=torch.long)
            batch_y = torch.tensor(targets[start:start + batch_size], dtype=torch.long)
            optimizer.zero_grad()
            log_probs = model(batch_x)
            loss = criterion(log_probs, batch_y)
            loss.backward()
            optimizer.step()

    return model.embed_layer.weight.detach().cpu().numpy()


def build_embedding_metadata(selected_indices, selected_states, vocabularies):
    require_torch()
    metadata = {}
    activity_vocab = vocabularies[0]
    metadata["activity_vocab_size"] = len(activity_vocab)

    for feature_idx, state in zip(selected_indices, selected_states):
        if feature_idx == 0 or state < 3:
            vocab = vocabularies[feature_idx]
            embed_dim = 4
            vocab_len = max(len(vocab), 1)
            while vocab_len > 16:
                vocab_len /= 4
                embed_dim += 4
            metadata[str(feature_idx)] = nn.Embedding(len(vocab), embed_dim).weight.detach().cpu().numpy()

    return metadata


def paper_fpm_collate_fn(batch):
    require_torch()
    batch.sort(key=lambda item: item["length"], reverse=True)
    feature_seq = nn.utils.rnn.pad_sequence(
        [item["feature_seq"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    target_seq = nn.utils.rnn.pad_sequence(
        [item["target_seq"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    mask = torch.arange(feature_seq.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    variant_freq = torch.stack([item["variant_freq"] for item in batch])
    variant_freq_seq = variant_freq.unsqueeze(1).expand_as(target_seq)
    return {
        "feature_seq": feature_seq,
        "target_seq": target_seq,
        "variant_freq": variant_freq,
        "variant_freq_seq": variant_freq_seq,
        "lengths": lengths,
        "mask": mask,
    }


def save_selected_features(output_dir, dataset_name, selected_indices, selected_names, selected_states, selector_mae):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"paper_fpm_selected_features_{dataset_name}.csv")
    rows = []
    for rank, (feature_index, feature_name, feature_state) in enumerate(
        zip(selected_indices, selected_names, selected_states),
        start=1,
    ):
        rows.append(
            {
                "rank": rank,
                "feature_index": feature_index,
                "feature": feature_name,
                "state": feature_state,
                "selector_mae": selector_mae,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_fpm_feature_artifact(
    raw_dataset_path,
    train_case_ids=None,
    test_case_ids=None,
    seed=42,
    max_seq_len=50,
    max_prefixes_per_case=None,
    output_dir="results",
    dataset_name="dataset",
    tolerance=0.2,
):
    del train_case_ids, test_case_ids, max_seq_len, max_prefixes_per_case, tolerance

    require_lightgbm()
    random.seed(seed)
    np.random.seed(seed)

    paper_log = load_paper_event_log(raw_dataset_path, dataset_name=dataset_name)
    train_traces, test_traces, split_summary = split_traces_with_paper_strategy(
        paper_log["traces"],
        paper_log["trace_times"],
    )
    selected_indices, selected_names, selector_mae = select_features_with_paper_strategy(
        train_traces=train_traces,
        test_traces=test_traces,
        header=paper_log["header"],
        categorical_indices=paper_log["categorical_indices"],
    )
    selected_states = [paper_log["state"][feature_idx] for feature_idx in selected_indices]

    require_torch()
    embedding_metadata = build_embedding_metadata(
        selected_indices=selected_indices,
        selected_states=selected_states,
        vocabularies=paper_log["vocabularies"],
    )
    embedding_metadata["0"] = train_activity_cbow_embedding(
        train_traces=train_traces,
        activity_vocab_size=len(paper_log["vocabularies"][0]),
    )

    max_case_length = max(len(trace) for trace in paper_log["traces"])
    model_metadata = {
        "feature_indices": selected_indices,
        "feature_names": selected_names,
        "feature_states": selected_states,
        "all_states": paper_log["state"],
        "embeddings": embedding_metadata,
        "max_case_length": max_case_length,
        "time_unit": paper_log.get("time_unit", "day"),
    }

    selected_feature_path = save_selected_features(
        output_dir=output_dir,
        dataset_name=dataset_name,
        selected_indices=selected_indices,
        selected_names=selected_names,
        selected_states=selected_states,
        selector_mae=selector_mae,
    )

    train_dataset = PaperTraceDataset(train_traces, selected_indices)
    test_dataset = PaperTraceDataset(test_traces, selected_indices)

    return FPMFeatureArtifact(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        selected_features=selected_names,
        selected_feature_indices=selected_indices,
        selected_feature_states=selected_states,
        selector_valid_mae=selector_mae,
        selected_feature_path=selected_feature_path,
        input_dim=len(selected_indices),
        collate_fn=paper_fpm_collate_fn,
        model_metadata=model_metadata,
        split_summary=split_summary,
    )
