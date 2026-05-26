# -*- coding: utf-8 -*-
"""
PGR-TC: Prefix-aware Global Retrieval Network with Temporal Consistency-friendly training.

This script keeps baseline.py and baseline_dnc_ica.py unchanged.  It builds on the
existing ProcessPrefixDataset output style and adds a new experiment path for
business-process remaining-time prediction:

    Prefix statistical features + LocalTCN + Dual Event-Aligned GTR + Transformer
    + log1p Tail-weighted Huber loss

Run:
    python baseline_pgr_tc.py

For quick smoke test, temporarily set EPOCHS=1, SEEDS=[42], MAX_PREFIXES=20.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline import (
    LSTMBaseline,
    append_bucket_rows if False else set_seed,  # keep import linter-neutral in simple environments
    evaluate_model,
    train_model,
    train_random_forest,
)
from model import VanillaTransformerBaseline
from prefix import ProcessPrefixDataset, ProcessVocab
from training_utils import batch_tail_mask, evaluate_bucket_regression, split_dataframe


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _safe_float_array(values: List[float], dtype=np.float32) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    return arr


def _skewness(arr: np.ndarray) -> float:
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr))
    if std < 1e-6:
        return 0.0
    mean = float(np.mean(arr))
    return float(np.mean(((arr - mean) / std) ** 3))


def _kurtosis(arr: np.ndarray) -> float:
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr))
    if std < 1e-6:
        return 0.0
    mean = float(np.mean(arr))
    return float(np.mean(((arr - mean) / std) ** 4))


def _ewma_last(arr: np.ndarray, alpha: float = 0.5) -> float:
    if arr.size == 0:
        return 0.0
    state = float(arr[0])
    for value in arr[1:]:
        state = alpha * float(value) + (1.0 - alpha) * state
    return state


def _transition_entropy(ids: List[int]) -> float:
    if len(ids) <= 1:
        return 0.0
    counts: Dict[Tuple[int, int], int] = {}
    for a, b in zip(ids[:-1], ids[1:]):
        counts[(int(a), int(b))] = counts.get((int(a), int(b)), 0) + 1
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * np.log(p + 1e-12)
    return float(entropy)


def _count_non_adjacent_repeats(ids: List[int]) -> int:
    if len(ids) <= 2:
        return 0
    seen = set()
    count = 0
    for idx, item in enumerate(ids):
        if item in seen and (idx == 0 or item != ids[idx - 1]):
            count += 1
        seen.add(item)
    return int(count)


# -----------------------------------------------------------------------------
# Dataset adapter: enrich existing prefix samples without modifying prefix.py
# -----------------------------------------------------------------------------


class EnhancedProcessPrefixDataset(ProcessPrefixDataset):
    """Add calendar/progress buckets and prefix-level statistical features.

    The base repository already builds prefix samples with act_seq, res_seq,
    time_seq, mask, target_rem_time and variant_freq.  This subclass preserves all
    original fields and adds:
        prefix_stat_feats: FloatTensor[F_stat]
        calendar_bucket_ids: LongTensor[L]
        progress_bucket_ids: LongTensor[L]
        prefix_len: LongTensor[]
        case_id: string kept inside prefixes for analysis where possible

    All features are causal: only events inside the current prefix are used.
    """

    prefix_stat_feature_names = [
        "prefix_delta_mean",
        "prefix_delta_std",
        "prefix_delta_max",
        "prefix_delta_min",
        "prefix_delta_median",
        "prefix_delta_ewma",
        "prefix_delta_skewness",
        "prefix_delta_kurtosis",
        "recent_delta_mean_3",
        "recent_delta_std_3",
        "recent_delta_mean_5",
        "recent_delta_std_5",
        "unique_activity_count",
        "unique_activity_ratio",
        "activity_repeat_count",
        "activity_loop_count",
        "recent_activity_repeat_count_3",
        "activity_transition_entropy",
        "unique_resource_count",
        "unique_resource_ratio",
        "resource_switch_count",
        "resource_switch_ratio",
        "recent_resource_switch_count_3",
        "prefix_len",
        "normalized_prefix_pos",
        "last_hour",
        "last_weekday",
        "is_weekend",
    ]

    def _generate_prefixes(self, df):
        print("正在生成 PGR-TC 增强前缀序列 (Sliding Trace Prefixes + prefix stats metadata)...")
        grouped = df.groupby("CaseID")
        has_timestamp = "Timestamp" in df.columns

        for case_id, group in grouped:
            group = group.sort_values("Timestamp") if has_timestamp else group
            acts = group["Activity"].tolist()
            ress = group["Resource"].tolist() if "Resource" in group.columns else ["[UNK]"] * len(group)
            time_lasts = group["TimeSinceLast"].tolist()
            time_starts = group["TimeSinceStart"].tolist()
            next_acts = group["Next_Activity"].tolist()
            next_times = group["Next_Event_Time"].tolist()
            rem_times = group["Remaining_Time"].tolist()

            if has_timestamp:
                timestamps = pd.to_datetime(group["Timestamp"], errors="coerce").tolist()
            else:
                timestamps = [pd.NaT] * len(group)

            total_events = len(group)
            start_idx = 1
            if self.max_prefixes_per_case and total_events > self.max_prefixes_per_case:
                start_idx = total_events - self.max_prefixes_per_case + 1

            for i in range(start_idx, total_events + 1):
                s = max(0, i - self.max_seq_len)
                prefix_acts_raw = acts[s:i]
                prefix_ress_raw = ress[s:i]
                prefix_time_lasts = time_lasts[s:i]
                prefix_time_starts = time_starts[s:i]
                prefix_timestamps = timestamps[s:i]

                act_ids = [self.vocab.get_act_id(a) for a in prefix_acts_raw]
                res_ids = [self.vocab.get_res_id(r) for r in prefix_ress_raw]
                variant_key = "->".join([str(a) for a in prefix_acts_raw])

                self.prefixes.append(
                    {
                        "case_id": case_id,
                        "act_seq": act_ids,
                        "res_seq": res_ids,
                        "time_last_seq": prefix_time_lasts,
                        "time_start_seq": prefix_time_starts,
                        "timestamps": prefix_timestamps,
                        "target_act": self.vocab.get_act_id(next_acts[i - 1]),
                        "target_next_time": next_times[i - 1],
                        "target_rem_time": rem_times[i - 1],
                        "variant_key": variant_key,
                    }
                )

        print(f" -> 成功生成了 {len(self.prefixes)} 个增强前缀样本。")

    def _calendar_buckets(self, timestamps, seq_len: int) -> List[int]:
        buckets = []
        for ts in timestamps:
            if pd.isna(ts):
                buckets.append(0)
            else:
                buckets.append(int(ts.weekday()) * 24 + int(ts.hour))
        return buckets[:seq_len]

    def _prefix_stat_features(self, item, seq_len: int) -> np.ndarray:
        delta = _safe_float_array(item["time_last_seq"])
        if delta.size == 0:
            delta = np.asarray([0.0], dtype=np.float32)

        act_ids = [int(x) for x in item["act_seq"]]
        res_ids = [int(x) for x in item["res_seq"]]
        recent3 = delta[-3:]
        recent5 = delta[-5:]

        unique_act = len(set(act_ids)) if act_ids else 0
        unique_res = len(set(res_ids)) if res_ids else 0
        activity_repeat = sum(1 for a, b in zip(act_ids[:-1], act_ids[1:]) if a == b)
        resource_switch = sum(1 for a, b in zip(res_ids[:-1], res_ids[1:]) if a != b)
        recent_act = act_ids[-3:]
        recent_activity_repeat = sum(1 for a, b in zip(recent_act[:-1], recent_act[1:]) if a == b)
        recent_res = res_ids[-3:]
        recent_resource_switch = sum(1 for a, b in zip(recent_res[:-1], recent_res[1:]) if a != b)

        timestamps = item.get("timestamps", [])
        last_ts = timestamps[-1] if timestamps else pd.NaT
        if pd.isna(last_ts):
            hour = 0.0
            weekday = 0.0
            is_weekend = 0.0
        else:
            hour = float(last_ts.hour) / 23.0
            weekday = float(last_ts.weekday()) / 6.0
            is_weekend = 1.0 if int(last_ts.weekday()) >= 5 else 0.0

        max_len = max(1, int(self.max_seq_len))
        features = np.asarray(
            [
                float(np.mean(delta)),
                float(np.std(delta)),
                float(np.max(delta)),
                float(np.min(delta)),
                float(np.median(delta)),
                _ewma_last(delta),
                _skewness(delta),
                _kurtosis(delta),
                float(np.mean(recent3)),
                float(np.std(recent3)),
                float(np.mean(recent5)),
                float(np.std(recent5)),
                float(unique_act),
                float(unique_act) / max(float(seq_len), 1.0),
                float(activity_repeat),
                float(_count_non_adjacent_repeats(act_ids)),
                float(recent_activity_repeat),
                _transition_entropy(act_ids),
                float(unique_res),
                float(unique_res) / max(float(seq_len), 1.0),
                float(resource_switch),
                float(resource_switch) / max(float(seq_len - 1), 1.0),
                float(recent_resource_switch),
                float(seq_len),
                float(seq_len) / float(max_len),
                hour,
                weekday,
                is_weekend,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def __getitem__(self, idx):
        item = self.prefixes[idx]
        base = super().__getitem__(idx)
        seq_len = int(base["mask"].sum().item())
        pad_len = self.max_seq_len - seq_len

        calendar = self._calendar_buckets(item.get("timestamps", []), seq_len)
        calendar = calendar + [0] * max(0, pad_len)
        positions = np.arange(self.max_seq_len, dtype=np.float32)
        progress = np.floor(positions / max(float(self.max_seq_len), 1.0) * 20.0).astype(np.int64)
        progress = np.clip(progress, 0, 19).tolist()

        base["prefix_stat_feats"] = torch.tensor(self._prefix_stat_features(item, seq_len), dtype=torch.float32)
        base["calendar_bucket_ids"] = torch.tensor(calendar[: self.max_seq_len], dtype=torch.long)
        base["progress_bucket_ids"] = torch.tensor(progress, dtype=torch.long)
        base["prefix_len"] = torch.tensor(seq_len, dtype=torch.long)
        return base


# -----------------------------------------------------------------------------
# Model blocks
# -----------------------------------------------------------------------------


class EventEmbedding(nn.Module):
    def __init__(self, vocab_size_act, vocab_size_res, num_num_features=2, d_model=128, dropout=0.1):
        super().__init__()
        self.act_emb = nn.Embedding(vocab_size_act, d_model, padding_idx=0)
        self.res_emb = nn.Embedding(vocab_size_res, d_model, padding_idx=0)
        self.num_proj = nn.Sequential(nn.Linear(num_num_features, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, act_seq, res_seq, num_feats):
        safe_num = torch.log1p(torch.clamp(num_feats, min=0.0))
        x = self.act_emb(act_seq) + self.res_emb(res_seq) + self.num_proj(safe_num)
        return self.dropout(self.norm(x))


class LocalTCN(nn.Module):
    def __init__(self, d_model=128, num_layers=2, kernel_size=3, dropout=0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "conv1": nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model),
                        "conv2": nn.Conv1d(d_model, d_model, 1),
                        "dropout": nn.Dropout(dropout),
                        "norm": nn.LayerNorm(d_model),
                    }
                )
            )

    def forward(self, h, mask):
        valid = mask.unsqueeze(-1).to(h.dtype)
        out = h * valid
        for layer in self.layers:
            residual = out
            x = out.transpose(1, 2)
            x = layer["conv1"](x)
            x = F.gelu(x)
            x = layer["dropout"](x)
            x = layer["conv2"](x).transpose(1, 2)
            x = layer["dropout"](x)
            out = layer["norm"](residual + x) * valid
        return out


class EventAlignedGTR(nn.Module):
    def __init__(self, d_model=128, num_buckets=168, conv_kernel_size=3, dropout=0.1, use_residual_gate=True):
        super().__init__()
        if conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd")
        self.global_memory = nn.Embedding(num_buckets, d_model)
        nn.init.trunc_normal_(self.global_memory.weight, std=0.02)
        self.conv = nn.Sequential(
            nn.Conv1d(2 * d_model, d_model, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.use_residual_gate = use_residual_gate
        self.gate = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, bucket_ids, mask):
        valid = mask.unsqueeze(-1).to(h.dtype)
        q = self.global_memory(bucket_ids.clamp(min=0, max=self.global_memory.num_embeddings - 1))
        z = torch.cat([h, q], dim=-1).transpose(1, 2)
        out = self.conv(z).transpose(1, 2)
        if self.use_residual_gate:
            gate = torch.sigmoid(self.gate(torch.cat([h, out], dim=-1)))
            out = gate * out
        return self.norm(h + self.dropout(out)) * valid


class DualRetrieverFusion(nn.Module):
    def __init__(self, d_model=128, num_calendar_buckets=168, num_progress_buckets=20, dropout=0.1):
        super().__init__()
        self.calendar_gtr = EventAlignedGTR(d_model=d_model, num_buckets=num_calendar_buckets, dropout=dropout)
        self.progress_gtr = EventAlignedGTR(d_model=d_model, num_buckets=num_progress_buckets, dropout=dropout)
        self.gate_mlp = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, calendar_bucket_ids, progress_bucket_ids, mask):
        h_cal = self.calendar_gtr(h, calendar_bucket_ids, mask)
        h_prog = self.progress_gtr(h, progress_bucket_ids, mask)
        gate = torch.sigmoid(self.gate_mlp(torch.cat([h, h_cal, h_prog], dim=-1)))
        out = gate * h_cal + (1.0 - gate) * h_prog
        valid = mask.unsqueeze(-1).to(h.dtype)
        return self.norm(h + self.dropout(out)) * valid


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class PGRTCModel(nn.Module):
    def __init__(
        self,
        vocab_size_act,
        vocab_size_res,
        num_num_features=2,
        num_prefix_stat_features=28,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=50,
        use_prefix_stat=True,
        use_local_tcn=True,
        use_calendar_gtr=False,
        use_progress_gtr=False,
        use_dual_gtr=True,
    ):
        super().__init__()
        self.use_prefix_stat = use_prefix_stat
        self.use_local_tcn = use_local_tcn
        self.use_calendar_gtr = use_calendar_gtr
        self.use_progress_gtr = use_progress_gtr
        self.use_dual_gtr = use_dual_gtr

        self.embedding = EventEmbedding(vocab_size_act, vocab_size_res, num_num_features, d_model, dropout)
        self.local_tcn = LocalTCN(d_model=d_model, num_layers=2, kernel_size=3, dropout=dropout)
        self.calendar_gtr = EventAlignedGTR(d_model=d_model, num_buckets=168, dropout=dropout)
        self.progress_gtr = EventAlignedGTR(d_model=d_model, num_buckets=20, dropout=dropout)
        self.dual_gtr = DualRetrieverFusion(d_model=d_model, num_calendar_buckets=168, num_progress_buckets=20, dropout=dropout)
        self.pos = PositionalEncoding(d_model, max_len=max_seq_len + 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        head_in = d_model + (num_prefix_stat_features if use_prefix_stat else 0)
        self.regressor = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, max(d_model // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_model // 2, 32), 1),
        )

    @staticmethod
    def _last_valid(h, mask):
        lengths = mask.long().sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        return h.gather(dim=1, index=idx).squeeze(1)

    def forward(self, batch):
        act = batch["act_seq"]
        res = batch["res_seq"]
        time_seq = batch["time_seq"]
        mask = batch["mask"].bool()
        h = self.embedding(act, res, time_seq)
        if self.use_local_tcn:
            h = self.local_tcn(h, mask)
        if self.use_dual_gtr:
            h = self.dual_gtr(h, batch["calendar_bucket_ids"], batch["progress_bucket_ids"], mask)
        else:
            if self.use_calendar_gtr:
                h = self.calendar_gtr(h, batch["calendar_bucket_ids"], mask)
            if self.use_progress_gtr:
                h = self.progress_gtr(h, batch["progress_bucket_ids"], mask)
        h = self.pos(h)
        h = self.encoder(h, src_key_padding_mask=~mask)
        pooled = self._last_valid(h, mask)
        if self.use_prefix_stat:
            pooled = torch.cat([pooled, batch["prefix_stat_feats"]], dim=-1)
        return self.regressor(pooled).squeeze(-1)


class TailWeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0, alpha=0.5, use_log_target=True):
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.use_log_target = use_log_target

    def forward(self, pred, target):
        if self.use_log_target:
            target_loss = torch.log1p(torch.clamp(target, min=0.0))
        else:
            target_loss = target
        base = F.huber_loss(pred, target_loss, delta=self.delta, reduction="none")
        tail_signal = torch.log1p(torch.clamp(target, min=0.0))
        weight = 1.0 + self.alpha * tail_signal / tail_signal.mean().clamp(min=1e-6)
        weight = weight / weight.mean().clamp(min=1e-6)
        return (weight * base).mean()


def _move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def evaluate_pgr_tc_model(model, data_loader, device, tail_q1, tail_q2, use_log_target=True):
    model.eval()
    abs_errors, sq_errors, tail_errors = [], [], []
    y_true_all, y_pred_all, variant_freq_all, prefix_len_all = [], [], [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = _move_batch(batch, device)
            y_true = batch["target_rem_time"]
            y_pred = model(batch)
            if use_log_target:
                y_pred = torch.expm1(y_pred).clamp(min=0.0)
            err = torch.abs(y_pred - y_true)
            abs_errors.append(err.cpu())
            sq_errors.append(((y_pred - y_true) ** 2).cpu())
            y_true_all.append(y_true.cpu())
            y_pred_all.append(y_pred.cpu())
            variant_freq_all.append(batch["variant_freq"].cpu())
            prefix_len_all.append(batch["prefix_len"].cpu())
            tail_m = batch_tail_mask(batch["variant_freq"], tail_q1)
            if tail_m.any():
                tail_errors.append(err[tail_m].cpu())
    abs_cat = torch.cat(abs_errors)
    sq_cat = torch.cat(sq_errors)
    mae = abs_cat.mean().item()
    rmse = torch.sqrt(sq_cat.mean()).item()
    tail_mae = torch.cat(tail_errors).mean().item() if tail_errors else mae
    y_true_np = torch.cat(y_true_all).numpy()
    y_pred_np = torch.cat(y_pred_all).numpy()
    variant_np = torch.cat(variant_freq_all).numpy()
    prefix_np = torch.cat(prefix_len_all).numpy()
    bucket_report = evaluate_bucket_regression(y_true_np, y_pred_np, variant_np, tail_q1, tail_q2)
    prefix_report = {}
    for name, m in {
        "short": prefix_np <= 3,
        "middle": (prefix_np >= 4) & (prefix_np <= 7),
        "long": prefix_np > 7,
    }.items():
        prefix_report[name] = float(np.mean(np.abs(y_pred_np[m] - y_true_np[m]))) if np.any(m) else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "tail_mae": tail_mae,
        "bucket_report": bucket_report,
        "prefix_report": prefix_report,
        "score": 0.5 * mae + 0.5 * tail_mae,
    }


def train_pgr_tc_model(model, train_loader, val_loader, epochs, lr, device, tail_q1, tail_q2, use_log_target=True):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = TailWeightedHuberLoss(delta=1.0, alpha=0.5, use_log_target=use_log_target)
    best_metrics, best_score = None, float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss, n = 0.0, 0
        start = time.time()
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch["target_rem_time"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            bs = batch["act_seq"].size(0)
            train_loss += loss.item() * bs
            n += bs
        metrics = evaluate_pgr_tc_model(model, val_loader, device, tail_q1, tail_q2, use_log_target=use_log_target)
        elapsed = time.time() - start
        print(
            f"    Epoch {epoch + 1:02d}/{epochs} | TrainLoss={train_loss / max(n, 1):.4f} | "
            f"ValMAE={metrics['mae']:.4f} | TailMAE={metrics['tail_mae']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | Score={metrics['score']:.4f} | Time={elapsed:.1f}s"
        )
        if metrics["score"] < best_score:
            best_score = metrics["score"]
            best_metrics = metrics
    return best_metrics


def append_bucket_rows(rows, seed, model_name, metrics):
    report = metrics.get("bucket_report", {})
    q1 = report.get("quantiles", {}).get("q1", 0.0)
    q2 = report.get("quantiles", {}).get("q2", 0.0)
    for bucket in ["overall", "head", "torso", "tail"]:
        item = report.get(bucket, {"count": 0.0, "mae": 0.0, "rmse": 0.0})
        rows.append({"seed": seed, "model": model_name, "bucket": bucket, "count": item.get("count", 0.0), "mae": item.get("mae", 0.0), "rmse": item.get("rmse", 0.0), "q1": q1, "q2": q2})


def build_summary(results_df):
    rows = []
    for name in results_df["model"].unique():
        sub = results_df[results_df["model"] == name]
        rows.append(
            {
                "model": name,
                "mae_mean": sub["mae"].mean(),
                "mae_std": sub["mae"].std(),
                "rmse_mean": sub["rmse"].mean(),
                "rmse_std": sub["rmse"].std(),
                "tail_mae_mean": sub["tail_mae"].mean(),
                "tail_mae_std": sub["tail_mae"].std(),
                "score_mean": sub["score"].mean(),
                "score_std": sub["score"].std(),
                "mae_short_mean": sub["mae_short"].mean(),
                "mae_middle_mean": sub["mae_middle"].mean(),
                "mae_long_mean": sub["mae_long"].mean(),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    DATASET_PATH = "dataset/processed_BPIC2015_1.csv"
    REPORT_DIR = "results"
    TRAIN_SPLIT_RATIO = 0.8
    SPLIT_STRATEGY = "case"
    MAX_SEQ_LENGTH = 50
    MAX_PREFIXES = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    EPOCHS = 30
    D_MODEL = 128
    NUM_HEADS = 8
    NUM_LAYERS = 2
    RF_N_ESTIMATORS = 300
    RF_MAX_DEPTH = None
    RF_N_JOBS = -1
    SEEDS = [42, 67, 80, 89]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(REPORT_DIR, exist_ok=True)

    print(f"Dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    vocab = ProcessVocab()
    vocab.build_vocab(df)
    train_df, val_df = split_dataframe(df, train_ratio=TRAIN_SPLIT_RATIO, strategy=SPLIT_STRATEGY, case_col="CaseID", time_col="Timestamp")
    train_dataset = EnhancedProcessPrefixDataset(train_df, vocab, max_seq_len=MAX_SEQ_LENGTH, max_prefixes_per_case=MAX_PREFIXES, fit_normalization=True)
    norm_stats = train_dataset.get_normalization_stats()
    val_dataset = EnhancedProcessPrefixDataset(val_df, vocab, max_seq_len=MAX_SEQ_LENGTH, max_prefixes_per_case=MAX_PREFIXES, normalization_stats=norm_stats, fit_normalization=False)
    train_variant_freq = np.asarray([float(x.get("variant_freq", 1.0)) for x in train_dataset.prefixes], dtype=np.float32)
    tail_q1 = float(np.quantile(train_variant_freq, 0.33)) if len(train_variant_freq) > 0 else 1.0
    tail_q2 = float(np.quantile(train_variant_freq, 0.66)) if len(train_variant_freq) > 0 else tail_q1
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    def build_pgr(**kwargs):
        return PGRTCModel(
            vocab_size_act=len(vocab.act2id),
            vocab_size_res=len(vocab.res2id),
            num_num_features=2,
            num_prefix_stat_features=len(EnhancedProcessPrefixDataset.prefix_stat_feature_names),
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            num_layers=NUM_LAYERS,
            max_seq_len=MAX_SEQ_LENGTH,
            **kwargs,
        )

    model_configs = [
        {"name": "VanillaTransformer", "kind": "dl_baseline", "build": lambda: VanillaTransformerBaseline(len(vocab.act2id), len(vocab.res2id), d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)},
        {"name": "LSTM", "kind": "dl_baseline", "build": lambda: LSTMBaseline(len(vocab.act2id), len(vocab.res2id), d_model=D_MODEL, num_layers=2)},
        {"name": "RandomForest", "kind": "rf"},
        {"name": "PGR_Transformer", "kind": "pgr", "build": lambda: build_pgr(use_prefix_stat=False, use_local_tcn=False, use_dual_gtr=False)},
        {"name": "PGR_PrefixStat", "kind": "pgr", "build": lambda: build_pgr(use_prefix_stat=True, use_local_tcn=False, use_dual_gtr=False)},
        {"name": "PGR_LocalTCN", "kind": "pgr", "build": lambda: build_pgr(use_prefix_stat=False, use_local_tcn=True, use_dual_gtr=False)},
        {"name": "PGR_CalendarGTR", "kind": "pgr", "build": lambda: build_pgr(use_prefix_stat=False, use_local_tcn=False, use_calendar_gtr=True, use_dual_gtr=False)},
        {"name": "PGR_ProgressGTR", "kind": "pgr", "build": lambda: build_pgr(use_prefix_stat=False, use_local_tcn=False, use_progress_gtr=True, use_dual_gtr=False)},
        {"name": "PGR_DualGTR", "kind": "pgr", "build": lambda: build_pgr(use_prefix_stat=False, use_local_tcn=False, use_dual_gtr=True)},
        {"name": "PGR_TC_MVP", "kind": "pgr", "build": lambda: build_pgr(use_prefix_stat=True, use_local_tcn=True, use_dual_gtr=True)},
    ]

    all_results, all_bucket_rows = [], []
    total_runs, run_idx = len(SEEDS) * len(model_configs), 0
    for seed in SEEDS:
        set_seed(seed)
        print(f"\n{'=' * 60}\n  Seed = {seed}\n{'=' * 60}")
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
        for cfg in model_configs:
            run_idx += 1
            name = cfg["name"]
            print(f"\n[{run_idx}/{total_runs}] {name} | seed={seed}")
            if cfg["kind"] == "rf":
                metrics = train_random_forest(train_dataset, val_dataset, tail_q1, tail_q2, RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_N_JOBS, seed)
            elif cfg["kind"] == "dl_baseline":
                model = cfg["build"]()
                metrics = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, DEVICE, tail_q1, tail_q2)
                del model
            else:
                model = cfg["build"]()
                metrics = train_pgr_tc_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, DEVICE, tail_q1, tail_q2, use_log_target=True)
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            prefix_report = metrics.get("prefix_report", {})
            all_results.append(
                {
                    "seed": seed,
                    "model": name,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "tail_mae": metrics["tail_mae"],
                    "score": metrics["score"],
                    "mae_short": prefix_report.get("short", np.nan),
                    "mae_middle": prefix_report.get("middle", np.nan),
                    "mae_long": prefix_report.get("long", np.nan),
                }
            )
            append_bucket_rows(all_bucket_rows, seed, name, metrics)
            print(f"     [*] {name} done | MAE={metrics['mae']:.4f} | TailMAE={metrics['tail_mae']:.4f} | RMSE={metrics['rmse']:.4f} | Score={metrics['score']:.4f}")

    results_df = pd.DataFrame(all_results)
    summary_df = build_summary(results_df)
    per_seed_path = os.path.join(REPORT_DIR, "pgr_tc_per_seed.csv")
    summary_path = os.path.join(REPORT_DIR, "pgr_tc_summary.csv")
    bucket_path = os.path.join(REPORT_DIR, "pgr_tc_bucket_metrics.csv")
    features_path = os.path.join(REPORT_DIR, "pgr_tc_prefix_stat_feature_names.csv")
    results_df.to_csv(per_seed_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(all_bucket_rows).to_csv(bucket_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": EnhancedProcessPrefixDataset.prefix_stat_feature_names}).to_csv(features_path, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 90)
    print(summary_df)
    print(f"Saved per-seed results: {per_seed_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved bucket metrics: {bucket_path}")
    print(f"Saved feature names: {features_path}")
