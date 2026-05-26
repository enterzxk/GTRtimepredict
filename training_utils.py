import math
import random
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import BatchSampler


def compute_variant_weights(variant_freq: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = torch.pow(variant_freq + eps, -0.5)
    return w / w.mean().clamp(min=eps)


def hybrid_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y_true: torch.Tensor,
    variant_freq: torch.Tensor,
    alpha: float = 0.6,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w = compute_variant_weights(variant_freq, eps=eps)
    l1 = (w * torch.abs(y_true - mu)).mean()

    # Keep heteroscedastic NLL numerically stable for large-scale remaining-time targets.
    target_scale = torch.clamp(y_true.detach().std(unbiased=False), min=1.0)
    sigma_safe = sigma.clamp(min=0.1 * target_scale, max=10.0 * target_scale)
    z2 = ((y_true - mu) / (sigma_safe + eps)) ** 2
    z2 = torch.clamp(z2, max=100.0)
    nll = (w * (0.5 * z2 + torch.log(sigma_safe + eps))).mean()

    loss = alpha * l1 + (1.0 - alpha) * nll
    return loss, l1, nll


def _rank_bucket_indices(freq: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
    n = int(freq.size)
    if n == 0:
        return [], [], []

    order = np.argsort(freq, kind="stable")

    if n == 1:
        idx = int(order[0])
        return [idx], [idx], [idx]

    if n == 2:
        low = int(order[0])
        high = int(order[1])
        return [high], [low], [low]

    n_tail = max(1, n // 3)
    n_torso = max(1, n // 3)
    n_head = n - n_tail - n_torso
    if n_head <= 0:
        n_head = 1
        n_torso = max(1, n - n_tail - n_head)

    s1 = n_tail
    s2 = n_tail + n_torso
    tail = order[:s1].tolist()
    torso = order[s1:s2].tolist()
    head = order[s2:].tolist()
    return head, torso, tail


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_activity_pair_count_matrix(prefix_dataset, vocab_size: int) -> torch.Tensor:
    """
    Count pair frequency count(a_i, a_j) across training prefixes.
    """
    pair_counts = torch.zeros((vocab_size, vocab_size), dtype=torch.long)

    prefixes = getattr(prefix_dataset, "prefixes", None)
    if prefixes is None:
        return pair_counts

    for item in prefixes:
        act_seq = item.get("act_seq", None)
        if act_seq is None:
            continue

        seq_len = len(act_seq)
        for i in range(seq_len):
            ai = int(act_seq[i])
            if ai < 0 or ai >= vocab_size:
                continue
            for j in range(seq_len):
                aj = int(act_seq[j])
                if aj < 0 or aj >= vocab_size:
                    continue
                pair_counts[ai, aj] += 1

    return pair_counts


class VariantBucketBatchSampler(BatchSampler):
    """
    Fixed head/torso/tail ratio batch sampler, default 4:3:3.
    Buckets are split by variant frequency quantiles.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        ratio: Sequence[int] = (4, 3, 3),
        drop_last: bool = False,
        seed: int = 2026,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.indices = list(range(len(dataset)))
        self.head_idx, self.torso_idx, self.tail_idx = self._build_buckets()

        ratio = list(ratio)
        if len(ratio) != 3 or sum(ratio) <= 0:
            raise ValueError("ratio must have 3 positive parts, e.g. (4,3,3)")

        ratio_sum = float(sum(ratio))
        self.n_head = max(1, int(round(self.batch_size * ratio[0] / ratio_sum)))
        self.n_torso = max(1, int(round(self.batch_size * ratio[1] / ratio_sum)))
        self.n_tail = self.batch_size - self.n_head - self.n_torso
        if self.n_tail <= 0:
            self.n_tail = 1
            self.n_head = max(1, self.n_head - 1)

    def _variant_freq_array(self) -> np.ndarray:
        prefixes = getattr(self.dataset, "prefixes", None)
        if prefixes is None or len(prefixes) == 0:
            return np.ones((len(self.dataset),), dtype=np.float32)

        freq = np.zeros((len(prefixes),), dtype=np.float32)
        for i, item in enumerate(prefixes):
            freq[i] = float(item.get("variant_freq", 1.0))
        return freq

    def _build_buckets(self):
        freq = self._variant_freq_array()
        if freq.size == 0:
            return [], [], []

        q1 = float(np.quantile(freq, 0.33))
        q2 = float(np.quantile(freq, 0.66))

        if q1 < q2:
            tail = [i for i, f in enumerate(freq) if f <= q1]
            torso = [i for i, f in enumerate(freq) if q1 < f <= q2]
            head = [i for i, f in enumerate(freq) if f > q2]

            if len(head) == 0 or len(torso) == 0 or len(tail) == 0:
                head, torso, tail = _rank_bucket_indices(freq)
        else:
            # Heavy duplicate frequencies (e.g., q1 == q2) cause quantile collapse.
            head, torso, tail = _rank_bucket_indices(freq)

        return head, torso, tail

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample_from_bucket(self, rng: random.Random, bucket: List[int], n: int) -> List[int]:
        if len(bucket) >= n:
            return rng.sample(bucket, n)
        return [rng.choice(bucket) for _ in range(n)]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        n_batches = len(self)
        for _ in range(n_batches):
            batch = []
            batch.extend(self._sample_from_bucket(rng, self.head_idx, self.n_head))
            batch.extend(self._sample_from_bucket(rng, self.torso_idx, self.n_torso))
            batch.extend(self._sample_from_bucket(rng, self.tail_idx, self.n_tail))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        size = len(self.dataset)
        if self.drop_last:
            return size // self.batch_size
        return (size + self.batch_size - 1) // self.batch_size


def batch_tail_mask(variant_freq: torch.Tensor, tail_threshold: float) -> torch.Tensor:
    return variant_freq <= tail_threshold


def split_dataframe(
    df,
    train_ratio: float = 0.8,
    strategy: str = "row",
    case_col: str = "CaseID",
    time_col: str = "Timestamp",
):
    """
    Split dataframe by row or by case-start time.
    strategy:
      - row: legacy df.iloc split
      - case: split by case IDs ordered by start timestamp
    """
    train_ratio = float(min(max(train_ratio, 0.05), 0.95))
    strategy = str(strategy).lower().strip()

    if strategy == "row":
        split_idx = int(len(df) * train_ratio)
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    if case_col not in df.columns:
        raise ValueError(f"Missing case column: {case_col}")

    if strategy != "case":
        raise ValueError("strategy must be 'row' or 'case'")

    if time_col in df.columns:
        ts = df[[case_col, time_col]].copy()
        ts[time_col] = pd.to_datetime(ts[time_col], errors="coerce")
        case_start = ts.groupby(case_col)[time_col].min().sort_values(kind="stable")
        case_ids = case_start.index.tolist()
    else:
        case_ids = df[case_col].drop_duplicates().tolist()

    split_case_idx = int(len(case_ids) * train_ratio)
    train_case_ids = set(case_ids[:split_case_idx])
    train_df = df[df[case_col].isin(train_case_ids)].copy()
    val_df = df[~df[case_col].isin(train_case_ids)].copy()
    return train_df, val_df


def evaluate_bucket_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variant_freq: np.ndarray,
    q1: Optional[float] = None,
    q2: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    variant_freq = np.asarray(variant_freq, dtype=np.float64)

    if y_true.size == 0:
        return {
            "overall": {"count": 0.0, "mae": 0.0, "rmse": 0.0},
            "head": {"count": 0.0, "mae": 0.0, "rmse": 0.0},
            "torso": {"count": 0.0, "mae": 0.0, "rmse": 0.0},
            "tail": {"count": 0.0, "mae": 0.0, "rmse": 0.0},
            "quantiles": {"q1": 0.0, "q2": 0.0},
        }

    if q1 is None:
        q1 = float(np.quantile(variant_freq, 0.33))
    if q2 is None:
        q2 = float(np.quantile(variant_freq, 0.66))

    abs_err = np.abs(y_pred - y_true)
    sq_err = (y_pred - y_true) ** 2

    def _bucket(mask):
        if mask.sum() == 0:
            return {"count": 0.0, "mae": 0.0, "rmse": 0.0}
        return {
            "count": float(mask.sum()),
            "mae": float(abs_err[mask].mean()),
            "rmse": float(np.sqrt(sq_err[mask].mean())),
        }

    if q1 < q2:
        tail_mask = variant_freq <= q1
        torso_mask = (variant_freq > q1) & (variant_freq <= q2)
        head_mask = variant_freq > q2

        if tail_mask.sum() == 0 or torso_mask.sum() == 0 or head_mask.sum() == 0:
            head_idx, torso_idx, tail_idx = _rank_bucket_indices(variant_freq)
            head_mask = np.zeros_like(variant_freq, dtype=bool)
            torso_mask = np.zeros_like(variant_freq, dtype=bool)
            tail_mask = np.zeros_like(variant_freq, dtype=bool)
            head_mask[head_idx] = True
            torso_mask[torso_idx] = True
            tail_mask[tail_idx] = True
    else:
        head_idx, torso_idx, tail_idx = _rank_bucket_indices(variant_freq)
        head_mask = np.zeros_like(variant_freq, dtype=bool)
        torso_mask = np.zeros_like(variant_freq, dtype=bool)
        tail_mask = np.zeros_like(variant_freq, dtype=bool)
        head_mask[head_idx] = True
        torso_mask[torso_idx] = True
        tail_mask[tail_idx] = True

    return {
        "overall": {
            "count": float(y_true.size),
            "mae": float(abs_err.mean()),
            "rmse": float(np.sqrt(sq_err.mean())),
        },
        "head": _bucket(head_mask),
        "torso": _bucket(torso_mask),
        "tail": _bucket(tail_mask),
        "quantiles": {"q1": float(q1), "q2": float(q2)},
    }


def save_bucket_report(report_path: str, model_name: str, report: Dict[str, Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    rows = []
    for bucket in ["overall", "head", "torso", "tail"]:
        item = report.get(bucket, {"count": 0.0, "mae": 0.0, "rmse": 0.0})
        rows.append(
            {
                "model": model_name,
                "bucket": bucket,
                "count": item.get("count", 0.0),
                "mae": item.get("mae", 0.0),
                "rmse": item.get("rmse", 0.0),
                "q1": report.get("quantiles", {}).get("q1", 0.0),
                "q2": report.get("quantiles", {}).get("q2", 0.0),
            }
        )

    fieldnames = ["model", "bucket", "count", "mae", "rmse", "q1", "q2"]
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Optional dependency import kept at bottom to avoid unnecessary heavy import if unused.
import pandas as pd
