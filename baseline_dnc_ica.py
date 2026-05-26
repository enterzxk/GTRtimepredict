# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline import (
    LSTMBaseline,
    evaluate_model,
    set_seed,
    train_model,
    train_random_forest,
)
from model import VanillaTransformerBaseline
from prefix import ProcessPrefixDataset, ProcessVocab
from training_utils import split_dataframe


class LocalGlobalDNCConv(nn.Module):
    """Local-global denoising convolution inspired by GTR-style retrieval."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 50,
        local_kernel_size: int = 5,
        global_kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        if local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be odd")
        if global_kernel_size % 2 == 0:
            raise ValueError("global_kernel_size must be odd")

        self.d_model = d_model
        self.max_len = max(1, int(max_len))

        self.local_denoise = nn.Sequential(
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=local_kernel_size,
                padding=local_kernel_size // 2,
                groups=d_model,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.global_memory = nn.Parameter(torch.zeros(self.max_len, d_model))
        nn.init.trunc_normal_(self.global_memory, std=0.02)
        self.global_proj = nn.Linear(d_model, d_model)

        self.local_global_fuse = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(2, global_kernel_size),
            padding=(0, global_kernel_size // 2),
            bias=False,
        )
        self.gate = nn.Linear(2 * d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _retrieve_global(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device) % self.max_len
        global_tokens = self.global_proj(self.global_memory[pos])
        return global_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.unsqueeze(-1).to(x.dtype)
        x = x * valid

        batch_size, seq_len, _ = x.shape
        local_feat = self.local_denoise(x.transpose(1, 2)).transpose(1, 2)
        global_feat = self._retrieve_global(batch_size, seq_len, x.device)

        stacked = torch.stack(
            [local_feat.transpose(1, 2), global_feat.transpose(1, 2)],
            dim=2,
        )
        fused = self.local_global_fuse(stacked.reshape(batch_size * self.d_model, 1, 2, seq_len))
        fused = fused.reshape(batch_size, self.d_model, seq_len).transpose(1, 2)

        gate = torch.sigmoid(self.gate(torch.cat([local_feat, global_feat], dim=-1)))
        out = self.out_proj(local_feat + gate * fused)
        return self.norm(x + self.dropout(out)) * valid


class ICAMixer(nn.Module):
    """Instance-channel adaptive mixer used instead of quadratic self-attention."""

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.local_mixer = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.token_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.unsqueeze(-1).to(x.dtype)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (x * valid).sum(dim=1) / denom

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.unsqueeze(-1).to(x.dtype)
        x = x * valid

        context = self._masked_mean(x, mask)
        local_feat = self.local_mixer(x.transpose(1, 2)).transpose(1, 2)
        channel = self.channel_gate(context).unsqueeze(1)
        token = self.token_gate(x)

        mixed = self.out_proj(channel * local_feat + token * x)
        mixed = self.norm(x + self.dropout(mixed)) * valid

        scores = self.score(mixed).squeeze(-1).masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.sum(mixed * weights.unsqueeze(-1), dim=1)


class CausalTemporalFeatureBuilder(nn.Module):
    """Build robust causal time features and impute abnormal values."""

    def __init__(self, time_dim: int = 2, eps: float = 1e-6):
        super().__init__()
        self.time_dim = time_dim
        self.eps = eps
        self.feature_groups = [
            "clean",
            "delta",
            "roll3_mean",
            "roll3_std",
            "roll3_median",
            "roll5_mean",
            "roll5_std",
            "ewma",
            "robust_z",
            "anomaly_flag",
            "missing_flag",
            "seq_skew",
            "seq_kurtosis",
        ]
        self.output_dim = len(self.feature_groups) * time_dim

    def feature_names(self):
        names = []
        for group in self.feature_groups:
            for idx in range(self.time_dim):
                names.append(f"{group}_t{idx}")
        return names

    def _valid_stats(self, x: torch.Tensor, mask: torch.Tensor):
        valid = mask.unsqueeze(-1).to(x.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (x * valid).sum(dim=1, keepdim=True) / denom
        var = (((x - mean) ** 2) * valid).sum(dim=1, keepdim=True) / denom
        std = torch.sqrt(var + self.eps)
        return mean, std, valid

    def _clean(self, time_features: torch.Tensor, mask: torch.Tensor):
        finite = torch.isfinite(time_features)
        safe = torch.nan_to_num(time_features, nan=0.0, posinf=0.0, neginf=0.0)
        safe = torch.clamp(safe, min=0.0)
        mean, std, valid = self._valid_stats(safe, mask)
        z = (safe - mean) / std.clamp(min=self.eps)
        anomaly = (torch.abs(z) > 6.0) & (mask.unsqueeze(-1) > 0)
        cleaned = torch.where(anomaly, mean.expand_as(safe), safe)
        missing = (~finite).to(safe.dtype)
        return cleaned * valid, z * valid, anomaly.to(safe.dtype), missing * valid

    def _causal_rolling(self, x: torch.Tensor, mask: torch.Tensor, window: int, with_median: bool = False):
        means = []
        stds = []
        medians = []
        for pos in range(x.size(1)):
            start = max(0, pos - window + 1)
            seg = x[:, start : pos + 1, :]
            seg_mask = mask[:, start : pos + 1].unsqueeze(-1).to(x.dtype)
            denom = seg_mask.sum(dim=1).clamp(min=1.0)
            mean = (seg * seg_mask).sum(dim=1) / denom
            var = (((seg - mean.unsqueeze(1)) ** 2) * seg_mask).sum(dim=1) / denom
            means.append(mean)
            stds.append(torch.sqrt(var + self.eps))
            if with_median:
                padded_seg = torch.where(seg_mask > 0, seg, mean.unsqueeze(1))
                medians.append(torch.median(padded_seg, dim=1).values)

        out = [torch.stack(means, dim=1), torch.stack(stds, dim=1)]
        if with_median:
            out.append(torch.stack(medians, dim=1))
        return out

    def _ewma(self, x: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5):
        values = []
        state = x[:, 0, :]
        for pos in range(x.size(1)):
            valid = mask[:, pos : pos + 1].to(x.dtype)
            current = x[:, pos, :]
            state = torch.where(valid > 0, alpha * current + (1.0 - alpha) * state, state)
            values.append(state)
        return torch.stack(values, dim=1)

    def forward(self, time_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        clean, robust_z, anomaly_flag, missing_flag = self._clean(time_features, mask)
        valid = mask.unsqueeze(-1).to(clean.dtype)

        delta = clean - torch.cat([clean[:, :1, :], clean[:, :-1, :]], dim=1)
        roll3_mean, roll3_std, roll3_median = self._causal_rolling(clean, mask, window=3, with_median=True)
        roll5_mean, roll5_std = self._causal_rolling(clean, mask, window=5, with_median=False)
        ewma = self._ewma(clean, mask)

        seq_mean, seq_std, _ = self._valid_stats(clean, mask)
        centered = (clean - seq_mean) * valid
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        seq_skew = (centered.pow(3).sum(dim=1, keepdim=True) / denom) / seq_std.pow(3).clamp(min=self.eps)
        seq_kurt = (centered.pow(4).sum(dim=1, keepdim=True) / denom) / seq_std.pow(4).clamp(min=self.eps)
        seq_skew = seq_skew.expand_as(clean)
        seq_kurt = seq_kurt.expand_as(clean)

        features = torch.cat(
            [
                clean,
                delta,
                roll3_mean,
                roll3_std,
                roll3_median,
                roll5_mean,
                roll5_std,
                ewma,
                robust_z,
                anomaly_flag,
                missing_flag,
                seq_skew,
                seq_kurt,
            ],
            dim=-1,
        )
        return features * valid


class FeatureGate(nn.Module):
    """Learnable feature-selection gate over constructed temporal variables."""

    def __init__(self, num_features: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.logits).view(1, 1, -1)

    def l1_penalty(self) -> torch.Tensor:
        return torch.sigmoid(self.logits).mean()

    def values(self) -> torch.Tensor:
        return torch.sigmoid(self.logits).detach().cpu()


class DNCICABaseline(nn.Module):
    """Baseline-style model with DNC convolution, LSTM backbone, and ICA readout."""

    def __init__(
        self,
        vocab_size_act: int,
        vocab_size_res: int,
        max_seq_len: int = 50,
        d_model: int = 128,
        lstm_layers: int = 2,
        dnc_layers: int = 2,
        dropout: float = 0.1,
        time_dim: int = 2,
    ):
        super().__init__()
        self.act_embedding = nn.Embedding(vocab_size_act, d_model, padding_idx=0)
        self.res_embedding = nn.Embedding(vocab_size_res, d_model, padding_idx=0)
        self.time_feature_builder = CausalTemporalFeatureBuilder(time_dim=time_dim)
        self.time_feature_gate = FeatureGate(self.time_feature_builder.output_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_feature_builder.output_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.dnc_layers = nn.ModuleList(
            [
                LocalGlobalDNCConv(
                    d_model=d_model,
                    max_len=max_seq_len,
                    local_kernel_size=5,
                    global_kernel_size=7,
                    dropout=dropout,
                )
                for _ in range(dnc_layers)
            ]
        )

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.ica_readout = ICAMixer(d_model=d_model, kernel_size=3, dropout=dropout)

        head_hidden = max(d_model // 2, 16)
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, act_seq, res_seq, time_features, mask):
        engineered_time = self.time_feature_builder(time_features, mask)
        selected_time = self.time_feature_gate(engineered_time)
        x = self.act_embedding(act_seq) + self.res_embedding(res_seq) + self.time_proj(selected_time)
        x = self.input_dropout(self.input_norm(x))

        for layer in self.dnc_layers:
            x = layer(x, mask)

        x, _ = self.lstm(x)
        h = self.ica_readout(x, mask)
        return self.regressor(h).squeeze(-1)

    def feature_gate_l1_penalty(self) -> torch.Tensor:
        return self.time_feature_gate.l1_penalty()

    def feature_gate_report(self):
        values = self.time_feature_gate.values().numpy()
        names = self.time_feature_builder.feature_names()
        return [{"feature": name, "gate": float(value)} for name, value in zip(names, values)]


def append_bucket_rows(all_bucket_rows, seed, model_name, metrics):
    report = metrics.get("bucket_report", {})
    q1 = report.get("quantiles", {}).get("q1", 0.0)
    q2 = report.get("quantiles", {}).get("q2", 0.0)
    for bucket in ["overall", "head", "torso", "tail"]:
        item = report.get(bucket, {"count": 0.0, "mae": 0.0, "rmse": 0.0})
        all_bucket_rows.append(
            {
                "seed": seed,
                "model": model_name,
                "bucket": bucket,
                "count": item.get("count", 0.0),
                "mae": item.get("mae", 0.0),
                "rmse": item.get("rmse", 0.0),
                "q1": q1,
                "q2": q2,
            }
        )


def append_feature_gate_rows(all_gate_rows, seed, model_name, model):
    if not hasattr(model, "feature_gate_report"):
        return
    for item in model.feature_gate_report():
        all_gate_rows.append(
            {
                "seed": seed,
                "model": model_name,
                "feature": item["feature"],
                "gate": item["gate"],
            }
        )


def adaptive_dal_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    variant_freq: torch.Tensor,
    time_features: torch.Tensor,
    mask: torch.Tensor,
    model: nn.Module,
    gate_lambda: float = 1e-4,
) -> torch.Tensor:
    abs_err = torch.abs(y_pred - y_true)
    huber = F.smooth_l1_loss(y_pred, y_true, reduction="none")

    rare_weight = torch.pow(variant_freq + 1e-6, -0.5)
    rare_weight = rare_weight / rare_weight.mean().clamp(min=1e-6)

    y_std = y_true.std(unbiased=False).clamp(min=1e-6)
    peak_weight = 1.0 + 0.35 * torch.sigmoid((y_true - y_true.mean()) / y_std)

    valid = mask.unsqueeze(-1).to(time_features.dtype)
    denom = valid.sum(dim=1).clamp(min=1.0)
    time_mean = (time_features * valid).sum(dim=1) / denom
    time_var = (((time_features - time_mean.unsqueeze(1)) ** 2) * valid).sum(dim=1) / denom
    volatility = torch.sqrt(time_var + 1e-6).mean(dim=-1)
    volatility_weight = 1.0 + 0.2 * volatility / volatility.mean().clamp(min=1e-6)

    weight = rare_weight * peak_weight * volatility_weight
    weight = weight / weight.mean().clamp(min=1e-6)

    loss = (weight * (0.7 * abs_err + 0.3 * huber)).mean()
    if hasattr(model, "feature_gate_l1_penalty"):
        loss = loss + gate_lambda * model.feature_gate_l1_penalty()
    return loss


def train_dnc_ica_model(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    tail_q1,
    tail_q2,
    gate_lambda=1e-4,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_metrics = None
    best_score = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n = 0
        start = time.time()

        for batch in train_loader:
            act = batch["act_seq"].to(device)
            res = batch["res_seq"].to(device)
            time_feat = batch["time_seq"].to(device)
            mask = batch["mask"].to(device)
            y_true = batch["target_rem_time"].to(device)
            variant_freq = batch["variant_freq"].to(device)

            optimizer.zero_grad()
            y_pred = model(act, res, time_feat, mask)
            loss = adaptive_dal_loss(
                y_pred=y_pred,
                y_true=y_true,
                variant_freq=variant_freq,
                time_features=time_feat,
                mask=mask,
                model=model,
                gate_lambda=gate_lambda,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = act.size(0)
            train_loss += loss.item() * bs
            n += bs

        train_loss /= max(n, 1)
        val_metrics = evaluate_model(model, val_loader, device, tail_q1, tail_q2)
        elapsed = time.time() - start

        print(
            f"    Epoch {epoch + 1:02d}/{epochs} | TrainDAL={train_loss:.4f} | "
            f"ValMAE={val_metrics['mae']:.4f} | TailMAE={val_metrics['tail_mae']:.4f} | "
            f"RMSE={val_metrics['rmse']:.4f} | Score={val_metrics['score']:.4f} | Time={elapsed:.1f}s"
        )

        if val_metrics["score"] < best_score:
            best_score = val_metrics["score"]
            best_metrics = val_metrics

    return best_metrics


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
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
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    DATASET_PATH = "dataset/processed_BPIC2015_1.csv"
    REPORT_DIR = "results"
    TRAIN_SPLIT_RATIO = 0.8
    SPLIT_STRATEGY = "case"  # row | case
    MAX_SEQ_LENGTH = 50
    MAX_PREFIXES = 100

    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    EPOCHS = 30
    FEATURE_GATE_LAMBDA = 1e-4

    D_MODEL = 128
    NUM_HEADS = 8
    NUM_LAYERS = 4
    LSTM_NUM_LAYERS = 2
    DNC_LAYERS = 2
    RF_N_ESTIMATORS = 300
    RF_MAX_DEPTH = None
    RF_N_JOBS = -1

    SEEDS = [42, 67, 80, 89]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(REPORT_DIR, exist_ok=True)

    print(f"Dataset: {DATASET_PATH}")
    print(f"Seeds: {SEEDS}")
    df = pd.read_csv(DATASET_PATH)
    vocab = ProcessVocab()
    vocab.build_vocab(df)

    train_df, val_df = split_dataframe(
        df,
        train_ratio=TRAIN_SPLIT_RATIO,
        strategy=SPLIT_STRATEGY,
        case_col="CaseID",
        time_col="Timestamp",
    )

    train_dataset = ProcessPrefixDataset(
        train_df,
        vocab,
        max_seq_len=MAX_SEQ_LENGTH,
        max_prefixes_per_case=MAX_PREFIXES,
        fit_normalization=True,
    )
    norm_stats = train_dataset.get_normalization_stats()
    val_dataset = ProcessPrefixDataset(
        val_df,
        vocab,
        max_seq_len=MAX_SEQ_LENGTH,
        max_prefixes_per_case=MAX_PREFIXES,
        normalization_stats=norm_stats,
        fit_normalization=False,
    )

    train_variant_freq = np.asarray(
        [float(x.get("variant_freq", 1.0)) for x in train_dataset.prefixes],
        dtype=np.float32,
    )
    tail_q1 = float(np.quantile(train_variant_freq, 0.33)) if len(train_variant_freq) > 0 else 1.0
    tail_q2 = float(np.quantile(train_variant_freq, 0.66)) if len(train_variant_freq) > 0 else tail_q1

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_configs = [
        {
            "name": "DNCICA",
            "kind": "dl",
            "build": lambda: DNCICABaseline(
                vocab_size_act=len(vocab.act2id),
                vocab_size_res=len(vocab.res2id),
                max_seq_len=MAX_SEQ_LENGTH,
                d_model=D_MODEL,
                lstm_layers=LSTM_NUM_LAYERS,
                dnc_layers=DNC_LAYERS,
            ),
        },
        {
            "name": "VanillaTransformer",
            "kind": "dl",
            "build": lambda: VanillaTransformerBaseline(
                vocab_size_act=len(vocab.act2id),
                vocab_size_res=len(vocab.res2id),
                d_model=D_MODEL,
                num_heads=NUM_HEADS,
                num_layers=NUM_LAYERS,
            ),
        },
        {
            "name": "LSTM",
            "kind": "dl",
            "build": lambda: LSTMBaseline(
                vocab_size_act=len(vocab.act2id),
                vocab_size_res=len(vocab.res2id),
                d_model=D_MODEL,
                num_layers=LSTM_NUM_LAYERS,
            ),
        },
        {
            "name": "RandomForest",
            "kind": "rf",
        },
    ]

    all_results = []
    all_bucket_rows = []
    all_gate_rows = []
    total_runs = len(SEEDS) * len(model_configs)
    run_idx = 0

    for seed in SEEDS:
        set_seed(seed)
        print(f"\n{'=' * 60}")
        print(f"  Seed = {seed}")
        print(f"{'=' * 60}")

        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
        )

        for cfg in model_configs:
            run_idx += 1
            name = cfg["name"]
            print(f"\n[{run_idx}/{total_runs}] {name} | seed={seed}")

            if cfg["kind"] == "rf":
                metrics = train_random_forest(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    tail_q1=tail_q1,
                    tail_q2=tail_q2,
                    n_estimators=RF_N_ESTIMATORS,
                    max_depth=RF_MAX_DEPTH,
                    n_jobs=RF_N_JOBS,
                    seed=seed,
                )
            else:
                model = cfg["build"]()
                if name == "DNCICA":
                    metrics = train_dnc_ica_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=EPOCHS,
                        lr=LEARNING_RATE,
                        device=DEVICE,
                        tail_q1=tail_q1,
                        tail_q2=tail_q2,
                        gate_lambda=FEATURE_GATE_LAMBDA,
                    )
                    append_feature_gate_rows(all_gate_rows, seed, name, model)
                else:
                    metrics = train_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=EPOCHS,
                        lr=LEARNING_RATE,
                        device=DEVICE,
                        tail_q1=tail_q1,
                        tail_q2=tail_q2,
                    )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_results.append(
                {
                    "seed": seed,
                    "model": name,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "tail_mae": metrics["tail_mae"],
                    "score": metrics["score"],
                }
            )
            append_bucket_rows(all_bucket_rows, seed, name, metrics)
            print(
                f"     [*] {name} seed={seed} done | "
                f"MAE={metrics['mae']:.4f} | TailMAE={metrics['tail_mae']:.4f} | "
                f"RMSE={metrics['rmse']:.4f} | Score={metrics['score']:.4f}"
            )

    results_df = pd.DataFrame(all_results)
    summary_df = build_summary(results_df)

    print("\n" + "=" * 90)
    print(f"{'DNCICA Baseline Results (mean +/- std over {} seeds)'.format(len(SEEDS)):^90}")
    print("=" * 90)
    print(f"{'Model':<22} | {'MAE':<20} | {'Tail MAE':<20} | {'RMSE':<20} | {'Score':<20}")
    print("-" * 90)
    for _, row in summary_df.iterrows():
        print(
            f"{row['model']:<22} | "
            f"{row['mae_mean']:.4f} +/- {row['mae_std']:.4f}   | "
            f"{row['tail_mae_mean']:.4f} +/- {row['tail_mae_std']:.4f}   | "
            f"{row['rmse_mean']:.4f} +/- {row['rmse_std']:.4f}   | "
            f"{row['score_mean']:.4f} +/- {row['score_std']:.4f}"
        )
    print("=" * 90)

    per_seed_path = os.path.join(REPORT_DIR, "baseline_dnc_ica_per_seed.csv")
    results_df.to_csv(per_seed_path, index=False, encoding="utf-8-sig")
    print(f"Saved per-seed results: {per_seed_path}")

    summary_path = os.path.join(REPORT_DIR, "baseline_dnc_ica_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved summary: {summary_path}")

    bucket_path = os.path.join(REPORT_DIR, "baseline_dnc_ica_bucket_metrics.csv")
    pd.DataFrame(all_bucket_rows).to_csv(bucket_path, index=False, encoding="utf-8-sig")
    print(f"Saved bucket metrics: {bucket_path}")

    gate_path = os.path.join(REPORT_DIR, "baseline_dnc_ica_feature_gates.csv")
    pd.DataFrame(all_gate_rows).to_csv(gate_path, index=False, encoding="utf-8-sig")
    print(f"Saved feature gates: {gate_path}")
