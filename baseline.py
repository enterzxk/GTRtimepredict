# -*- coding: utf-8 -*-
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader

from model import VanillaTransformerBaseline
from prefix import ProcessPrefixDataset, ProcessVocab
from training_utils import (
    batch_tail_mask,
    evaluate_bucket_regression,
    split_dataframe,
)


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        vocab_size_act,
        vocab_size_res,
        d_model=128,
        num_layers=2,
        dropout=0.1,
        time_dim=2,
    ):
        super().__init__()
        self.act_embedding = nn.Embedding(vocab_size_act, d_model, padding_idx=0)
        self.res_embedding = nn.Embedding(vocab_size_res, d_model, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=d_model * 2 + time_dim,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        head_hidden = max(d_model // 2, 16)
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, act_seq, res_seq, time_features, mask):
        act_emb = self.act_embedding(act_seq)
        res_emb = self.res_embedding(res_seq)
        x = torch.cat([act_emb, res_emb, time_features], dim=-1)

        output, _ = self.lstm(x)
        lengths = mask.long().sum(dim=1).clamp(min=1)
        last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.size(-1))
        last_hidden = output.gather(dim=1, index=last_idx).squeeze(1)
        return self.regressor(last_hidden).squeeze(-1)


def evaluate_model(model, data_loader, device, tail_q1, tail_q2):
    model.eval()
    abs_errors = []
    sq_errors = []
    tail_errors = []
    y_true_all = []
    y_pred_all = []
    variant_freq_all = []

    with torch.no_grad():
        for batch in data_loader:
            act = batch["act_seq"].to(device)
            res = batch["res_seq"].to(device)
            time_feat = batch["time_seq"].to(device)
            mask = batch["mask"].to(device)
            y_true = batch["target_rem_time"].to(device)
            variant_freq = batch["variant_freq"].to(device)

            y_pred = model(act, res, time_feat, mask)
            err = torch.abs(y_pred - y_true)
            abs_errors.append(err)
            sq_errors.append((y_pred - y_true) ** 2)
            y_true_all.append(y_true.detach().cpu())
            y_pred_all.append(y_pred.detach().cpu())
            variant_freq_all.append(variant_freq.detach().cpu())

            tail_m = batch_tail_mask(variant_freq, tail_q1)
            if tail_m.any():
                tail_errors.append(err[tail_m])

    abs_errors = torch.cat(abs_errors)
    sq_errors = torch.cat(sq_errors)

    mae = abs_errors.mean().item()
    rmse = torch.sqrt(sq_errors.mean()).item()
    tail_mae = torch.cat(tail_errors).mean().item() if len(tail_errors) > 0 else mae

    bucket_report = evaluate_bucket_regression(
        y_true=torch.cat(y_true_all).numpy(),
        y_pred=torch.cat(y_pred_all).numpy(),
        variant_freq=torch.cat(variant_freq_all).numpy(),
        q1=tail_q1,
        q2=tail_q2,
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "tail_mae": tail_mae,
        "bucket_report": bucket_report,
        "score": 0.5 * mae + 0.5 * tail_mae,
    }


def train_model(model, train_loader, val_loader, epochs, lr, device, tail_q1, tail_q2):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.L1Loss()

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

            optimizer.zero_grad()
            y_pred = model(act, res, time_feat, mask)
            loss = criterion(y_pred, y_true)
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
            f"    Epoch {epoch + 1:02d}/{epochs} | TrainMAE={train_loss:.4f} | "
            f"ValMAE={val_metrics['mae']:.4f} | TailMAE={val_metrics['tail_mae']:.4f} | "
            f"RMSE={val_metrics['rmse']:.4f} | Score={val_metrics['score']:.4f} | Time={elapsed:.1f}s"
        )

        if val_metrics["score"] < best_score:
            best_score = val_metrics["score"]
            best_metrics = val_metrics

    return best_metrics


def dataset_to_tabular_arrays(dataset):
    features = []
    targets = []
    variant_freqs = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        act_seq = item["act_seq"].detach().cpu().numpy().astype(np.float32)
        res_seq = item["res_seq"].detach().cpu().numpy().astype(np.float32)
        time_seq = item["time_seq"].detach().cpu().numpy().astype(np.float32).reshape(-1)
        mask = item["mask"].detach().cpu().numpy().astype(np.float32)
        variant_freq = float(item["variant_freq"].item())
        seq_len = float(mask.sum())

        features.append(
            np.concatenate(
                [act_seq, res_seq, time_seq, mask,
                 np.asarray([seq_len, variant_freq], dtype=np.float32)]
            )
        )
        targets.append(float(item["target_rem_time"].item()))
        variant_freqs.append(variant_freq)

    if not features:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.vstack(features).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(variant_freqs, dtype=np.float32),
    )


def evaluate_tabular(y_true, y_pred, variant_freq, tail_q1, tail_q2):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    variant_freq = np.asarray(variant_freq, dtype=np.float64)

    abs_errors = np.abs(y_pred - y_true)
    sq_errors = (y_pred - y_true) ** 2
    mae = float(abs_errors.mean())
    rmse = float(np.sqrt(sq_errors.mean()))

    tail_mask = variant_freq <= tail_q1
    tail_mae = float(abs_errors[tail_mask].mean()) if tail_mask.any() else mae
    bucket_report = evaluate_bucket_regression(
        y_true=y_true,
        y_pred=y_pred,
        variant_freq=variant_freq,
        q1=tail_q1,
        q2=tail_q2,
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "tail_mae": tail_mae,
        "bucket_report": bucket_report,
        "score": 0.5 * mae + 0.5 * tail_mae,
    }


def train_random_forest(train_dataset, val_dataset, tail_q1, tail_q2,
                        n_estimators, max_depth, n_jobs, seed):
    start = time.time()
    x_train, y_train, _ = dataset_to_tabular_arrays(train_dataset)
    x_val, y_val, val_variant_freq = dataset_to_tabular_arrays(val_dataset)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    metrics = evaluate_tabular(
        y_true=y_val,
        y_pred=y_pred,
        variant_freq=val_variant_freq,
        tail_q1=tail_q1,
        tail_q2=tail_q2,
    )

    elapsed = time.time() - start
    print(
        f"    RF | Trees={n_estimators} | ValMAE={metrics['mae']:.4f} | "
        f"TailMAE={metrics['tail_mae']:.4f} | RMSE={metrics['rmse']:.4f} | "
        f"Score={metrics['score']:.4f} | Time={elapsed:.1f}s"
    )
    return metrics


if __name__ == "__main__":
    # -------------------- Config --------------------
    DATASET_PATH = "dataset/processed_Sepsis.csv"
    REPORT_DIR = "results"
    TRAIN_SPLIT_RATIO = 0.8
    SPLIT_STRATEGY = "case"  # row | case
    MAX_SEQ_LENGTH = 50
    MAX_PREFIXES = 100

    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    EPOCHS = 30

    D_MODEL = 128
    NUM_HEADS = 8
    NUM_LAYERS = 4
    LSTM_NUM_LAYERS = 2
    RF_N_ESTIMATORS = 300
    RF_MAX_DEPTH = None
    RF_N_JOBS = -1

    SEEDS = [42, 67, 80, 89, 123]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(REPORT_DIR, exist_ok=True)

    # -------------------- Load & split data (fixed across seeds) --------------------
    print(f"加载数据: {DATASET_PATH}")
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

    # -------------------- Multi-seed experiments --------------------
    all_results = []  # list of dicts: {seed, model, mae, rmse, tail_mae, score}
    all_bucket_rows = []  # list of dicts for bucket CSV

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

            all_results.append({
                "seed": seed,
                "model": name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "tail_mae": metrics["tail_mae"],
                "score": metrics["score"],
            })

            print(
                f"     [*] {name} seed={seed} 完成 | "
                f"MAE={metrics['mae']:.4f} | TailMAE={metrics['tail_mae']:.4f} | "
                f"RMSE={metrics['rmse']:.4f} | Score={metrics['score']:.4f}"
            )

            report = metrics.get("bucket_report", {})
            q1 = report.get("quantiles", {}).get("q1", 0.0)
            q2 = report.get("quantiles", {}).get("q2", 0.0)
            for bucket in ["overall", "head", "torso", "tail"]:
                item = report.get(bucket, {"count": 0.0, "mae": 0.0, "rmse": 0.0})
                all_bucket_rows.append({
                    "seed": seed,
                    "model": name,
                    "bucket": bucket,
                    "count": item.get("count", 0.0),
                    "mae": item.get("mae", 0.0),
                    "rmse": item.get("rmse", 0.0),
                    "q1": q1,
                    "q2": q2,
                })

            # Free GPU memory between DL runs
            if cfg["kind"] == "dl":
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # -------------------- Aggregate: mean ± std --------------------
    results_df = pd.DataFrame(all_results)

    summary_rows = []
    for name in results_df["model"].unique():
        sub = results_df[results_df["model"] == name]
        summary_rows.append({
            "model": name,
            "mae_mean": sub["mae"].mean(),
            "mae_std": sub["mae"].std(),
            "rmse_mean": sub["rmse"].mean(),
            "rmse_std": sub["rmse"].std(),
            "tail_mae_mean": sub["tail_mae"].mean(),
            "tail_mae_std": sub["tail_mae"].std(),
            "score_mean": sub["score"].mean(),
            "score_std": sub["score"].std(),
        })
    summary_df = pd.DataFrame(summary_rows)

    # -------------------- Print results --------------------
    print("\n" + "=" * 90)
    print(f"{'Baseline Results (mean ± std over {} seeds)'.format(len(SEEDS)):^90}")
    print("=" * 90)
    print(
        f"{'Model':<22} | {'MAE':<20} | {'Tail MAE':<20} | {'RMSE':<20} | {'Score':<20}"
    )
    print("-" * 90)
    for _, row in summary_df.iterrows():
        print(
            f"{row['model']:<22} | "
            f"{row['mae_mean']:.4f} ± {row['mae_std']:.4f}   | "
            f"{row['tail_mae_mean']:.4f} ± {row['tail_mae_std']:.4f}   | "
            f"{row['rmse_mean']:.4f} ± {row['rmse_std']:.4f}   | "
            f"{row['score_mean']:.4f} ± {row['score_std']:.4f}"
        )
    print("=" * 90)

    # Per-seed detail table
    print(f"\n{'Per-Seed Detail':^90}")
    print("-" * 90)
    print(f"{'Model':<22} | {'Seed':<8} | {'MAE':<12} | {'Tail MAE':<12} | {'RMSE':<12} | {'Score':<12}")
    print("-" * 90)
    for _, row in results_df.iterrows():
        print(
            f"{row['model']:<22} | {row['seed']:<8} | {row['mae']:<12.4f} | "
            f"{row['tail_mae']:<12.4f} | {row['rmse']:<12.4f} | {row['score']:<12.4f}"
        )
    print("=" * 90)

    # -------------------- Save CSVs --------------------
    per_seed_path = os.path.join(REPORT_DIR, "baseline_per_seed.csv")
    results_df.to_csv(per_seed_path, index=False, encoding="utf-8-sig")
    print(f"每轮结果已导出: {per_seed_path}")

    summary_path = os.path.join(REPORT_DIR, "baseline_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"汇总结果已导出: {summary_path}")

    bucket_path = os.path.join(REPORT_DIR, "baseline_bucket_metrics.csv")
    bucket_df = pd.DataFrame(all_bucket_rows)
    bucket_df.to_csv(bucket_path, index=False, encoding="utf-8-sig")
    print(f"分桶评估已导出: {bucket_path}")
