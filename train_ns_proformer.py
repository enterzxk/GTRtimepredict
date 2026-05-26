import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from ns_proformer import (
    MotifBPETokenizer,
    NSProFormer,
    NSProFormerPrefixDataset,
    ProcessStructurePrior,
    split_by_case,
)


@dataclass
class NSProFormerTrainConfig:
    dataset_path: str = "dataset/processed_BPIC2015_1.csv"
    save_dir: str = "checkpoints"
    save_name: str = "best_ns_proformer.pth"

    train_ratio: float = 0.8
    random_seed: int = 42
    split_strategy: str = "row"  # options: "row", "case"
    use_resource_in_token: bool = True
    resource_token_delimiter: str = "||"

    max_raw_prefix_len: int = 80
    max_token_len: int = 128
    max_prefixes_per_case: int = 100

    motif_vocab_size: int = 4000
    min_pair_count: int = 20

    batch_size: int = 128
    epochs: int = 30
    lr: float = 3e-4

    d_model: int = 64
    num_layers: int = 3
    num_heads: int = 4
    d_ff: int = 1024
    num_mixtures: int = 5
    dropout: float = 0.1

    target_transform: str = "log1p"  # options: "none", "log1p"
    early_stop_patience: int = 8
    lr_scheduler_patience: int = 3
    min_lr: float = 1e-6

    num_workers: int = 0
    deterministic: bool = True

    aux_point_loss_weight: float = 0.2
    eval_point_estimator: str = "median_sample"  # options: "mean", "median_sample"
    eval_sample_count: int = 200

    reachability_mode: str = "direct"  # options: "direct", "k_hop", "transitive"
    reachability_hops: int = 1


def resolve_project_path(path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return path if os.path.isabs(path) else os.path.join(script_dir, path)


def compute_metrics(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0.0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def summarize_prefix_baselines(train_dataset, val_dataset) -> None:
    y_train = np.asarray([float(s["target_rem_time"]) for s in train_dataset.samples], dtype=np.float32)
    y_val = np.asarray([float(s["target_rem_time"]) for s in val_dataset.samples], dtype=np.float32)

    mean_pred = float(y_train.mean()) if len(y_train) > 0 else 0.0
    median_pred = float(np.median(y_train)) if len(y_train) > 0 else 0.0

    mae_zero = float(np.mean(np.abs(y_val - 0.0))) if len(y_val) > 0 else float("nan")
    mae_mean = float(np.mean(np.abs(y_val - mean_pred))) if len(y_val) > 0 else float("nan")
    mae_median = float(np.mean(np.abs(y_val - median_pred))) if len(y_val) > 0 else float("nan")

    print("Prefix-level naive MAE baselines:")
    print(f"  Predict 0      : {mae_zero:.4f}")
    print(f"  Predict mean   : {mae_mean:.4f} (mean={mean_pred:.4f})")
    print(f"  Predict median : {mae_median:.4f} (median={median_pred:.4f})")


def summarize_reachability_density(dataset, sample_size: int = 2000) -> None:
    n = min(sample_size, len(dataset.samples))
    if n == 0:
        print("Reachability mask density: N/A (empty dataset)")
        return

    densities = []
    for i in range(n):
        m = dataset.samples[i]["reachability_mask"]
        if m.size > 0:
            densities.append(float(m.mean()))

    if not densities:
        print("Reachability mask density: N/A (no valid masks)")
        return

    arr = np.asarray(densities, dtype=np.float32)
    print(
        "Reachability mask density "
        f"(mean/p50/p90 over {len(arr)} samples): "
        f"{arr.mean():.4f}/{np.quantile(arr, 0.5):.4f}/{np.quantile(arr, 0.9):.4f}"
    )


def transform_target(y: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return y
    if mode == "log1p":
        return torch.log1p(torch.clamp(y, min=0.0))
    raise ValueError(f"Unsupported target transform mode: {mode}")


def mdn_point_prediction(out: dict, target_transform: str) -> torch.Tensor:
    pi = out["pi"]
    mu = out["mu"]
    sigma = out["sigma"]

    if target_transform == "log1p":
        # If log(1+y) ~ N(mu, sigma^2), then E[y] = exp(mu + 0.5*sigma^2) - 1.
        component_means = torch.expm1(mu + 0.5 * sigma.pow(2))
        pred = torch.sum(pi * component_means, dim=-1)
        return torch.clamp(pred, min=0.0)

    pred = torch.sum(pi * mu, dim=-1)
    return torch.clamp(pred, min=0.0)


def mdn_point_in_target_space(out: dict) -> torch.Tensor:
    return torch.sum(out["pi"] * out["mu"], dim=-1)


def mdn_eval_prediction(
    out: dict,
    target_transform: str,
    estimator: str = "median_sample",
    sample_count: int = 200,
) -> torch.Tensor:
    if target_transform == "none":
        pred = torch.sum(out["pi"] * out["mu"], dim=-1)
        return torch.clamp(pred, min=0.0)

    if estimator == "mean":
        return mdn_point_prediction(out, target_transform)

    # MAE-oriented estimator: median in original space.
    if estimator == "median_sample":
        samples = []
        for _ in range(sample_count):
            comp_idx = torch.multinomial(out["pi"], num_samples=1, replacement=True).squeeze(-1)
            mu_sel = out["mu"].gather(1, comp_idx.unsqueeze(-1)).squeeze(-1)
            sigma_sel = out["sigma"].gather(1, comp_idx.unsqueeze(-1)).squeeze(-1)
            eps = torch.randn_like(mu_sel)
            z = mu_sel + sigma_sel * eps
            y = torch.expm1(z)
            samples.append(torch.clamp(y, min=0.0))
        stacked = torch.stack(samples, dim=1)
        return torch.median(stacked, dim=1).values

    raise ValueError(f"Unsupported eval point estimator: {estimator}")


def train_ns_proformer(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    save_path,
    target_transform_mode="log1p",
    early_stop_patience=8,
    lr_scheduler_patience=3,
    min_lr=1e-6,
    aux_point_loss_weight=0.2,
    eval_point_estimator="median_sample",
    eval_sample_count=200,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=lr_scheduler_patience,
        min_lr=min_lr,
    )

    best_mae = float("inf")
    best_rmse = float("inf")
    epochs_without_improve = 0

    print(f"Start training NSProFormer (device: {device})")

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        train_nll = 0.0

        for batch in train_loader:
            token_ids = batch["token_ids"].to(device)
            time_agg = batch["time_agg"].to(device)
            reachability_mask = batch["reachability_mask"].to(device)
            marking = batch["marking"].to(device)
            valid_mask = batch["mask"].to(device)
            y_true = batch["target_rem_time"].to(device)
            y_target = transform_target(y_true, target_transform_mode)

            optimizer.zero_grad()

            out = model(
                token_ids=token_ids,
                time_agg=time_agg,
                reachability_mask=reachability_mask,
                marking=marking,
                padding_mask=valid_mask,
                target=y_target,
            )

            point_target_pred = mdn_point_in_target_space(out)
            point_loss = F.l1_loss(point_target_pred, y_target)
            loss = out["nll"] + aux_point_loss_weight * point_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_nll += loss.item() * token_ids.size(0)

        train_nll /= len(train_loader.dataset)

        model.eval()
        preds, trues = [], []
        val_nll = 0.0

        with torch.no_grad():
            for batch in val_loader:
                token_ids = batch["token_ids"].to(device)
                time_agg = batch["time_agg"].to(device)
                reachability_mask = batch["reachability_mask"].to(device)
                marking = batch["marking"].to(device)
                valid_mask = batch["mask"].to(device)
                y_true = batch["target_rem_time"].to(device)
                y_target = transform_target(y_true, target_transform_mode)

                out = model(
                    token_ids=token_ids,
                    time_agg=time_agg,
                    reachability_mask=reachability_mask,
                    marking=marking,
                    padding_mask=valid_mask,
                    target=y_target,
                )

                val_nll += out["nll"].item() * token_ids.size(0)
                y_pred = mdn_eval_prediction(
                    out,
                    target_transform=target_transform_mode,
                    estimator=eval_point_estimator,
                    sample_count=eval_sample_count,
                )

                preds.extend(y_pred.cpu().numpy())
                trues.extend(y_true.cpu().numpy())

        val_nll /= len(val_loader.dataset)
        mae, rmse = compute_metrics(np.asarray(trues), np.asarray(preds))
        scheduler.step(mae)

        if mae < best_mae:
            best_mae = mae
            best_rmse = rmse
            torch.save(model.state_dict(), save_path)
            print(f"   New best model saved to: {save_path}")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | Train NLL: {train_nll:.4f} | "
            f"Val NLL: {val_nll:.4f} | Val MAE: {mae:.4f} | Val RMSE: {rmse:.4f} | "
            f"LR: {current_lr:.6g} | Time: {epoch_time:.1f}s"
        )

        if epochs_without_improve >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    print(f"Training complete | Best MAE: {best_mae:.4f} | Best RMSE: {best_rmse:.4f}")


def build_training_sequences(df):
    seqs = []
    working_df = df.copy()
    if "Timestamp" in working_df.columns:
        working_df = working_df.sort_values(["CaseID", "Timestamp"])

    for _, group in working_df.groupby("CaseID", sort=False):
        seqs.append(group["Activity"].astype(str).tolist())
    return seqs


def build_training_sequences_by_col(df, activity_col: str):
    seqs = []
    working_df = df.copy()
    if "Timestamp" in working_df.columns:
        working_df = working_df.sort_values(["CaseID", "Timestamp"])

    for _, group in working_df.groupby("CaseID", sort=False):
        seqs.append(group[activity_col].astype(str).tolist())
    return seqs


def prepare_event_token_column(df: pd.DataFrame, config: NSProFormerTrainConfig):
    working_df = df.copy()

    if config.use_resource_in_token:
        if "Resource" not in working_df.columns:
            raise ValueError("use_resource_in_token=True requires 'Resource' column in dataset")
        working_df["EventToken"] = (
            working_df["Activity"].astype(str)
            + config.resource_token_delimiter
            + working_df["Resource"].astype(str)
        )
        return working_df, "EventToken"

    return working_df, "Activity"


def split_dataframe(df: pd.DataFrame, config: NSProFormerTrainConfig):
    strategy = config.split_strategy.lower().strip()
    if strategy == "case":
        return split_by_case(df, train_ratio=config.train_ratio, seed=config.random_seed)

    if strategy == "row":
        working_df = df.copy()
        if "CaseID" in working_df.columns and "Timestamp" in working_df.columns:
            working_df = working_df.sort_values(["CaseID", "Timestamp"]).reset_index(drop=True)
        else:
            working_df = working_df.reset_index(drop=True)

        split_idx = int(len(working_df) * config.train_ratio)
        train_df = working_df.iloc[:split_idx].copy()
        val_df = working_df.iloc[split_idx:].copy()
        return train_df, val_df

    raise ValueError("split_strategy must be one of ['row', 'case']")


def run_from_pycharm(config: NSProFormerTrainConfig) -> None:
    dataset_path = resolve_project_path(config.dataset_path)
    save_dir = resolve_project_path(config.save_dir)
    save_path = os.path.join(save_dir, config.save_name)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir, exist_ok=True)
    set_global_seed(config.random_seed, deterministic=config.deterministic)

    print(f"Dataset path: {dataset_path}")
    print(f"Checkpoint path: {save_path}")
    print(f"Config: {config}")
    if config.split_strategy.lower().strip() == "row":
        print("Split strategy: row (same protocol as baseline.py; easier, may include case leakage).")
    else:
        print("Split strategy: case (strict protocol, no case leakage).")

    df = pd.read_csv(dataset_path)
    df, activity_col = prepare_event_token_column(df, config)
    train_df, val_df = split_dataframe(df, config)

    train_sequences = build_training_sequences_by_col(train_df, activity_col)

    tokenizer = MotifBPETokenizer(
        max_vocab_size=config.motif_vocab_size,
        min_pair_count=config.min_pair_count,
    )
    tokenizer.fit(train_sequences)

    prior = ProcessStructurePrior(
        tokenizer,
        train_sequences,
        reachability_mode=config.reachability_mode,
        reachability_hops=config.reachability_hops,
    )

    train_dataset = NSProFormerPrefixDataset(
        train_df,
        tokenizer,
        prior,
        max_seq_len=config.max_raw_prefix_len,
        max_token_len=config.max_token_len,
        max_prefixes_per_case=config.max_prefixes_per_case,
        activity_col=activity_col,
        fit_normalization=True,
    )
    ns_norm_stats = train_dataset.get_normalization_stats()

    val_dataset = NSProFormerPrefixDataset(
        val_df,
        tokenizer,
        prior,
        max_seq_len=config.max_raw_prefix_len,
        max_token_len=config.max_token_len,
        max_prefixes_per_case=config.max_prefixes_per_case,
        activity_col=activity_col,
        normalization_stats=ns_norm_stats,
        fit_normalization=False,
    )

    print(f"Train prefixes: {len(train_dataset)} | Val prefixes: {len(val_dataset)}")
    print(f"Time aggregate scales: {np.round(ns_norm_stats['time_agg_scale'], 4)}")
    summarize_prefix_baselines(train_dataset, val_dataset)
    summarize_reachability_density(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = NSProFormer(
        vocab_size=tokenizer.vocab_size,
        marking_dim=prior.marking_dim,
        g_dim=5,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        num_mixtures=config.num_mixtures,
        max_len=config.max_token_len,
        inject_marking_each_layer=True,
    )

    train_ns_proformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        lr=config.lr,
        device=device,
        save_path=save_path,
        target_transform_mode=config.target_transform,
        early_stop_patience=config.early_stop_patience,
        lr_scheduler_patience=config.lr_scheduler_patience,
        min_lr=config.min_lr,
        aux_point_loss_weight=config.aux_point_loss_weight,
        eval_point_estimator=config.eval_point_estimator,
        eval_sample_count=config.eval_sample_count,
    )


if __name__ == "__main__":
    # Edit this config in PyCharm, then click Run.
    config = NSProFormerTrainConfig(
        dataset_path="dataset/processed_BPIC2015_1.csv",
        save_dir="checkpoints",
        save_name="best_ns_proformer.pth",
    )
    run_from_pycharm(config)
