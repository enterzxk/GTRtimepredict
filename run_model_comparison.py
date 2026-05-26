import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prefix import ProcessVocab, ProcessPrefixDataset
from model import (
    LSTMBaseline,
    GRUBaseline,
    VanillaTransformerBaseline,
)
from stg_transformer import SingleTaskSTGTransformer


# ==========================================
# Shared experiment settings (keep consistent)
# ==========================================
DATASET_PATH = "dataset/processed_BPIC2015_2.csv"
RESULTS_PATH = "results/model_comparison_results.csv"
TRAIN_SPLIT_RATIO = 0.8

MAX_SEQ_LENGTH = 20
MAX_PREFIXES = 100
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

D_MODEL = 128
HIDDEN_SIZE = 128
NUM_HEADS = 4
NUM_LAYERS = 3


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def build_datasets():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    vocab = ProcessVocab()
    vocab.build_vocab(df)

    split_idx = int(len(df) * TRAIN_SPLIT_RATIO)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_dataset = ProcessPrefixDataset(
        train_df,
        vocab,
        max_seq_len=MAX_SEQ_LENGTH,
        max_prefixes_per_case=MAX_PREFIXES,
    )
    test_dataset = ProcessPrefixDataset(
        test_df,
        vocab,
        max_seq_len=MAX_SEQ_LENGTH,
        max_prefixes_per_case=MAX_PREFIXES,
    )

    return vocab, train_dataset, test_dataset


def make_loaders(train_dataset, test_dataset):
    # Rebuild loaders for each model with the same seed so shuffle behavior is aligned.
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def train_and_evaluate(model_name, model, train_dataset, test_dataset, device):
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.L1Loss()

    best_mae = float("inf")
    best_rmse = float("inf")
    best_epoch = 0

    print(f"\n=== Training {model_name} on {device} ===")

    for epoch in range(EPOCHS):
        train_loader, test_loader = make_loaders(train_dataset, test_dataset)

        start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            act_seq = batch["act_seq"].to(device)
            res_seq = batch["res_seq"].to(device)
            time_seq = batch["time_seq"].to(device)
            mask = batch["mask"].to(device)
            y_true = batch["target_rem_time"].to(device)

            optimizer.zero_grad()
            if model_name == "STG (Full)":
                time_matrix = batch["time_matrix"].to(device)
                y_pred = model(
                    act_seq,
                    res_seq,
                    time_seq,
                    time_matrix=time_matrix,
                    graph_matrix=None,
                    padding_mask=mask.long(),
                )
            else:
                y_pred = model(act_seq, res_seq, time_seq, mask)
            y_pred = torch.nan_to_num(y_pred, nan=0.0)

            loss = criterion(y_pred, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * act_seq.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                act_seq = batch["act_seq"].to(device)
                res_seq = batch["res_seq"].to(device)
                time_seq = batch["time_seq"].to(device)
                mask = batch["mask"].to(device)

                y_true = batch["target_rem_time"].cpu().numpy()
                if model_name == "STG (Full)":
                    time_matrix = batch["time_matrix"].to(device)
                    y_pred = model(
                        act_seq,
                        res_seq,
                        time_seq,
                        time_matrix=time_matrix,
                        graph_matrix=None,
                        padding_mask=mask.long(),
                    )
                else:
                    y_pred = model(act_seq, res_seq, time_seq, mask)
                y_pred = torch.nan_to_num(y_pred, nan=0.0).cpu().numpy()

                preds.extend(y_pred)
                trues.extend(y_true)

        mae, rmse = calculate_metrics(trues, preds)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train MAE: {train_loss:.4f} | Val MAE: {mae:.4f} | Val RMSE: {rmse:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if mae < best_mae:
            best_mae = mae
            best_rmse = rmse
            best_epoch = epoch + 1

    return {
        "Model": model_name,
        "Best_MAE": best_mae,
        "Best_RMSE": best_rmse,
        "Best_Epoch": best_epoch,
    }


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab, train_dataset, test_dataset = build_datasets()

    models = {
        "LSTM Baseline": LSTMBaseline(
            vocab_size_act=len(vocab.act2id),
            vocab_size_res=len(vocab.res2id),
            d_model=D_MODEL,
            hidden_size=HIDDEN_SIZE,
        ),
        "GRU Baseline": GRUBaseline(
            vocab_size_act=len(vocab.act2id),
            vocab_size_res=len(vocab.res2id),
            d_model=D_MODEL,
            hidden_size=HIDDEN_SIZE,
        ),
        "Vanilla Transformer Baseline": VanillaTransformerBaseline(
            vocab_size_act=len(vocab.act2id),
            vocab_size_res=len(vocab.res2id),
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dim_feedforward=D_MODEL * 4,
            dropout=0.1,
        ),
        "STG (Full)": SingleTaskSTGTransformer(
            num_activities=len(vocab.act2id),
            num_resources=len(vocab.res2id),
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            model_type="full",
        ),
    }

    all_results = []
    for model_name, model in models.items():
        set_seed(SEED)
        result = train_and_evaluate(model_name, model, train_dataset, test_dataset, device)
        all_results.append(result)

    result_df = pd.DataFrame(all_results).sort_values(by="Best_MAE", ascending=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    result_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(f"{'Model Comparison Results':^78}")
    print("=" * 78)
    print(f"{'Model':<34} | {'Best MAE':<12} | {'Best RMSE':<12} | {'Best Epoch':<10}")
    print("-" * 78)
    for row in result_df.itertuples(index=False):
        print(f"{row.Model:<34} | {row.Best_MAE:<12.4f} | {row.Best_RMSE:<12.4f} | {row.Best_Epoch:<10}")
    print("=" * 78)
    print(f"Saved result file: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
