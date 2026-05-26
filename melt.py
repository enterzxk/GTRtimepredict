import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time
import os

from prefix import ProcessVocab, ProcessPrefixDataset
from stg_transformer import SingleTaskSTGTransformer
from model import LSTMBaseline, GRUBaseline, VanillaTransformerBaseline


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_type, vocab, device):
    if model_type == "lstm":
        model = LSTMBaseline(
            vocab_size_act=len(vocab.act2id),
            vocab_size_res=len(vocab.res2id),
            d_model=64,
            hidden_size=128
        )
    elif model_type == "gru":
        model = GRUBaseline(
            vocab_size_act=len(vocab.act2id),
            vocab_size_res=len(vocab.res2id),
            d_model=64,
            hidden_size=128
        )
    elif model_type == "vanilla":
        model = VanillaTransformerBaseline(
            vocab_size_act=len(vocab.act2id),
            vocab_size_res=len(vocab.res2id),
            d_model=64,
            num_heads=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1
        )
    elif model_type == "full":
        model = SingleTaskSTGTransformer(
            num_activities=len(vocab.act2id),
            num_resources=len(vocab.res2id),
            d_model=128,
            num_heads=4,
            num_layers=3,
            model_type="full"
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    return model.to(device)


def forward_batch(model_type, model, batch, vocab, device):
    act_seq = batch['act_seq'].to(device)
    res_seq = batch['res_seq'].to(device)
    time_features = batch['time_seq'].to(device)
    target_rt = batch['target_rem_time'].to(device)

    if model_type == "full":
        time_matrix = batch['time_matrix'].to(device)
        graph_matrix = batch['graph_matrix'].to(device) if 'graph_matrix' in batch else None

        # prefix.py 里 mask 定义是 1=有效, 0=padding
        padding_mask = batch['mask'].to(device).long()

        pred_rt = model(
            act_seq,
            res_seq,
            time_features,
            time_matrix=time_matrix,
            graph_matrix=graph_matrix,
            padding_mask=padding_mask
        )
    else:
        mask = batch['mask'].to(device)
        pred_rt = model(act_seq, res_seq, time_features, mask)

    return pred_rt, target_rt


def evaluate(model_type, model, val_loader, vocab, device):
    model.eval()
    val_errors = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            pred_rt, target_rt = forward_batch(model_type, model, batch, vocab, device)
            val_errors.append(torch.abs(pred_rt - target_rt))
            val_targets.append(target_rt)

    all_errors = torch.cat(val_errors)
    all_targets = torch.cat(val_targets)

    mae = all_errors.mean().item()
    rmse = torch.sqrt((all_errors ** 2).mean()).item()
    mape = (all_errors / torch.clamp(all_targets, min=1e-4)).mean().item() * 100

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def train_and_eval_model(model_name, model_type, train_loader, val_loader, vocab, device, epochs=30):
    set_seed(42)

    print(f"\n---> 开始训练模型: {model_name} (模式: {model_type})")

    model = build_model(model_type, vocab, device)

    criterion_train = nn.L1Loss()
    lr = 5e-4 if model_type == "full" else 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_mae = float('inf')
    best_metrics = {}

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()

        for batch in train_loader:
            optimizer.zero_grad()
            pred_rt, target_rt = forward_batch(model_type, model, batch, vocab, device)
            loss = criterion_train(pred_rt, target_rt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        metrics = evaluate(model_type, model, val_loader, vocab, device)

        if metrics['MAE'] < best_mae:
            best_mae = metrics['MAE']
            best_metrics = metrics

        epoch_time = time.time() - epoch_start_time
        print(
            f"     Epoch [{epoch + 1}/{epochs}] | "
            f"耗时: {epoch_time:.1f}s | "
            f"Val MAE: {metrics['MAE']:.4f} | "
            f"Val RMSE: {metrics['RMSE']:.4f} | "
            f"Val MAPE: {metrics['MAPE']:.2f}%"
        )

    print(f"     [*] {model_name} 训练完成！最佳 Val MAE: {best_metrics['MAE']:.4f}")
    return best_metrics


def main():
    FILE_PATH = "dataset/processed_BPIC2015_1.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(FILE_PATH):
        print(f"[Error] 数据集缺失: {FILE_PATH}。请确保数据路径正确。")
        return

    df = pd.read_csv(FILE_PATH)
    vocab = ProcessVocab()
    vocab.build_vocab(df)

    case_start_times = df.groupby('CaseID')['Timestamp'].min().sort_values()
    sorted_case_ids = case_start_times.index.tolist()
    train_cases = int(len(sorted_case_ids) * 0.8)

    train_case_ids = sorted_case_ids[:train_cases]
    val_case_ids = sorted_case_ids[train_cases:]

    df_train = df[df['CaseID'].isin(train_case_ids)].copy()
    df_val = df[df['CaseID'].isin(val_case_ids)].copy()

    train_dataset = ProcessPrefixDataset(df_train, vocab, max_seq_len=20)
    val_dataset = ProcessPrefixDataset(df_val, vocab, max_seq_len=20)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    models_to_test = {
        'LSTM Baseline': 'lstm',
        'GRU Baseline': 'gru',
        'Vanilla Transformer': 'vanilla',
        'STG Full': 'full'
    }

    results = {}

    print("\n开始执行基准对比实验 (Baseline Comparison) ...")
    for name, m_type in models_to_test.items():
        metrics = train_and_eval_model(name, m_type, train_loader, val_loader, vocab, device)
        results[name] = metrics

    print("\n" + "=" * 80)
    print(f"{'基准对比实验结果总结 (Baseline Comparison)':^80}")
    print("=" * 80)
    print(f"{'Model Name':<30} | {'MAE (Hours)':<12} | {'RMSE (Hours)':<12} | {'MAPE (%)':<12}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<30} | {m['MAE']:<12.4f} | {m['RMSE']:<12.4f} | {m['MAPE']:<12.4f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()