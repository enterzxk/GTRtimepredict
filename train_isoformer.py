import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from STG_model import IsoFormerSTGTransformer
from prefix import ProcessPrefixDataset, ProcessVocab
from training_utils import (
    VariantBucketBatchSampler,
    batch_tail_mask,
    build_activity_pair_count_matrix,
    build_warmup_cosine_scheduler,
    evaluate_bucket_regression,
    hybrid_loss,
    save_bucket_report,
    split_dataframe,
)


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, data_loader, device, tail_q1, tail_q2):
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
            time_matrix = batch["time_matrix"].to(device)
            mask = batch["mask"].to(device)
            y_true = batch["target_rem_time"].to(device)
            variant_freq = batch["variant_freq"].to(device)

            mu = model(
                act_seq=act,
                res_seq=res,
                time_features=time_feat,
                time_matrix=time_matrix,
                padding_mask=mask,
                return_dist=False,
            )

            err = torch.abs(mu - y_true)
            abs_errors.append(err)
            sq_errors.append((mu - y_true) ** 2)
            y_true_all.append(y_true.detach().cpu())
            y_pred_all.append(mu.detach().cpu())
            variant_freq_all.append(variant_freq.detach().cpu())

            tail_m = batch_tail_mask(variant_freq, tail_q1)
            if tail_m.any():
                tail_errors.append(err[tail_m])

    abs_errors = torch.cat(abs_errors)
    sq_errors = torch.cat(sq_errors)

    mae = abs_errors.mean().item()
    rmse = torch.sqrt(sq_errors.mean()).item()

    if len(tail_errors) > 0:
        tail_mae = torch.cat(tail_errors).mean().item()
    else:
        tail_mae = mae

    bucket_report = evaluate_bucket_regression(
        y_true=torch.cat(y_true_all).numpy(),
        y_pred=torch.cat(y_pred_all).numpy(),
        variant_freq=torch.cat(variant_freq_all).numpy(),
        q1=tail_q1,
        q2=tail_q2,
    )

    early_stop_score = 0.5 * mae + 0.5 * tail_mae
    return {
        "mae": mae,
        "rmse": rmse,
        "tail_mae": tail_mae,
        "bucket_report": bucket_report,
        "early_stop_score": early_stop_score,
    }


def train_isoformer(
    model,
    train_loader,
    val_loader,
    train_sampler,
    stage_a_epochs,
    stage_b_epochs,
    lr,
    device,
    save_path,
    tail_q1,
    tail_q2,
    tail_boost,
    alpha,
    lambda_gamma_l1,
    bucket_report_path,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_epochs = stage_a_epochs + stage_b_epochs
    total_steps = total_epochs * len(train_loader)
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    best_score = float("inf")
    best_epoch = 0
    best_bucket_report = None

    global_step = 0
    print(f"开始训练 IsoFormerSTGTransformer (device={device})")

    for epoch in range(total_epochs):
        phase = "A" if epoch < stage_a_epochs else "B"
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_nll = 0.0
        sample_count = 0

        start_time = time.time()
        for batch in train_loader:
            act = batch["act_seq"].to(device)
            res = batch["res_seq"].to(device)
            time_feat = batch["time_seq"].to(device)
            time_matrix = batch["time_matrix"].to(device)
            mask = batch["mask"].to(device)
            y_true = batch["target_rem_time"].to(device)
            variant_freq = batch["variant_freq"].to(device)

            # Stage-B tail emphasis for high-variant long-tail samples.
            if phase == "B":
                tail_m = batch_tail_mask(variant_freq, tail_q1)
                adj_variant_freq = variant_freq.clone()
                adj_variant_freq[tail_m] = adj_variant_freq[tail_m] / float(tail_boost)
            else:
                adj_variant_freq = variant_freq

            optimizer.zero_grad()
            mu, sigma = model(
                act_seq=act,
                res_seq=res,
                time_features=time_feat,
                time_matrix=time_matrix,
                padding_mask=mask,
                return_dist=True,
            )

            loss, l1_part, nll_part = hybrid_loss(
                mu=mu,
                sigma=sigma,
                y_true=y_true,
                variant_freq=adj_variant_freq,
                alpha=alpha,
            )

            if lambda_gamma_l1 > 0.0:
                loss = loss + lambda_gamma_l1 * model.gamma_l1_penalty()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            bs = act.size(0)
            sample_count += bs
            epoch_loss += loss.item() * bs
            epoch_l1 += l1_part.item() * bs
            epoch_nll += nll_part.item() * bs
            global_step += 1

        epoch_loss /= max(sample_count, 1)
        epoch_l1 /= max(sample_count, 1)
        epoch_nll /= max(sample_count, 1)

        val_metrics = evaluate(model, val_loader, device=device, tail_q1=tail_q1, tail_q2=tail_q2)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1:02d}/{total_epochs} | Phase={phase} | "
            f"TrainLoss={epoch_loss:.4f} (L1={epoch_l1:.4f}, NLL={epoch_nll:.4f}) | "
            f"ValMAE={val_metrics['mae']:.4f} | ValTailMAE={val_metrics['tail_mae']:.4f} | "
            f"ValRMSE={val_metrics['rmse']:.4f} | Score={val_metrics['early_stop_score']:.4f} | "
            f"Time={elapsed:.1f}s"
        )

        if val_metrics["early_stop_score"] < best_score:
            best_score = val_metrics["early_stop_score"]
            best_epoch = epoch + 1
            best_bucket_report = val_metrics["bucket_report"]
            torch.save(model.state_dict(), save_path)
            print(f"  验证指标改进，已保存模型到 {save_path}")

    if best_bucket_report is not None:
        save_bucket_report(bucket_report_path, "IsoFormer_Best", best_bucket_report)
        print(f"已导出分桶评估报告: {bucket_report_path}")

    print(f"训练完成，最佳EarlyStopScore={best_score:.4f} (Epoch {best_epoch})")


if __name__ == "__main__":
    # -------------------- PyCharm direct-run config --------------------
    DATASET_PATH = "dataset/processed_BPIC2020.csv"
    SAVE_DIR = "checkpoints"
    MODEL_SAVE_PATH = f"{SAVE_DIR}/best_isoformer.pth"
    REPORT_DIR = "results"
    BUCKET_REPORT_PATH = f"{REPORT_DIR}/isoformer_bucket_metrics.csv"

    TRAIN_SPLIT_RATIO = 0.8
    SPLIT_STRATEGY = "case"  # row | case
    MAX_SEQ_LENGTH = 50
    MAX_PREFIXES = 100

    BATCH_SIZE = 128
    STAGE_A_EPOCHS = 40
    STAGE_B_EPOCHS = 10
    LEARNING_RATE = 3e-4
    SEED = 2026

    D_MODEL = 128
    NUM_HEADS = 8
    NUM_LAYERS = 4

    # Core strategy switches
    USE_DYNAMIC_FUSION = True
    USE_TIME_BIAS = True
    USE_TIME_VALUE_GATE = True
    # Optional low-rank activity bias (default off for high-variant logs)
    USE_ACT_LOWRANK_BIAS = False
    ACT_BIAS_RANK = 16
    PAIR_COUNT_THRESHOLD = 20
    ACT_GAMMA_INIT = -3.0
    LAMBDA_GAMMA_L1 = 2e-4 if USE_ACT_LOWRANK_BIAS else 0.0

    # Loss setup
    LOSS_ALPHA = 0.6
    TAIL_BOOST_STAGE_B = 2.0

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    set_seed(SEED)

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

    train_sampler = VariantBucketBatchSampler(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        ratio=(4, 3, 3),
        seed=SEED,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build pair frequency matrix for optional act low-rank bias threshold mask.
    pair_count_matrix = build_activity_pair_count_matrix(train_dataset, vocab_size=len(vocab.act2id))

    model = IsoFormerSTGTransformer(
        num_activities=len(vocab.act2id),
        num_resources=len(vocab.res2id),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        use_dynamic_fusion=USE_DYNAMIC_FUSION,
        use_time_bias=USE_TIME_BIAS,
        use_time_value_gate=USE_TIME_VALUE_GATE,
        use_act_lowrank_bias=USE_ACT_LOWRANK_BIAS,
        act_bias_rank=ACT_BIAS_RANK,
        pair_count_threshold=PAIR_COUNT_THRESHOLD,
        act_gamma_init=ACT_GAMMA_INIT,
        model_type="full",
    ).to(DEVICE)
    model.set_activity_pair_count_matrix(pair_count_matrix)

    train_variant_freq = np.asarray([float(x.get("variant_freq", 1.0)) for x in train_dataset.prefixes], dtype=np.float32)
    tail_q1 = float(np.quantile(train_variant_freq, 0.33)) if len(train_variant_freq) > 0 else 1.0
    tail_q2 = float(np.quantile(train_variant_freq, 0.66)) if len(train_variant_freq) > 0 else tail_q1

    train_isoformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        stage_a_epochs=STAGE_A_EPOCHS,
        stage_b_epochs=STAGE_B_EPOCHS,
        lr=LEARNING_RATE,
        device=DEVICE,
        save_path=MODEL_SAVE_PATH,
        tail_q1=tail_q1,
        tail_q2=tail_q2,
        tail_boost=TAIL_BOOST_STAGE_B,
        alpha=LOSS_ALPHA,
        lambda_gamma_l1=LAMBDA_GAMMA_L1,
        bucket_report_path=BUCKET_REPORT_PATH,
    )
