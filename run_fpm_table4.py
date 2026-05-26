# -*- coding: utf-8 -*-
"""Standalone reproduction runner for paper Table 4 FPM LSTM/Transformer experiments.

This script follows the paper repository path:
Main/IPF.py -> FPM feature selection (FeatureSel.LightGBMNew style) ->
NoFill/LSTM and Transformer models. The legacy BPP_Frame package layout is
adapted to the current processed_*.csv datasets through feature_selection_fpm.py.
"""

import argparse
import csv
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

from feature_selection_fpm import prepare_fpm_table4_data, run_fpm_framework_experiment
from run_fpm_vs_baseline import DATASETS, Tee, ensure_dependencies


PAPER_TABLE4_LOCAL_DATASETS = [
    "BPIC2015_1",
    "BPIC2015_2",
    "BPIC2015_3",
    "BPIC2015_4",
    "BPIC2015_5",
    "Helpdesk",
    "Sepsis",
]

MODEL_ALIASES = {
    "lstm": "FPM_LSTM",
    "fpm_lstm": "FPM_LSTM",
    "transformer": "FPM_Transformer",
    "fpm_transformer": "FPM_Transformer",
}

METRIC_FIELDS = [
    "dataset",
    "model",
    "status",
    "error",
    "mae_hour",
    "rmse_hour",
    "score_hour",
    "mae_day",
    "rmse_day",
    "score_day",
    "paper_reference_mae_day",
    "paper_delta_mae_day",
    "best_epoch",
    "epochs",
    "batch_size",
    "lr",
    "seed",
    "selected_feature_count",
    "selected_features",
    "split_strategy",
    "train_trace_count",
    "test_trace_count",
    "elapsed",
    "log",
    "source_path",
    "source_repo",
]

SELECTED_FIELDS = [
    "dataset",
    "seed",
    "rank",
    "feature_index",
    "feature",
    "state",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run standalone paper Table 4 FPM LSTM/Transformer experiments.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset names, e.g. BPIC2015_1 Sepsis. Default: BPIC2015_1.")
    parser.add_argument("--paper-datasets", action="store_true", help="Run Table 4 local paper datasets: BPIC2015_1-5, Helpdesk, Sepsis.")
    parser.add_argument("--all-datasets", action="store_true", help="Run every mapped processed dataset.")
    parser.add_argument("--list-datasets", action="store_true", help="Print available dataset names and exit.")
    parser.add_argument("--models", nargs="*", default=["FPM_LSTM", "FPM_Transformer"], help="Models to run: LSTM Transformer.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs. Paper IPF.py uses 200.")
    parser.add_argument("--batch-size", type=int, default=100, help="Training batch size. Paper IPF.py uses 100.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=80)
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cuda or cpu.")
    parser.add_argument("--output-dir", default="results/fpm_table4")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Overwrite previous Table 4 metric tables instead of upserting.")
    return parser.parse_args()


def normalize_model_names(model_names):
    normalized = []
    for name in model_names:
        value = MODEL_ALIASES.get(str(name).lower(), name)
        if value not in {"FPM_LSTM", "FPM_Transformer"}:
            raise ValueError(f"Table4 standalone runner only supports LSTM/Transformer, got: {name}")
        if value not in normalized:
            normalized.append(value)
    return normalized


def available_dataset_names():
    return [item["dataset"] for item in DATASETS]


def select_datasets(args):
    if args.all_datasets:
        names = available_dataset_names()
    elif args.paper_datasets:
        names = PAPER_TABLE4_LOCAL_DATASETS
    else:
        names = args.datasets or ["BPIC2015_1"]

    wanted = {name.lower() for name in names}
    selected = [
        item for item in DATASETS
        if item["dataset"].lower() in wanted or item["fpm_name"].lower() in wanted
    ]
    missing = [name for name in names if name.lower() not in {item["dataset"].lower() for item in selected}
               and name.lower() not in {item["fpm_name"].lower() for item in selected}]
    if missing:
        raise ValueError(f"No matched datasets: {missing}. Use --list-datasets to inspect valid names.")
    return selected


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows, fields):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def upsert_rows(path, new_rows, fields, key_fields, fresh=False):
    rows = [] if fresh else read_csv_rows(path)
    merged = {tuple(str(row.get(key, "")) for key in key_fields): row for row in rows}
    for row in new_rows:
        merged[tuple(str(row.get(key, "")) for key in key_fields)] = row
    write_csv_rows(path, list(merged.values()), fields)


def run_one_dataset(dataset, workspace_dir, output_dir, args, model_names):
    source_path = os.path.join(workspace_dir, dataset["processed_path"])
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"fpm_table4_{dataset['dataset']}.txt")
    start = time.time()

    metric_rows = []
    selected_rows = []
    print(f"FPM Table4 START | dataset={dataset['dataset']} | path={source_path} | log={log_path}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"Start Time: {datetime.now().isoformat()}", flush=True)
            print("Source Repo: https://github.com/gn874682003/Incremental-Prediction-Framework", flush=True)
            print("Paper Path: Main/IPF.py -> FeatureSel.LightGBMNew -> NoFill/LSTM + Transformer", flush=True)
            print(f"Dataset: {dataset['dataset']}", flush=True)
            print(f"Models: {model_names}", flush=True)
            print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}", flush=True)
            table4_data = prepare_fpm_table4_data(
                raw_dataset_path=source_path,
                dataset_name=dataset["fpm_name"],
                output_dir=output_dir,
                seed=args.seed,
            )
            rows = run_fpm_framework_experiment(
                table4_data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                model_names=model_names,
            )

    elapsed = time.time() - start
    split = table4_data["split_summary"]
    selected_features = ";".join(table4_data["selected_features"])
    for row in rows:
        metric_row = dict(row)
        metric_row.update(
            {
                "dataset": dataset["dataset"],
                "status": "OK",
                "error": "",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "selected_feature_count": len(table4_data["selected_feature_indices"]),
                "selected_features": selected_features,
                "split_strategy": split.get("split_strategy", ""),
                "train_trace_count": split.get("train_trace_count", ""),
                "test_trace_count": split.get("test_trace_count", ""),
                "elapsed": elapsed,
                "log": log_path,
                "source_path": source_path,
                "source_repo": "https://github.com/gn874682003/Incremental-Prediction-Framework",
            }
        )
        metric_rows.append(metric_row)

    for rank, (feature_idx, feature, state) in enumerate(
        zip(
            table4_data["selected_feature_indices"],
            table4_data["selected_features"],
            table4_data["selected_feature_states"],
        ),
        start=1,
    ):
        selected_rows.append(
            {
                "dataset": dataset["dataset"],
                "seed": args.seed,
                "rank": rank,
                "feature_index": feature_idx,
                "feature": feature,
                "state": state,
            }
        )

    print(
        f"FPM Table4 OK | dataset={dataset['dataset']} | elapsed={elapsed:.1f}s | "
        f"selected={len(table4_data['selected_feature_indices'])}",
        flush=True,
    )
    return metric_rows, selected_rows


def main():
    args = parse_args()
    if args.list_datasets:
        print("Available datasets:")
        for name in available_dataset_names():
            marker = " [paper-table4]" if name in PAPER_TABLE4_LOCAL_DATASETS else ""
            print(f"  {name}{marker}")
        return

    ensure_dependencies(skip_install=args.skip_install)
    model_names = normalize_model_names(args.models)
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(workspace_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    datasets = select_datasets(args)
    print(f"Selected datasets: {[item['dataset'] for item in datasets]}", flush=True)
    print(f"Selected models: {model_names}", flush=True)

    all_metric_rows = []
    all_selected_rows = []
    for dataset in datasets:
        print(f"\n=== FPM Table4 Dataset: {dataset['dataset']} ===", flush=True)
        try:
            metric_rows, selected_rows = run_one_dataset(dataset, workspace_dir, output_dir, args, model_names)
            all_metric_rows.extend(metric_rows)
            all_selected_rows.extend(selected_rows)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log_path = os.path.join(output_dir, "logs", f"fpm_table4_{dataset['dataset']}.txt")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            for model_name in model_names:
                all_metric_rows.append(
                    {
                        "dataset": dataset["dataset"],
                        "model": model_name,
                        "status": "FAIL",
                        "error": error,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "seed": args.seed,
                        "log": log_path,
                        "source_path": os.path.join(workspace_dir, dataset["processed_path"]),
                        "source_repo": "https://github.com/gn874682003/Incremental-Prediction-Framework",
                    }
                )
            print(f"FPM Table4 FAIL | dataset={dataset['dataset']} | {error}", flush=True)

    metrics_path = os.path.join(output_dir, "fpm_table4_metrics.csv")
    selected_path = os.path.join(output_dir, "fpm_table4_selected_features.csv")
    upsert_rows(
        metrics_path,
        all_metric_rows,
        METRIC_FIELDS,
        key_fields=["dataset", "model", "seed", "epochs"],
        fresh=args.fresh,
    )
    upsert_rows(
        selected_path,
        all_selected_rows,
        SELECTED_FIELDS,
        key_fields=["dataset", "seed", "rank"],
        fresh=args.fresh,
    )
    print(f"\nSaved Table4 FPM metrics to: {metrics_path}", flush=True)
    print(f"Saved selected features to: {selected_path}", flush=True)


if __name__ == "__main__":
    main()
