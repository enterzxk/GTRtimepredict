# -*- coding: utf-8 -*-
import argparse
import csv
import importlib.util
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime


DATASETS = [
    {
        "dataset": "BPIC2012",
        "fpm_name": "BPIC2012",
        "raw_path": "dataset/processed_BPIC2012.csv",
        "processed_path": "dataset/processed_BPIC2012.csv",
    },
    {
        "dataset": "BPIC2015_1",
        "fpm_name": "BPIC2015_1",
        "raw_path": "dataset/processed_BPIC2015_1.csv",
        "processed_path": "dataset/processed_BPIC2015_1.csv",
    },
    {
        "dataset": "BPIC2015_2",
        "fpm_name": "BPIC2015_2",
        "raw_path": "dataset/processed_BPIC2015_2.csv",
        "processed_path": "dataset/processed_BPIC2015_2.csv",
    },
    {
        "dataset": "BPIC2015_3",
        "fpm_name": "BPIC2015_3",
        "raw_path": "dataset/processed_BPIC2015_3.csv",
        "processed_path": "dataset/processed_BPIC2015_3.csv",
    },
    {
        "dataset": "BPIC2015_4",
        "fpm_name": "BPIC2015_4",
        "raw_path": "dataset/processed_BPIC2015_4.csv",
        "processed_path": "dataset/processed_BPIC2015_4.csv",
    },
    {
        "dataset": "BPIC2015_5",
        "fpm_name": "BPIC2015_5",
        "raw_path": "dataset/processed_BPIC2015_5.csv",
        "processed_path": "dataset/processed_BPIC2015_5.csv",
    },
    {
        "dataset": "BPIC2017",
        "fpm_name": "BPIC2017",
        "raw_path": "dataset/processed_BPIC2017.csv",
        "processed_path": "dataset/processed_BPIC2017.csv",
    },
    {
        "dataset": "BPIC2018",
        "fpm_name": "BPIC2018",
        "raw_path": "dataset/processed_BPIC2018.csv",
        "processed_path": "dataset/processed_BPIC2018.csv",
    },
    {
        "dataset": "BPIC2019",
        "fpm_name": "BPIC2019",
        "raw_path": "dataset/processed_BPIC2019.csv",
        "processed_path": "dataset/processed_BPIC2019.csv",
    },
    {
        "dataset": "BPIC2020",
        "fpm_name": "BPIC2020",
        "raw_path": "dataset/processed_BPIC2020.csv",
        "processed_path": "dataset/processed_BPIC2020.csv",
    },
    {
        "dataset": "BPIC2020_Dom",
        "fpm_name": "BPIC2020_Dom",
        "raw_path": "dataset/processed_BPIC2020_Dom.csv",
        "processed_path": "dataset/processed_BPIC2020_Dom.csv",
    },
    {
        "dataset": "BPIC2020_Inter",
        "fpm_name": "BPIC2020_Inter",
        "raw_path": "dataset/processed_BPIC2020_Inter.csv",
        "processed_path": "dataset/processed_BPIC2020_Inter.csv",
    },
    {
        "dataset": "BPIC2020_Per",
        "fpm_name": "BPIC2020_Per",
        "raw_path": "dataset/processed_BPIC2020_Per.csv",
        "processed_path": "dataset/processed_BPIC2020_Per.csv",
    },
    {
        "dataset": "BPIC2020_Pre",
        "fpm_name": "BPIC2020_Pre",
        "raw_path": "dataset/processed_BPIC2020_Pre.csv",
        "processed_path": "dataset/processed_BPIC2020_Pre.csv",
    },
    {
        "dataset": "BPIC2020_Req",
        "fpm_name": "BPIC2020_Req",
        "raw_path": "dataset/processed_BPIC2020_Req.csv",
        "processed_path": "dataset/processed_BPIC2020_Req.csv",
    },
    {
        "dataset": "Helpdesk",
        "fpm_name": "Helpdesk",
        "raw_path": "dataset/processed_Helpdesk.csv",
        "processed_path": "dataset/processed_Helpdesk.csv",
    },
    {
        "dataset": "Sepsis",
        "fpm_name": "Sepsis",
        "raw_path": "dataset/processed_Sepsis.csv",
        "processed_path": "dataset/processed_Sepsis.csv",
    },
    {
        "dataset": "Wind_Sheet1",
        "fpm_name": "Wind_Sheet1",
        "raw_path": "dataset/processed_Wind_Sheet1.csv",
        "processed_path": "dataset/processed_Wind_Sheet1.csv",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run paper-style FPM feature selection and baseline.py M0-M5 comparison.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset names to run, e.g. BPIC2015_1 Sepsis. Default: BPIC2015_1.",
    )
    parser.add_argument("--all-datasets", action="store_true", help="Run every mapped processed dataset.")
    parser.add_argument("--list-datasets", action="store_true", help="Print available dataset names and exit.")
    parser.add_argument("--output-dir", default="results/fpm_vs_baseline")
    parser.add_argument("--seed", type=int, default=80)
    parser.add_argument("--skip-install", action="store_true", help="Do not auto-install missing torch/lightgbm.")
    parser.add_argument("--skip-fpm", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true", default=None)
    parser.add_argument("--run-baseline", action="store_true", help="Run baseline.py after FPM. Table4 mode skips baseline by default.")
    parser.add_argument("--fpm-mode", choices=["table3", "table4", "framework", "full"], default="table4")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for FPM framework models.")
    parser.add_argument(
        "--fpm-framework-models",
        nargs="*",
        default=None,
        help="FPM Table4 models to train. Default: FPM_LSTM FPM_Transformer; full mode also adds FPM_AETS.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run BPIC2015_1 only with tiny baseline epochs.")
    parser.add_argument("--baseline-epochs-short", type=int, default=None)
    parser.add_argument("--baseline-stage-a-epochs", type=int, default=None)
    parser.add_argument("--baseline-stage-b-epochs", type=int, default=None)
    parser.add_argument("--baseline-split-strategy", choices=["row", "case"], default="row")
    parser.add_argument("--baseline-models", nargs="*", default=None, help="Optional baseline.py model names to run.")
    parser.add_argument("--baseline-timeout-seconds", type=int, default=0, help="0 means no timeout.")
    parser.add_argument("--baseline-heartbeat-seconds", type=int, default=15)
    parser.add_argument("--smoke-baseline-cases", type=int, default=30)
    return parser.parse_args()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def ensure_dependencies(skip_install=False):
    required = ["lightgbm", "torch"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if not missing:
        return
    if skip_install:
        raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")

    cmd = [sys.executable, "-m", "pip", "install", *missing]
    print(f"Installing missing dependencies: {' '.join(missing)}")
    subprocess.check_call(cmd)
    importlib.invalidate_caches()


def list_dataset_names():
    return [item["dataset"] for item in DATASETS]


def select_datasets(names=None, smoke=False, all_datasets=False):
    if smoke:
        return [DATASETS[0]]
    if all_datasets or (names and "all" in {name.lower() for name in names}):
        return DATASETS
    if not names:
        names = ["BPIC2015_1"]

    wanted = {name.lower() for name in names}
    selected = [
        item for item in DATASETS
        if item["dataset"].lower() in wanted or item["fpm_name"].lower() in wanted
    ]
    if not selected:
        raise ValueError(f"No matched datasets: {names}")
    return selected


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def make_smoke_processed_dataset(dataset, workspace_dir, output_dir, case_limit):
    import pandas as pd

    source_path = os.path.join(workspace_dir, dataset["processed_path"])
    smoke_dir = os.path.join(output_dir, "smoke_data")
    os.makedirs(smoke_dir, exist_ok=True)
    smoke_path = os.path.join(smoke_dir, f"processed_{dataset['dataset']}_smoke.csv")

    df = pd.read_csv(source_path)
    case_ids = df["CaseID"].drop_duplicates().head(max(int(case_limit), 2))
    smoke_df = df[df["CaseID"].isin(set(case_ids))].copy()
    smoke_df.to_csv(smoke_path, index=False, encoding="utf-8-sig")
    return smoke_path


def run_fpm(dataset, workspace_dir, output_dir, seed, args):
    from feature_selection_fpm import run_fpm_framework_experiment, run_fpm_table3_experiment

    raw_path = os.path.join(workspace_dir, dataset["raw_path"])
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"FPM dataset not found: {raw_path}")

    log_path = os.path.join(output_dir, "logs", f"fpm_{dataset['dataset']}.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    start = time.time()
    print(f"FPM START | dataset={dataset['dataset']} | data={raw_path} | log={log_path}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"Start Time: {datetime.now().isoformat()}")
            print(f"Dataset: {dataset['dataset']}")
            print(f"Raw Path: {raw_path}")
            table3_result = run_fpm_table3_experiment(
                raw_dataset_path=raw_path,
                dataset_name=dataset["fpm_name"],
                output_dir=output_dir,
                seed=seed,
            )
            framework_rows = []
            if args.fpm_mode in {"table4", "framework", "full"}:
                framework_model_names = args.fpm_framework_models
                if framework_model_names is None:
                    framework_model_names = (
                        ["FPM_LSTM", "FPM_Transformer", "FPM_AETS"]
                        if args.fpm_mode == "full"
                        else ["FPM_LSTM", "FPM_Transformer"]
                    )
                framework_rows = run_fpm_framework_experiment(
                    table3_result,
                    epochs=args.epochs,
                    model_names=framework_model_names,
                )
            print("[FPM:Incremental] period/month: placeholder, quantity/100: placeholder, concept-drift: placeholder")
            print(f"Elapsed Seconds: {time.time() - start:.2f}")
    table3_rows = table3_result["table3_rows"]
    efc_row = next((row for row in table3_rows if row["feature_set"] == "EFC"), table3_rows[0])
    result = {
        "dataset": dataset["dataset"],
        "status": "OK",
        "error": "",
        "log": log_path,
        "elapsed": time.time() - start,
        "selected_feature_indices": table3_result["selected_feature_indices"],
        "selected_features": table3_result["selected_features"],
        "selected_feature_states": table3_result["selected_feature_states"],
        "selected_feature_count": len(table3_result["selected_feature_indices"]),
        "table3_rows": table3_rows,
        "framework_rows": framework_rows,
        "mae": efc_row["mae_hour"],
    }
    print(
        f"FPM DONE | dataset={dataset['dataset']} | elapsed={result['elapsed']:.1f}s | "
        f"EFC_MAE_hour={result['mae']:.4f} | selected={result['selected_feature_count']}",
        flush=True,
    )
    return result


def run_baseline(dataset, workspace_dir, output_dir, seed, args):
    dataset_output_dir = os.path.join(output_dir, "baseline", dataset["dataset"])
    log_path = os.path.join(output_dir, "logs", f"baseline_{dataset['dataset']}.txt")
    metrics_path = os.path.join(dataset_output_dir, f"ablation_metrics_{dataset['dataset']}.csv")
    bucket_path = os.path.join(dataset_output_dir, f"ablation_bucket_metrics_{dataset['dataset']}.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(dataset_output_dir, exist_ok=True)

    processed_path = (
        make_smoke_processed_dataset(dataset, workspace_dir, output_dir, args.smoke_baseline_cases)
        if args.smoke
        else os.path.join(workspace_dir, dataset["processed_path"])
    )
    if not os.path.exists(processed_path):
        return [
            {
                "dataset": dataset["dataset"],
                "model": "baseline.py",
                "status": "FAIL",
                "error": f"dataset_not_found={processed_path}",
                "log": log_path,
                "elapsed": 0.0,
            }
        ]
    epochs_short = 1 if args.smoke else (args.baseline_epochs_short or 20)
    stage_a_epochs = 1 if args.smoke else (args.baseline_stage_a_epochs or 40)
    stage_b_epochs = 0 if args.smoke else (args.baseline_stage_b_epochs if args.baseline_stage_b_epochs is not None else 10)
    max_seq_length = 20 if args.smoke else 50
    max_prefixes = 20 if args.smoke else 100
    batch_size = 32 if args.smoke else 128

    cmd = [
        sys.executable,
        "-u",
        os.path.join(workspace_dir, "baseline.py"),
        "--dataset-path",
        processed_path,
        "--report-dir",
        dataset_output_dir,
        "--seed",
        str(seed),
        "--epochs-short",
        str(epochs_short),
        "--stage-a-epochs",
        str(stage_a_epochs),
        "--stage-b-epochs",
        str(stage_b_epochs),
        "--split-strategy",
        args.baseline_split_strategy,
        "--max-seq-length",
        str(max_seq_length),
        "--max-prefixes",
        str(max_prefixes),
        "--batch-size",
        str(batch_size),
    ]
    if args.baseline_models:
        cmd.append("--models")
        cmd.extend(args.baseline_models)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["BASELINE_METRICS_REPORT_PATH"] = metrics_path
    env["BASELINE_BUCKET_REPORT_PATH"] = bucket_path

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Start Time: {datetime.now().isoformat()}\n")
        log_file.write(f"Dataset: {dataset['dataset']}\n")
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.write("=" * 80 + "\n")
        log_file.flush()
        print(f"Baseline START | dataset={dataset['dataset']} | log={log_path}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=workspace_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        timed_out = False
        last_output = time.time()
        if proc.stdout is not None:
            while True:
                line = proc.stdout.readline()
                if line:
                    print(line, end="", flush=True)
                    log_file.write(line)
                    log_file.flush()
                    last_output = time.time()
                elif proc.poll() is not None:
                    break
                else:
                    if args.baseline_timeout_seconds > 0 and time.time() - start > args.baseline_timeout_seconds:
                        timed_out = True
                        proc.kill()
                        break
                    if time.time() - last_output > max(int(args.baseline_heartbeat_seconds), 5):
                        msg = f"[heartbeat] baseline still running for {time.time() - start:.1f}s...\n"
                        print(msg, end="", flush=True)
                        log_file.write(msg)
                        log_file.flush()
                        last_output = time.time()
                    time.sleep(1)
        return_code = proc.wait()
        log_file.write("\n" + "=" * 80 + "\n")
        if timed_out:
            log_file.write(f"Timed Out: {args.baseline_timeout_seconds}s\n")
        log_file.write(f"Return Code: {return_code}\n")
        log_file.write(f"Elapsed Seconds: {time.time() - start:.2f}\n")

    rows = []
    if return_code == 0 and os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["dataset"] = dataset["dataset"]
            row["status"] = "OK"
            row["error"] = ""
            row["log"] = log_path
            row["elapsed"] = time.time() - start
    else:
        error = "timeout" if timed_out else f"return_code={return_code}"
        rows.append(
            {
                "dataset": dataset["dataset"],
                "model": "baseline.py",
                "status": "FAIL",
                "error": error,
                "log": log_path,
                "elapsed": time.time() - start,
            }
        )
    return rows


def main():
    args = parse_args()
    if args.list_datasets:
        print("Available datasets:")
        for name in list_dataset_names():
            print(f"  {name}")
        return
    if args.skip_baseline is None:
        args.skip_baseline = args.fpm_mode == "table4" and not args.run_baseline
    elif args.run_baseline:
        args.skip_baseline = False
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(workspace_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ensure_dependencies(skip_install=args.skip_install)
    datasets = select_datasets(args.datasets, smoke=args.smoke, all_datasets=args.all_datasets)
    print(f"Selected datasets: {[item['dataset'] for item in datasets]}")

    fpm_table3_rows = []
    fpm_framework_rows = []
    selected_rows = []
    baseline_rows = []
    comparison_rows = []

    for dataset in datasets:
        print(f"\n=== Dataset: {dataset['dataset']} ===")

        if not args.skip_fpm:
            try:
                fpm_result = run_fpm(dataset, workspace_dir, output_dir, args.seed, args)
                for row in fpm_result["table3_rows"]:
                    row = dict(row)
                    row.update({"status": "OK", "error": "", "log": fpm_result["log"], "elapsed": fpm_result["elapsed"]})
                    fpm_table3_rows.append(row)
                    comparison_rows.append(
                        {
                            "dataset": row["dataset"],
                            "method_group": "FPM_Table3",
                            "model": row["feature_set"],
                            "mae": row["mae_hour"],
                            "tail_mae": row["tail_mae_hour"],
                            "rmse": row["rmse_hour"],
                            "score": row["score_hour"],
                            "mae_day": row["mae_day"],
                            "paper_reference_mae_day": row["paper_reference_mae_day"],
                            "paper_delta_mae_day": row.get("paper_delta_mae_day", ""),
                            "status": "OK",
                            "error": "",
                            "log": fpm_result["log"],
                        }
                    )
                for row in fpm_result["framework_rows"]:
                    row = dict(row)
                    row.update({"status": "OK", "error": "", "log": fpm_result["log"], "elapsed": fpm_result["elapsed"]})
                    fpm_framework_rows.append(row)
                    comparison_rows.append(
                        {
                            "dataset": row["dataset"],
                            "method_group": "FPM_Table4",
                            "model": row["model"],
                            "mae": row["mae_hour"],
                            "tail_mae": "",
                            "rmse": row["rmse_hour"],
                            "score": row["score_hour"],
                            "mae_day": row["mae_day"],
                            "paper_reference_mae_day": row.get("paper_reference_mae_day", ""),
                            "paper_delta_mae_day": row.get("paper_delta_mae_day", ""),
                            "status": "OK",
                            "error": "",
                            "log": fpm_result["log"],
                        }
                    )
                for rank, (feature_index, feature, state) in enumerate(
                    zip(
                        fpm_result["selected_feature_indices"],
                        fpm_result["selected_features"],
                        fpm_result["selected_feature_states"],
                    ),
                    start=1,
                ):
                    selected_rows.append(
                        {
                            "dataset": fpm_result["dataset"],
                            "rank": rank,
                            "feature_index": feature_index,
                            "feature": feature,
                            "state": state,
                        }
                    )
                print(f"FPM OK | EFC_MAE_hour={fpm_result['mae']:.4f} | selected={fpm_result['selected_feature_count']}")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                log_path = os.path.join(output_dir, "logs", f"fpm_{dataset['dataset']}.txt")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
                fpm_table3_rows.append({"dataset": dataset["dataset"], "status": "FAIL", "error": error, "log": log_path})
                comparison_rows.append(
                    {
                        "dataset": dataset["dataset"],
                        "method_group": "FPM_Table3",
                        "model": "FPM",
                        "status": "FAIL",
                        "error": error,
                        "log": log_path,
                    }
                )
                print(f"FPM FAIL | {error}")

        if not args.skip_baseline:
            rows = run_baseline(dataset, workspace_dir, output_dir, args.seed, args)
            baseline_rows.extend(rows)
            for row in rows:
                comparison_rows.append(
                    {
                        "dataset": row.get("dataset", dataset["dataset"]),
                        "method_group": "Baseline",
                        "model": row.get("model", ""),
                        "mae": row.get("mae", ""),
                        "tail_mae": row.get("tail_mae", ""),
                        "rmse": row.get("rmse", ""),
                        "score": row.get("score", ""),
                        "mae_day": (float(row["mae"]) / 24.0) if row.get("mae", "") not in {"", None} else "",
                        "paper_reference_mae_day": "",
                        "paper_delta_mae_day": "",
                        "status": row.get("status", ""),
                        "error": row.get("error", ""),
                        "log": row.get("log", ""),
                    }
                )
            ok_count = sum(1 for row in rows if row.get("status") == "OK")
            print(f"Baseline rows: {ok_count}/{len(rows)} OK")

    write_csv(
        os.path.join(output_dir, "fpm_table3_metrics.csv"),
        fpm_table3_rows,
        [
            "dataset", "feature_set", "selected_feature_count",
            "mae_hour", "tail_mae_hour", "rmse_hour", "score_hour",
            "mae_day", "tail_mae_day", "rmse_day", "score_day",
            "paper_reference_mae_day", "paper_delta_mae_day",
            "tail_q1", "status", "error", "log", "elapsed",
        ],
    )
    write_csv(
        os.path.join(output_dir, "fpm_framework_metrics.csv"),
        fpm_framework_rows,
        [
            "dataset", "model", "mae_hour", "rmse_hour", "score_hour",
            "mae_day", "rmse_day", "score_day", "best_epoch",
            "paper_reference_mae_day", "paper_delta_mae_day",
            "incremental_period_month", "incremental_quantity_100", "concept_drift",
            "status", "error", "log", "elapsed",
        ],
    )
    write_csv(
        os.path.join(output_dir, "fpm_table4_metrics.csv"),
        fpm_framework_rows,
        [
            "dataset", "model", "mae_hour", "rmse_hour", "score_hour",
            "mae_day", "rmse_day", "score_day", "best_epoch",
            "paper_reference_mae_day", "paper_delta_mae_day",
            "incremental_period_month", "incremental_quantity_100", "concept_drift",
            "status", "error", "log", "elapsed",
        ],
    )
    write_csv(
        os.path.join(output_dir, "fpm_selected_features.csv"),
        selected_rows,
        ["dataset", "rank", "feature_index", "feature", "state"],
    )
    write_csv(
        os.path.join(output_dir, "baseline_metrics.csv"),
        baseline_rows,
        ["dataset", "model", "mae", "tail_mae", "rmse", "score", "status", "error", "log", "elapsed"],
    )
    write_csv(
        os.path.join(output_dir, "comparison_summary.csv"),
        comparison_rows,
        [
            "dataset", "method_group", "model", "mae", "tail_mae", "rmse", "score",
            "mae_day", "paper_reference_mae_day", "paper_delta_mae_day", "status", "error", "log",
        ],
    )
    print(f"\nSaved comparison outputs to: {output_dir}")


if __name__ == "__main__":
    main()
