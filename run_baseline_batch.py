import argparse
import glob
import os
import subprocess
import sys
import time
from datetime import datetime


def run_one_dataset(
    python_executable,
    baseline_script,
    workspace_dir,
    dataset_path,
    log_path,
):
    cmd = [python_executable, "-u", baseline_script]
    run_env = os.environ.copy()
    run_env["BASELINE_DATASET_PATH"] = dataset_path
    run_env["PYTHONUNBUFFERED"] = "1"

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    start = time.time()
    merged_lines = []

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Start Time: {datetime.now().isoformat()}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"BASELINE_DATASET_PATH={dataset_path}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("=" * 80 + "\n")

        proc = subprocess.Popen(
            cmd,
            cwd=workspace_dir,
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )

        if proc.stdout is not None:
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
                merged_lines.append(line)

        proc.wait()

        elapsed = time.time() - start
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Return Code: {proc.returncode}\n")
        f.write(f"Elapsed Seconds: {elapsed:.2f}\n")

    error_hint = ""
    if proc.returncode != 0:
        non_empty_lines = [line.strip() for line in merged_lines if line.strip()]
        if non_empty_lines:
            error_hint = non_empty_lines[-1]

    return {
        "dataset": dataset_path,
        "log": log_path,
        "return_code": proc.returncode,
        "elapsed": elapsed,
        "error_hint": error_hint,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch run baseline.py on processed datasets and save console logs.")
    parser.add_argument("--dataset-glob", type=str, default="processed_*.csv", help="Glob pattern under dataset directory.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop batch execution on first failed dataset.")
    args = parser.parse_args()

    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(workspace_dir, "dataset")
    baseline_script = os.path.join(workspace_dir, "baseline.py")
    log_dir = os.path.join(workspace_dir, "results", "baseline_batch_logs")

    dataset_glob = args.dataset_glob
    python_executable = sys.executable

    stop_on_error = args.stop_on_error

    if not os.path.exists(baseline_script):
        raise FileNotFoundError(f"baseline.py not found: {baseline_script}")

    dataset_paths = sorted(glob.glob(os.path.join(dataset_dir, dataset_glob)))
    if not dataset_paths:
        raise FileNotFoundError(f"No datasets found with pattern {dataset_glob} in {dataset_dir}")

    print(f"Found {len(dataset_paths)} datasets. Start batch baseline run...")

    summary = []
    for idx, dataset_path in enumerate(dataset_paths, start=1):
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        log_path = os.path.join(log_dir, f"{dataset_name}.txt")

        print(f"[{idx}/{len(dataset_paths)}] Running baseline on {dataset_name} ...")
        result = run_one_dataset(
            python_executable=python_executable,
            baseline_script=baseline_script,
            workspace_dir=workspace_dir,
            dataset_path=dataset_path,
            log_path=log_path,
        )
        summary.append(result)

        status = "OK" if result["return_code"] == 0 else "FAIL"
        if result["return_code"] == 0:
            print(
                f"    -> {status} | return_code={result['return_code']} | "
                f"elapsed={result['elapsed']:.1f}s | log={result['log']}"
            )
        else:
            print(
                f"    -> {status} | return_code={result['return_code']} | "
                f"elapsed={result['elapsed']:.1f}s | error={result['error_hint']} | log={result['log']}"
            )

        if stop_on_error and result["return_code"] != 0:
            print("Stop on error is enabled. Batch run terminated.")
            break

    summary_path = os.path.join(log_dir, "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Batch Start Time: {datetime.now().isoformat()}\n")
        f.write(f"Python: {python_executable}\n")
        f.write(f"Baseline Script: {baseline_script}\n")
        f.write(f"Dataset Glob: {dataset_glob}\n")
        f.write("Hyperparameters: use baseline.py internal settings\n")
        f.write("=" * 80 + "\n")
        for item in summary:
            f.write(
                f"dataset={item['dataset']} | return_code={item['return_code']} | "
                f"elapsed={item['elapsed']:.2f}s | error={item.get('error_hint', '')} | log={item['log']}\n"
            )

    print(f"Batch run finished. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
