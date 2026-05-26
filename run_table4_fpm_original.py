# -*- coding: utf-8 -*-
"""Faithful Table 4 FPM runner based on the original IPF.py pipeline.

This entrypoint intentionally calls the original repository modules for:
LogConvert -> DivideData -> FeatureSel.LightGBM/PrefixLightGBM ->
word2vec -> Method.My.multiModel.trian/trianT.
"""

import argparse
import csv
import importlib
import inspect
import os
import random
import re
import sys
import time
import traceback
import types
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime

import numpy as np


def patch_numpy_legacy_aliases():
    """Compatibility patch for old upstream code under NumPy >= 2.0."""
    np_major = int(np.__version__.split(".")[0])
    if np_major < 2:
        return
    legacy_aliases = {
        "Inf": np.inf,
        "Infinity": np.inf,
        "infty": np.inf,
        "NaN": np.nan,
        "NAN": np.nan,
    }

    for name, value in legacy_aliases.items():
        np.__dict__[name] = value
        numpy_module = sys.modules.get("numpy")
        if numpy_module is not None:
            numpy_module.__dict__[name] = value
    print(f"[Compat] NumPy {np.__version__} >= 2.0: patched legacy aliases (Inf, NaN, etc.)", flush=True)


patch_numpy_legacy_aliases()

import torch
import torch.nn as nn
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)


SOURCE_REPO = "https://github.com/gn874682003/Incremental-Prediction-Framework"

DATASET_CONFIG = {
    "BPIC2015_1": {
        "eventlog": "BPIC2015_1",
        "raw_file": "BPIC2015_1.csv",
        "attribute": list(range(3, 19)),
    },
    "BPIC2015_2": {
        "eventlog": "BPIC2015_2",
        "raw_file": "BPIC2015_2.csv",
        "attribute": list(range(3, 19)),
    },
    "BPIC2015_3": {
        "eventlog": "BPIC2015_3",
        "raw_file": "BPIC2015_3.csv",
        "attribute": list(range(3, 19)),
    },
    "BPIC2015_4": {
        "eventlog": "BPIC2015_4",
        "raw_file": "BPIC2015_4.csv",
        "attribute": list(range(3, 19)),
    },
    "BPIC2015_5": {
        "eventlog": "BPIC2015_5",
        "raw_file": "BPIC2015_5.csv",
        "attribute": list(range(3, 19)),
    },
    "Helpdesk": {
        "eventlog": "hd",
        "raw_file": "hd.csv",
        "attribute": list(range(3, 14)),
    },
    "hd": {
        "eventlog": "hd",
        "raw_file": "hd.csv",
        "attribute": list(range(3, 14)),
    },
    "Sepsis": {
        "eventlog": "sepsis",
        "raw_file": "sepsis.csv",
        "attribute": [3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    },
    "sepsis": {
        "eventlog": "sepsis",
        "raw_file": "sepsis.csv",
        "attribute": [3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    },
}

PAPER_REFERENCE = {
    "BPIC2015_1": {"LSTM": 29.26, "PT": 24.71, "LSTM_FPM": 26.44, "Transformer_FPM": 26.52},
    "BPIC2015_2": {"LSTM_FPM": 66.69, "Transformer_FPM": 68.21},
    "BPIC2015_3": {"LSTM_FPM": 18.40, "Transformer_FPM": 16.80},
    "BPIC2015_4": {"LSTM_FPM": 49.02, "Transformer_FPM": 54.58},
    "BPIC2015_5": {"LSTM_FPM": 35.85, "Transformer_FPM": 36.45},
    "Helpdesk": {"LSTM_FPM": 4.59, "Transformer_FPM": 1.18},
    "Sepsis": {"LSTM_FPM": 30.64, "Transformer_FPM": 27.46},
}

PAPER_DATASETS = ["BPIC2015_1", "BPIC2015_2", "BPIC2015_3", "BPIC2015_4", "BPIC2015_5", "Helpdesk", "Sepsis"]

METRIC_FIELDS = [
    "dataset", "eventlog", "experiment_mode", "model", "status", "error",
    "mae_day", "paper_reference_mae_day", "paper_delta_mae_day",
    "epochs", "batch_size", "seed", "selector_seed", "split_mode", "max_traces",
    "embedding_scope", "run_type", "is_exact_paper_baseline", "baseline_note",
    "raw_file", "raw_md5", "raw_file_size",
    "trace_count", "train_trace_count", "test_trace_count",
    "selected_feature_count", "selected_features", "selected_feature_indices",
    "prefix_features", "cbow_dim", "categorical_embedding_dims",
    "max_case_length", "python_version", "numpy_version", "scipy_version",
    "sklearn_version", "lightgbm_version", "torch_version", "pandas_version",
    "elapsed", "log", "source_repo",
]

TABLE3_FIELDS = [
    "dataset",
    "eventlog",
    "experiment_mode",
    "split_mode",
    "seed",
    "selector_seed",
    "embedding_scope",
    "run_type",
    "max_traces",
    "raw_file",
    "raw_md5",
    "raw_file_size",
    "trace_count",
    "train_trace_count",
    "test_trace_count",
    "max_case_length",
    "activity_mae_day",
    "all_features_mae_day",
    "efc_mae_day",
    "efc_feature_count",
    "efc_features",
    "efc_indices",
    "prefix_mae_day",
    "prefix_mae_log",
    "prefix_features",
    "python_version",
    "numpy_version",
    "scipy_version",
    "sklearn_version",
    "lightgbm_version",
    "torch_version",
    "pandas_version",
    "log",
]

TABLE3_SUMMARY_FIELDS = [
    "dataset",
    "eventlog",
    "experiment_mode",
    "split_mode",
    "selector_seed",
    "embedding_scope",
    "run_type",
    "max_traces",
    "n",
    "activity_mae_day",
    "all_features_mae_day",
    "efc_mae_day",
    "efc_feature_count",
    "prefix_mae_day",
    "prefix_features",
    "activity_mae_day_min",
    "activity_mae_day_max",
    "all_features_mae_day_min",
    "all_features_mae_day_max",
    "efc_mae_day_min",
    "efc_mae_day_max",
    "efc_feature_count_min",
    "efc_feature_count_max",
    "prefix_mae_day_min",
    "prefix_mae_day_max",
]

SELECTED_FIELDS = [
    "dataset", "eventlog", "experiment_mode", "split_mode", "seed", "selector_seed", "embedding_scope", "rank", "feature_index", "feature", "state",
]

SEED_SUMMARY_FIELDS = [
    "dataset", "model", "experiment_mode", "split_mode", "selector_seed", "embedding_scope", "n",
    "mean_mae_day", "std_mae_day", "min_mae_day", "max_mae_day",
    "paper_reference_mae_day", "mean_delta_day",
]


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


class CaptureTee:
    def __init__(self, stream):
        self.stream = stream
        self.parts = []

    def write(self, data):
        self.parts.append(data)
        self.stream.write(data)
        self.stream.flush()

    def flush(self):
        self.stream.flush()

    def getvalue(self):
        return "".join(self.parts)


def file_md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size(path):
    return os.path.getsize(path)


def dependency_versions():
    import sklearn
    import lightgbm
    import scipy
    import pandas
    versions = {
        "python_version": sys.version.replace("\n", " "),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lightgbm.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pandas.__version__,
    }
    return versions


def print_dependency_versions():
    import sys
    print("Python:", sys.version, flush=True)

    import numpy
    print("numpy:", numpy.__version__, flush=True)
    print("NumPy legacy aliases patched:", "Inf" in np.__dict__, flush=True)

    import scipy
    print("scipy:", scipy.__version__, flush=True)

    import sklearn
    print("sklearn:", sklearn.__version__, flush=True)

    import lightgbm
    print("lightgbm:", lightgbm.__version__, flush=True)

    import torch
    print("torch:", torch.__version__, flush=True)

    import pandas
    print("pandas:", pandas.__version__, flush=True)

    for module_name in ["matplotlib", "seaborn", "xgboost", "catboost", "gensim", "minepy", "graphviz"]:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"{module_name}: {version}", flush=True)
        except Exception as exc:
            print(f"{module_name}: not installed ({type(exc).__name__}: {exc})", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Faithfully reproduce paper Table 4 FPM LSTM/Transformer results.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets, e.g. BPIC2015_1 Helpdesk Sepsis. Default: BPIC2015_1.")
    parser.add_argument("--paper-datasets", action="store_true", help="Run BPIC2015_1-5, Helpdesk, Sepsis.")
    parser.add_argument(
        "--run-all-paper",
        action="store_true",
        help="Shortcut for running all paper datasets. Equivalent to --paper-datasets.",
    )
    parser.add_argument(
        "--default-all-paper",
        action="store_true",
        help="If no --datasets is provided, run all paper datasets by default.",
    )
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["LSTM_ActivityOnly_Approx", "Transformer_ActivityOnly_Approx", "LSTM_FPM", "Transformer_FPM"],
        help="LSTM_ActivityOnly_Approx Transformer_ActivityOnly_Approx LSTM_FPM Transformer_FPM LSTM_AllFeatures_Diagnostic Transformer_AllFeatures_Diagnostic",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=80)
    parser.add_argument(
        "--selector-seed",
        type=int,
        default=20,
        help="Random state for train_test_split inside FeatureSel stage. Default 20 follows original IPF.py.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["divi", "time_year"],
        default="divi",
        help="Dataset split mode. 'divi' uses only DD.DiviData; 'time_year' uses DD.DiviData plus DD.DiviDataByTime(0.5, 'year').",
    )
    parser.add_argument("--max-traces", type=int, default=0, help="Smoke-test only: limit train/test traces after original split. 0 means full data.")
    parser.add_argument(
        "--embedding-scope",
        type=str,
        default="original_all_data",
        choices=["original_all_data", "train_only"],
        help=(
            "CBOW word2vec training data scope. "
            "original_all_data = use DR.AllData (train+test) matching original IPF.py code() — has data leakage; "
            "train_only = use DR.Train only — clean ML practice."
        ),
    )
    parser.add_argument("--table3-only", action="store_true", help="Run LogC, split, LightGBM and PrefixLightGBM only; skip CBOW and neural model training.")
    parser.add_argument("--output-dir", default="results/table4_fpm_original")
    parser.add_argument("--fresh", action="store_true", help="Overwrite CSVs instead of upserting rows.")
    parser.add_argument(
        "--seed-sweep",
        type=str,
        default="",
        help="Comma-separated seeds, e.g. 20,42,80,123. If provided, run all seeds sequentially.",
    )
    parser.add_argument(
        "--selector-seed-sweep",
        type=str,
        default="",
        help="Comma-separated selector seeds for Table3 diagnostics, e.g. 1,5,10,20,42,80,123.",
    )
    parser.add_argument(
        "--split-sweep",
        action="store_true",
        help="Run both divi and time_year split modes for Table3 diagnostics.",
    )
    parser.add_argument(
        "--experiment-mode",
        type=str,
        default="main",
        choices=["main", "sensitivity", "both"],
        help=(
            "Experiment mode. "
            "main = strict reproduction with split=divi and selector_seed=20; "
            "sensitivity = time_year sensitivity analysis with selector_seed=20; "
            "both = run both main and sensitivity."
        ),
    )
    return parser.parse_args()


def install_original_import_aliases(workspace_dir):
    patch_numpy_legacy_aliases()
    if workspace_dir not in sys.path:
        sys.path.insert(0, workspace_dir)

    frame_pkg = sys.modules.setdefault("Frame", types.ModuleType("Frame"))
    frame_pkg.__path__ = []
    bpp_pkg = sys.modules.setdefault("BPP_Frame", types.ModuleType("BPP_Frame"))
    bpp_pkg.__path__ = []

    import Log.Repeat as repeat
    sys.modules["Log.Repeat"] = repeat
    sys.modules["Frame.Repeat"] = repeat
    sys.modules["Repeat"] = repeat

    import Feature.plotFig as plot_fig
    sys.modules["Feature.plotFig"] = plot_fig
    sys.modules["Frame.plotFig"] = plot_fig
    sys.modules["plotFig"] = plot_fig

    import Feature.tree as tree
    sys.modules["Feature.tree"] = tree
    sys.modules["Frame.tree"] = tree
    sys.modules["tree"] = tree

    import Method.My.Model as model
    sys.modules["Method.My.Model"] = model
    sys.modules["Frame.Model"] = model
    sys.modules["Model"] = model

    for package_name in ["Log", "Feature", "Method", "Main", "Code"]:
        module = importlib.import_module(package_name)
        sys.modules[f"BPP_Frame.{package_name}"] = module
    for module_name in [
        "Log.DataRecord", "Log.DivideData", "Log.LogAnalysis", "Log.LogConvert", "Log.Prefix", "Log.Repeat",
        "Feature.FeatureSel", "Feature.tree", "Feature.plotFig",
        "Method.My.multiModel", "Code.word2vec",
    ]:
        module = importlib.import_module(module_name)
        sys.modules[f"BPP_Frame.{module_name}"] = module
    patch_numpy_legacy_aliases()


def import_original_modules(workspace_dir):
    patch_numpy_legacy_aliases()
    install_original_import_aliases(workspace_dir)
    patch_numpy_legacy_aliases()
    import Log.DataRecord as DR
    import Log.DivideData as DD
    import Log.LogAnalysis as LA
    import Log.LogConvert as LC
    import Log.Prefix as P
    import Feature.FeatureSel as FS
    import Code.word2vec as w2v
    import Method.My.multiModel as M
    patch_numpy_legacy_aliases()
    return DR, DD, LA, LC, P, FS, w2v, M


def set_repro_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def canonical_dataset_name(name):
    lower = str(name).lower()
    if lower == "hd":
        return "Helpdesk"
    if lower == "sepsis":
        return "Sepsis"
    return name


def selected_configs(names, paper_datasets=False):
    names = PAPER_DATASETS if paper_datasets else (names or ["BPIC2015_1"])
    configs = []
    for name in names:
        if name not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset {name}. Use --list-datasets.")
        config = dict(DATASET_CONFIG[name])
        config["dataset"] = canonical_dataset_name(name)
        configs.append(config)
    return configs


def parse_int_list(value, default_value, label):
    if not str(value or "").strip():
        return [int(default_value)]
    items = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            items.append(int(raw))
        except ValueError as exc:
            raise ValueError(f"Invalid {label} value: {raw}") from exc
    if not items:
        raise ValueError(f"{label} did not contain any integer values: {value}")
    return items


def split_experiment_mode(split_mode):
    if split_mode == "divi":
        return "main"
    if split_mode == "time_year":
        return "sensitivity"
    return ""


def normalize_models(models):
    normalized = []
    for model in models:
        key = str(model).lower()
        if key in {"lstm", "lstm_activityonly", "lstm_activity_only", "lstm_activityonly_approx", "lstm_activity_only_approx"}:
            value = "LSTM_ActivityOnly_Approx"
        elif key in {"transformer", "transformer_activityonly", "transformer_activity_only", "transformer_activityonly_approx", "transformer_activity_only_approx"}:
            value = "Transformer_ActivityOnly_Approx"
        elif key in {"lstm_fpm", "fpm_lstm"}:
            value = "LSTM_FPM"
        elif key in {"transformer_fpm", "fpm_transformer"}:
            value = "Transformer_FPM"
        elif key in {"lstm_nofpm", "lstm_no_fpm", "lstm_allfeatures", "lstm_allfeatures_diagnostic"}:
            value = "LSTM_AllFeatures_Diagnostic"
        elif key in {"transformer_nofpm", "transformer_no_fpm", "transformer_allfeatures", "transformer_allfeatures_diagnostic"}:
            value = "Transformer_AllFeatures_Diagnostic"
        else:
            raise ValueError(f"Unsupported model for Table4 FPM: {model}")
        if value not in normalized:
            normalized.append(value)
    return normalized


@contextmanager
def patched_raw_reader(LC, raw_path):
    original_readcsv = LC.FR.readcsv

    def _readcsv(_ignored_path):
        return original_readcsv(raw_path)

    LC.FR.readcsv = _readcsv
    try:
        yield
    finally:
        LC.FR.readcsv = original_readcsv


def reset_data_record(DR):
    for name in [
        "Convert", "AllData", "header", "ConvertReflact", "FilterLog", "Train", "Test",
        "Train_X", "Train_Y", "Test_X", "Test_Y", "model", "Metric", "State", "Tests",
    ]:
        setattr(DR, name, [])


def build_data_record(DR, DD, LC, config, raw_path, max_traces=0, split_mode="divi"):
    reset_data_record(DR)
    attribute = config["attribute"]
    with patched_raw_reader(LC, raw_path):
        DR.Convert, DR.header, DR.ConvertReflact, _max_a, _max_r = LC.LogC(config["eventlog"], attribute)

    DR.State = []
    DR.State.append(0)
    for i in range(4, len(DR.Convert[0]) - 9):
        if i in attribute:
            DR.State.append(1)
        else:
            DR.State.append(3)
    for _ in range(6):
        DR.State.append(3)

    DR.Train, DR.Test, DR.AllData = DD.DiviData(DR.Convert, DR.State)
    if split_mode == "time_year":
        DR.Train, DR.Test, DR.Tests = DD.DiviDataByTime(DR.AllData, 0.5, "year")
    elif split_mode == "divi":
        DR.Tests = []
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")
    if max_traces and max_traces > 0:
        train_limit = max(int(max_traces), 5)
        test_limit = max(int(max_traces) // 4, 2)
        DR.Train = DR.Train[:train_limit]
        DR.Test = DR.Test[:test_limit]
        DR.AllData = DR.Train + DR.Test
    return DR


def _extract_first_float(pattern, text):
    match = re.search(pattern, text)
    if not match:
        return ""
    try:
        return float(match.group(1))
    except Exception:
        return match.group(1)


def parse_table3_capture(text):
    prefix_lines = []
    prefix_numeric_lines = []
    after_prefix = False
    for line in text.splitlines():
        if "PrefixLightGBM start" in line:
            after_prefix = True
        if after_prefix and "NewMAE" in line:
            prefix_lines.append(line.strip())
        if after_prefix and re.match(r"^\s*[-+0-9.eE]+\s*$", line):
            prefix_numeric_lines.append(line.strip())
    prefix_mae_log = prefix_lines[-1] if prefix_lines else ""
    prefix_mae_day = _extract_first_float(r"NewMAE\s*[:=]\s*([-+0-9.eE]+)", prefix_mae_log) if prefix_mae_log else ""
    if prefix_mae_day == "" and prefix_numeric_lines:
        prefix_mae_log = prefix_numeric_lines[-1]
        prefix_mae_day = _extract_first_float(r"^\s*([-+0-9.eE]+)\s*$", prefix_mae_log)
    activity_mae = _extract_first_float(r"NR:Activity\s*[:=]?\s*([-+0-9.eE]+)", text)
    all_features_mae = _extract_first_float(r"NR:All\s*[:=]?\s*([-+0-9.eE]+)", text)
    if not activity_mae:
        print("[Warning] parse_table3_capture: failed to extract activity_mae_day (NR:Activity)", flush=True)
    if not all_features_mae:
        print("[Warning] parse_table3_capture: failed to extract all_features_mae_day (NR:All)", flush=True)
    if not prefix_mae_day:
        print("[Warning] parse_table3_capture: failed to extract prefix_mae_day (NewMAE)", flush=True)
    return {
        "activity_mae_day": activity_mae,
        "all_features_mae_day": all_features_mae,
        "prefix_mae_day": prefix_mae_day,
        "prefix_mae_log": prefix_mae_log,
    }


def run_feature_selection(DR, FS, train_split, val_split, cat_id):
    print("[Original:Table4] FeatureSel.LightGBM start", flush=True)
    original_lightgbm = FS.LightGBM
    original_lgbm_regressor = FS.lgb.LGBMRegressor
    original_lgbm_classifier = FS.lgb.LGBMClassifier

    class _FreshFitLGBM:
        def __init__(self, estimator_cls, *args, **kwargs):
            self._estimator_cls = estimator_cls
            self._args = args
            self._kwargs = kwargs
            self._model = None

        def fit(self, *args, **kwargs):
            self._model = self._estimator_cls(*self._args, **self._kwargs)
            return self._model.fit(*args, **kwargs)

        def predict(self, *args, **kwargs):
            return self._model.predict(*args, **kwargs)

        @property
        def feature_importances_(self):
            return self._model.feature_importances_

        def __getattr__(self, name):
            if self._model is None:
                raise AttributeError(name)
            return getattr(self._model, name)

    class _FreshFitRegressor(_FreshFitLGBM):
        def __init__(self, *args, **kwargs):
            super().__init__(original_lgbm_regressor, *args, **kwargs)

    class _FreshFitClassifier(_FreshFitLGBM):
        def __init__(self, *args, **kwargs):
            super().__init__(original_lgbm_classifier, *args, **kwargs)

    if len(inspect.signature(original_lightgbm).parameters) == 5:
        print("[Compat] FeatureSel.LightGBM has 5 params; injecting val_split as Test1 (matches original IPF.py behavior)", flush=True)
        def _official_lightgbm(train, val, header, categorical_ids):
            return original_lightgbm(train, val, val, header, categorical_ids)

        FS.LightGBM = _official_lightgbm
    FS.lgb.LGBMRegressor = _FreshFitRegressor
    FS.lgb.LGBMClassifier = _FreshFitClassifier
    capture = CaptureTee(sys.stdout)
    with redirect_stdout(capture):
        try:
            fr = FS.LightGBM(train_split, val_split, DR.header, cat_id)
            print(f"[Original:Table3] NewMAE={fr[0]}", flush=True)
            print(f"[Original:Table4] FeatureSel.LightGBM done | FR={fr}", flush=True)

            state = [j for i, j in zip(DR.State, range(len(DR.State))) if i == 2 or i == 4]
            print("[Original:Table4] PrefixLightGBM start", flush=True)
            pr = FS.PrefixLightGBM(train_split, val_split, val_split, DR.header, state, cat_id, fr)
            print(f"[Original:Table4] PrefixLightGBM done | PR={pr}", flush=True)
        finally:
            FS.LightGBM = original_lightgbm
            FS.lgb.LGBMRegressor = original_lgbm_regressor
            FS.lgb.LGBMClassifier = original_lgbm_classifier

    table3_metrics = parse_table3_capture(capture.getvalue())
    return fr, pr, table3_metrics


def build_datafr(DR, P, FS, w2v, train_split, val_split, cat_id, attribute, output_dir, eventlog, embedding_scope="original_all_data", run_tag=None):
    fr, pr, table3_metrics = run_feature_selection(DR, FS, train_split, val_split, cat_id)

    cbow_source = DR.AllData if embedding_scope == "original_all_data" else DR.Train
    print(f"[Original:Table4] CBOW activity embedding start | scope={embedding_scope} ({len(cbow_source)} traces)", flush=True)
    train_xa, train_ya = P.cutPrefixBy(cbow_source, [0], label=-3, batchSize=20, LEN=3)
    emb_a, acc_e = w2v.word2vec(train_xa, train_ya, DR.ConvertReflact)
    datafr = {
        "0": emb_a.detach().numpy(),
        "name": fr[1],
        "index": [fr[2]],
        "state": [[DR.State[i] for i in fr[2]]],
        "result": fr[0],
        "prefix": pr,
        "ACCE": acc_e,
    }

    categorical_dims = {}
    for i in range(1, len(DR.Train[0][0]) - 3):
        if i + 3 in attribute:
            olen = len(DR.ConvertReflact[attribute.index(i + 3)])
            eim = 4
            while olen > 16:
                olen /= 4
                eim += 4
            emb_s = nn.Embedding(len(DR.ConvertReflact[attribute.index(i + 3)]), eim)
            if i in datafr["index"][0]:
                datafr[str(i)] = emb_s.weight.detach().numpy()
                categorical_dims[str(i)] = eim

    os.makedirs(os.path.join(output_dir, "intermediate"), exist_ok=True)
    mat_tag = run_tag if run_tag else eventlog
    mat_path = os.path.join(output_dir, "intermediate", f"preFR_{mat_tag}.mat")
    savemat(mat_path, datafr)
    loaded = loadmat(mat_path)
    return loaded, fr, pr, categorical_dims, mat_path, table3_metrics


def build_datafr_all_features(DR, P, w2v, attribute, output_dir, eventlog, datafr_mode="mat", embedding_scope="original_all_data", run_tag=None):
    all_feature_indices = list(range(0, len(DR.Train[0][0]) - 3))
    cbow_source = DR.AllData if embedding_scope == "original_all_data" else DR.Train
    print(f"[Diagnostic] Build all-feature diagnostic dataFR start | scope={embedding_scope} ({len(cbow_source)} traces)", flush=True)
    train_xa, train_ya = P.cutPrefixBy(cbow_source, [0], label=-3, batchSize=20, LEN=3)
    emb_a, acc_e = w2v.word2vec(train_xa, train_ya, DR.ConvertReflact)
    datafr_all = {
        "0": emb_a.detach().numpy(),
        "name": [DR.header[i] for i in all_feature_indices],
        "index": [all_feature_indices],
        "state": [[DR.State[i] for i in all_feature_indices]],
        "result": "",
        "prefix": [],
        "ACCE": acc_e,
    }

    categorical_dims = {}
    for i in range(1, len(DR.Train[0][0]) - 3):
        if i + 3 in attribute:
            olen = len(DR.ConvertReflact[attribute.index(i + 3)])
            eim = 4
            while olen > 16:
                olen /= 4
                eim += 4
            emb_s = nn.Embedding(len(DR.ConvertReflact[attribute.index(i + 3)]), eim)
            if i in datafr_all["index"][0]:
                datafr_all[str(i)] = emb_s.weight.detach().numpy()
                categorical_dims[str(i)] = eim

    if datafr_mode == "direct":
        print(
            f"[Diagnostic] Build all-feature diagnostic dataFR done | features={len(all_feature_indices)}",
            flush=True,
        )
        return datafr_all, all_feature_indices, categorical_dims
    if datafr_mode != "mat":
        raise ValueError(f"Unsupported datafr_mode: {datafr_mode}")

    os.makedirs(os.path.join(output_dir, "intermediate"), exist_ok=True)
    mat_tag = run_tag if run_tag else eventlog
    mat_path = os.path.join(output_dir, "intermediate", f"preFR_all_{mat_tag}.mat")
    savemat(mat_path, datafr_all)
    loaded = loadmat(mat_path)
    print(
        f"[Diagnostic] Build all-feature diagnostic dataFR done | features={len(all_feature_indices)}",
        flush=True,
    )
    return loaded, all_feature_indices, categorical_dims


def build_datafr_activity_only(DR, P, w2v, output_dir, eventlog, datafr_mode="mat", embedding_scope="original_all_data", run_tag=None):
    feature_indices = [0]
    cbow_source = DR.AllData if embedding_scope == "original_all_data" else DR.Train
    print(f"[Original:Table4] Build activity-only baseline dataFR start | scope={embedding_scope} ({len(cbow_source)} traces)", flush=True)
    train_xa, train_ya = P.cutPrefixBy(cbow_source, feature_indices, label=-3, batchSize=20, LEN=3)
    emb_a, acc_e = w2v.word2vec(train_xa, train_ya, DR.ConvertReflact)
    datafr_activity = {
        "0": emb_a.detach().numpy(),
        "name": [DR.header[0]],
        "index": [feature_indices],
        "state": [[DR.State[0]]],
        "result": "",
        "prefix": [],
        "ACCE": acc_e,
    }

    if datafr_mode == "direct":
        print("[Original:Table4] Build activity-only baseline dataFR done | features=1", flush=True)
        return datafr_activity, feature_indices, {}
    if datafr_mode != "mat":
        raise ValueError(f"Unsupported datafr_mode: {datafr_mode}")

    os.makedirs(os.path.join(output_dir, "intermediate"), exist_ok=True)
    mat_tag = run_tag if run_tag else eventlog
    mat_path = os.path.join(output_dir, "intermediate", f"preFR_activity_{mat_tag}.mat")
    savemat(mat_path, datafr_activity)
    loaded = loadmat(mat_path)
    print("[Original:Table4] Build activity-only baseline dataFR done | features=1", flush=True)
    return loaded, feature_indices, {}


def print_activity_only_warning():
    print(
        "[Warning] LSTM_ActivityOnly_Approx is an activity-only approximation. Main/IPF.py uses "
        "dataFR['index'][-1] after FeatureSel.LightGBM/PrefixLightGBM for its NoFill LSTM path, "
        "not P.NoFill(..., [0], ...).",
        flush=True,
    )
    print(
        "[Warning] Transformer_ActivityOnly_Approx is an approximation and is not confirmed to be PT [21].",
        flush=True,
    )


def run_paper_lstm_baseline(DR, P, M, args, datafr_activity, feature_activity):
    print_activity_only_warning()
    print("[Original:Table4] Train LSTM ActivityOnly_Approx | P.NoFill + M.trian(... input_size=0)", flush=True)
    train_batch = P.NoFill(DR.Train.copy(), feature_activity, -1, args.batch_size)
    test_x, test_y = P.changeLen(DR.Test.copy(), feature_activity, -1, 1)
    _model, metric = M.trian(
        train_batch, test_x, test_y, args.epochs, 2, 0, 32, 1, "rnn", datafr_activity, isEarly=0
    )
    return "LSTM_ActivityOnly_Approx", metric


def run_paper_transformer_baseline(DR, P, M, args, datafr_activity, feature_activity, max_case_length):
    print_activity_only_warning()
    print("[Original:Table4] Train Transformer ActivityOnly_Approx | P.NoFill + M.trianT(... method='tran')", flush=True)
    train_batch = P.NoFill(DR.Train.copy(), feature_activity, -1, args.batch_size)
    test_x, test_y = P.changeLen(DR.Test.copy(), feature_activity, -1, 1)
    _model, metric, _count = M.trianT(
        train_batch, test_x, test_y, args.epochs, 2, max_case_length, "tran", datafr_activity
    )
    return "Transformer_ActivityOnly_Approx", metric


def metric_value(metric):
    if isinstance(metric, (list, tuple, np.ndarray)):
        return float(np.asarray(metric).reshape(-1)[0])
    return float(metric)


def package_metric_row(config, model_name, metric, args, raw_path, raw_md5, raw_file_size, split_info, fr, pr, categorical_dims, cbow_dim, max_case_length, elapsed, log_path):
    dataset = config["dataset"]
    ref = PAPER_REFERENCE.get(dataset, {}).get(model_name, "") if args.split_mode == "divi" else ""
    mae_day = metric_value(metric)
    selected_indices = [int(x) for x in np.asarray(fr[2]).reshape(-1).tolist()]
    selected_features = [str(x) for x in np.asarray(fr[1]).reshape(-1).tolist()]
    if isinstance(pr, str):
        prefix_features = pr
    else:
        prefix_features = ";".join(str(int(x)) for x in np.asarray(pr).reshape(-1).tolist()) if pr is not None else ""
    versions = dependency_versions()
    PAPER_BASELINE_MODELS = {"LSTM_FPM", "Transformer_FPM"}
    is_exact = (
        model_name in PAPER_BASELINE_MODELS
        and args.embedding_scope == "original_all_data"
        and args.split_mode == "divi"
        and args.selector_seed == 20
        and args.max_traces == 0
    )
    run_type = "full" if args.max_traces == 0 else "smoke_test"
    baseline_note = ""
    if "ActivityOnly_Approx" in model_name:
        baseline_note = "approximation: activity-only features, not exact paper baseline"
    elif "AllFeatures_Diagnostic" in model_name:
        baseline_note = "diagnostic: all input features without FPM selection"
    elif args.embedding_scope == "train_only":
        baseline_note = "clean_ml: CBOW trained on DR.Train only (no data leakage)"
    elif args.max_traces > 0:
        baseline_note = f"smoke_test: max_traces={args.max_traces}"
    return {
        "dataset": dataset,
        "eventlog": config["eventlog"],
        "experiment_mode": split_info.get("experiment_mode", args.experiment_mode),
        "model": model_name,
        "status": "OK",
        "error": "",
        "mae_day": mae_day,
        "paper_reference_mae_day": ref,
        "paper_delta_mae_day": (mae_day - ref) if isinstance(ref, (int, float)) else "",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "selector_seed": args.selector_seed,
        "split_mode": args.split_mode,
        "max_traces": args.max_traces,
        "embedding_scope": args.embedding_scope,
        "run_type": run_type,
        "is_exact_paper_baseline": str(is_exact).lower(),
        "baseline_note": baseline_note,
        "raw_file": raw_path,
        "raw_md5": raw_md5,
        "raw_file_size": raw_file_size,
        "trace_count": split_info["trace_count"],
        "train_trace_count": split_info["train_trace_count"],
        "test_trace_count": split_info["test_trace_count"],
        "selected_feature_count": len(selected_indices),
        "selected_features": ";".join(selected_features),
        "selected_feature_indices": ";".join(str(x) for x in selected_indices),
        "prefix_features": prefix_features,
        "cbow_dim": cbow_dim,
        "categorical_embedding_dims": ";".join(f"{k}:{v}" for k, v in sorted(categorical_dims.items())),
        "max_case_length": max_case_length,
        "python_version": versions["python_version"],
        "numpy_version": versions["numpy_version"],
        "scipy_version": versions["scipy_version"],
        "sklearn_version": versions["sklearn_version"],
        "lightgbm_version": versions["lightgbm_version"],
        "torch_version": versions["torch_version"],
        "pandas_version": versions["pandas_version"],
        "elapsed": elapsed,
        "log": log_path,
        "source_repo": SOURCE_REPO,
    }


def table3_diagnostic_row(config, args, raw_path, raw_md5, raw_file_size, split_info, max_case_length, fr, pr, table3_metrics, log_path):
    selected_indices = [int(x) for x in np.asarray(fr[2]).reshape(-1).tolist()]
    selected_features = [str(x) for x in np.asarray(fr[1]).reshape(-1).tolist()]
    prefix_features = ";".join(str(int(x)) for x in np.asarray(pr).reshape(-1).tolist()) if pr is not None else ""
    versions = dependency_versions()
    return {
        "dataset": config["dataset"],
        "eventlog": config["eventlog"],
        "experiment_mode": split_info.get("experiment_mode", args.experiment_mode),
        "split_mode": args.split_mode,
        "seed": args.seed,
        "selector_seed": args.selector_seed,
        "embedding_scope": args.embedding_scope,
        "run_type": "full" if args.max_traces == 0 else "smoke_test",
        "max_traces": args.max_traces,
        "raw_file": raw_path,
        "raw_md5": raw_md5,
        "raw_file_size": raw_file_size,
        "trace_count": split_info["trace_count"],
        "train_trace_count": split_info["train_trace_count"],
        "test_trace_count": split_info["test_trace_count"],
        "max_case_length": max_case_length,
        "python_version": versions["python_version"],
        "numpy_version": versions["numpy_version"],
        "scipy_version": versions["scipy_version"],
        "sklearn_version": versions["sklearn_version"],
        "lightgbm_version": versions["lightgbm_version"],
        "torch_version": versions["torch_version"],
        "pandas_version": versions["pandas_version"],
        "activity_mae_day": table3_metrics.get("activity_mae_day", ""),
        "all_features_mae_day": table3_metrics.get("all_features_mae_day", ""),
        "efc_mae_day": fr[0],
        "efc_feature_count": len(selected_indices),
        "efc_features": ";".join(selected_features),
        "efc_indices": ";".join(str(x) for x in selected_indices),
        "prefix_mae_day": table3_metrics.get("prefix_mae_day", ""),
        "prefix_mae_log": table3_metrics.get("prefix_mae_log", ""),
        "prefix_features": prefix_features,
        "log": log_path,
    }


def selected_feature_rows(config, args, fr, states):
    rows = []
    selected_indices = [int(x) for x in np.asarray(fr[2]).reshape(-1).tolist()]
    selected_features = [str(x) for x in np.asarray(fr[1]).reshape(-1).tolist()]
    for rank, (feature_idx, feature) in enumerate(zip(selected_indices, selected_features), start=1):
        rows.append({
            "dataset": config["dataset"],
            "eventlog": config["eventlog"],
            "experiment_mode": args.experiment_mode,
            "split_mode": args.split_mode,
            "seed": args.seed,
            "selector_seed": args.selector_seed,
            "embedding_scope": args.embedding_scope,
            "rank": rank,
            "feature_index": feature_idx,
            "feature": feature,
            "state": states[feature_idx],
        })
    return rows


def summarize_seed_results(rows):
    groups = {}
    for row in rows:
        if row.get("status") != "OK":
            continue
        key = (
            row.get("dataset", ""),
            row.get("model", ""),
            row.get("experiment_mode", split_experiment_mode(row.get("split_mode", ""))),
            row.get("split_mode", ""),
            row.get("selector_seed", ""),
            row.get("embedding_scope", ""),
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for (dataset, model, experiment_mode, split_mode, selector_seed, embedding_scope), items in groups.items():
        values = []
        ref = ""
        for item in items:
            try:
                values.append(float(item.get("mae_day", "")))
            except Exception:
                pass
            ref = item.get("paper_reference_mae_day", ref)

        if not values:
            continue

        arr = np.array(values, dtype=float)
        mean_mae = float(arr.mean())
        std_mae = float(arr.std()) if len(arr) > 1 else 0.0
        min_mae = float(arr.min())
        max_mae = float(arr.max())

        try:
            ref_float = float(ref)
            delta_mean = mean_mae - ref_float
        except Exception:
            delta_mean = ""

        summary.append({
            "dataset": dataset,
            "model": model,
            "experiment_mode": experiment_mode,
            "split_mode": split_mode,
            "selector_seed": selector_seed,
            "embedding_scope": embedding_scope,
            "n": len(values),
            "mean_mae_day": mean_mae,
            "std_mae_day": std_mae,
            "min_mae_day": min_mae,
            "max_mae_day": max_mae,
            "paper_reference_mae_day": ref,
            "mean_delta_day": delta_mean,
        })

    return summary


def _float_values(items, field):
    values = []
    for item in items:
        try:
            value = item.get(field, "")
            if value != "":
                values.append(float(value))
        except Exception:
            pass
    return values


def _summary_stat(items, field, suffix):
    values = _float_values(items, field)
    if not values:
        return {
            f"{field}_{suffix}mean": "",
            f"{field}_{suffix}min": "",
            f"{field}_{suffix}max": "",
        }
    arr = np.array(values, dtype=float)
    return {
        f"{field}_{suffix}mean": float(arr.mean()),
        f"{field}_{suffix}min": float(arr.min()),
        f"{field}_{suffix}max": float(arr.max()),
    }


def summarize_table3_diagnostics(rows):
    groups = {}
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("eventlog", ""),
            row.get("experiment_mode", split_experiment_mode(row.get("split_mode", ""))),
            row.get("split_mode", ""),
            row.get("selector_seed", ""),
            row.get("embedding_scope", ""),
            row.get("run_type", "full"),
            row.get("max_traces", 0),
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for (dataset, eventlog, experiment_mode, split_mode, selector_seed, embedding_scope, run_type, max_traces), items in groups.items():
        prefix_features = sorted({
            str(item.get("prefix_features", ""))
            for item in items
            if str(item.get("prefix_features", ""))
        })
        out = {
            "dataset": dataset,
            "eventlog": eventlog,
            "experiment_mode": experiment_mode,
            "split_mode": split_mode,
            "selector_seed": selector_seed,
            "embedding_scope": embedding_scope,
            "run_type": run_type,
            "max_traces": max_traces,
            "n": len(items),
            "prefix_features": " | ".join(prefix_features),
        }
        for field in [
            "activity_mae_day",
            "all_features_mae_day",
            "efc_mae_day",
            "efc_feature_count",
            "prefix_mae_day",
        ]:
            values = _float_values(items, field)
            if values:
                arr = np.array(values, dtype=float)
                out[field] = float(arr.mean())
                out[f"{field}_min"] = float(arr.min())
                out[f"{field}_max"] = float(arr.max())
            else:
                out[field] = ""
                out[f"{field}_min"] = ""
                out[f"{field}_max"] = ""

        summary.append(out)
    return summary


def print_baseline_path_note(workspace_dir):
    ipf_path = os.path.join(workspace_dir, "Main", "IPF.py")
    if not os.path.exists(ipf_path):
        print("[BaselinePath] Main/IPF.py not found; activity-only baseline remains an approximation.", flush=True)
        return
    print(
        "[BaselinePath] Main/IPF.py train() uses P.NoFill(feature), P.changeLen(feature), "
        "M.trian(... input_size=-2, method='rnn', dataFR) and M.trianT(... method='tran', dataFR). "
        "In its main flow, both NoFill/LSTM and Transformer are called with dataFR['index'][-1] "
        "after FS.LightGBM/PrefixLightGBM, not with feature=[0].",
        flush=True,
    )
    print(
        "[BaselinePath] The repository's PT-like path appears only as commented M0.trianPT/testPT code "
        "in Main/multiFea.py, so this runner does not rename Transformer_ActivityOnly to PT.",
        flush=True,
    )


def print_seed_sweep_recommendation():
    print("\nRecommended multi-seed command:", flush=True)
    print(
        "python run_table4_fpm_original.py --datasets BPIC2015_1 "
        "--models LSTM_ActivityOnly Transformer_ActivityOnly LSTM_FPM Transformer_FPM "
        "--epochs 200 --batch-size 100 --split-mode divi --selector-seed 20 "
        "--seed-sweep 20,42,80,123 --fresh",
        flush=True,
    )


def print_recommended_commands():
    print("\nRecommended commands:", flush=True)
    print(
        "\nStrict main reproduction Table3-only (original_all_data):\n"
        "python run_table4_fpm_original.py \\\n"
        "  --run-all-paper \\\n"
        "  --models LSTM_FPM \\\n"
        "  --table3-only \\\n"
        "  --experiment-mode main \\\n"
        "  --embedding-scope original_all_data \\\n"
        "  --seed 80 \\\n"
        "  --selector-seed 20 \\\n"
        "  --fresh",
        flush=True,
    )
    print(
        "\nStrict main reproduction full Table4 (original_all_data):\n"
        "python run_table4_fpm_original.py \\\n"
        "  --run-all-paper \\\n"
        "  --models LSTM_FPM Transformer_FPM \\\n"
        "  --epochs 200 \\\n"
        "  --batch-size 100 \\\n"
        "  --experiment-mode main \\\n"
        "  --embedding-scope original_all_data \\\n"
        "  --seed-sweep 42,80 \\\n"
        "  --selector-seed 20 \\\n"
        "  --fresh",
        flush=True,
    )
    print(
        "\nClean ML reproduction (train_only, no data leakage):\n"
        "python run_table4_fpm_original.py \\\n"
        "  --run-all-paper \\\n"
        "  --models LSTM_FPM Transformer_FPM \\\n"
        "  --epochs 200 \\\n"
        "  --batch-size 100 \\\n"
        "  --experiment-mode main \\\n"
        "  --embedding-scope train_only \\\n"
        "  --seed-sweep 42,80 \\\n"
        "  --selector-seed 20 \\\n"
        "  --fresh",
        flush=True,
    )
    print(
        "\nSupplemental sensitivity analysis Table3-only:\n"
        "python run_table4_fpm_original.py \\\n"
        "  --run-all-paper \\\n"
        "  --models LSTM_FPM \\\n"
        "  --table3-only \\\n"
        "  --experiment-mode sensitivity \\\n"
        "  --embedding-scope original_all_data \\\n"
        "  --seed 80 \\\n"
        "  --selector-seed 20 \\\n"
        "  --fresh",
        flush=True,
    )
    print(
        "\nselector_seed diagnostic experiment:\n"
        "python run_table4_fpm_original.py \\\n"
        "  --run-all-paper \\\n"
        "  --models LSTM_FPM \\\n"
        "  --table3-only \\\n"
        "  --experiment-mode main \\\n"
        "  --embedding-scope original_all_data \\\n"
        "  --seed 80 \\\n"
        "  --selector-seed-sweep 1,5,10,20,42,80,100,123 \\\n"
        "  --fresh",
        flush=True,
    )


def run_dataset(config, args, models, workspace_dir, modules):
    DR, DD, LA, LC, P, FS, w2v, M = modules
    raw_path = os.path.join(workspace_dir, "dataset", config["raw_file"])
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset file not found: {raw_path}")
    raw_md5 = file_md5(raw_path)
    raw_file_size = file_size(raw_path)

    output_dir = os.path.join(workspace_dir, args.output_dir)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(
        logs_dir,
        f"{config['dataset']}_{args.split_mode}_seed{args.seed}_selector{args.selector_seed}_{args.embedding_scope}_mt{args.max_traces}.txt",
    )
    start = time.time()
    metric_rows = []
    selected_rows = []
    table3_rows = []

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"Start Time: {datetime.now().isoformat()}", flush=True)
            print(f"Source Repo: {SOURCE_REPO}", flush=True)
            print("Original Path: Main/IPF.py -> LogC -> DiviData -> LightGBM -> PrefixLightGBM -> word2vec -> trian/trianT", flush=True)
            print(f"Dataset: {config['dataset']} | raw={raw_path}", flush=True)
            print(
                f"Epochs={args.epochs} | batchSize={args.batch_size} | "
                f"seed={args.seed} | selector_seed={args.selector_seed} | split_mode={args.split_mode}",
                flush=True,
            )
            print(f"Split mode: {args.split_mode}", flush=True)
            print(f"Selector seed: {args.selector_seed}", flush=True)
            print(f"Raw MD5: {raw_md5}", flush=True)
            print(f"Raw file size: {raw_file_size}", flush=True)
            print(
                "[Note] LSTM_ActivityOnly_Approx and Transformer_ActivityOnly_Approx are activity-only approximations "
                "unless the original IPF.py baseline path confirms otherwise.",
                flush=True,
            )
            if "Transformer_ActivityOnly_Approx" in models:
                print(
                    "[Warning] Transformer_ActivityOnly_Approx is an approximation and is not confirmed to be PT [21].",
                    flush=True,
                )
            print_baseline_path_note(workspace_dir)
            set_repro_seed(args.seed)

            build_data_record(DR, DD, LC, config, raw_path, max_traces=args.max_traces, split_mode=args.split_mode)
            max_case_length = LA.GeneralIndicator(DR, DR.AllData)
            run_tag = f"{config['eventlog']}_{args.split_mode}_{args.experiment_mode}_seed{args.seed}_sel{args.selector_seed}_{args.embedding_scope}"
            split_info = {
                "experiment_mode": args.experiment_mode,
                "split_mode": args.split_mode,
                "trace_count": len(DR.AllData),
                "train_trace_count": len(DR.Train),
                "test_trace_count": len(DR.Test),
            }
            print(f"[Original:Table4] Split | {split_info} | max_case_length={max_case_length}", flush=True)

            train_split, val_split = train_test_split(
                DR.Train,
                test_size=0.2,
                random_state=args.selector_seed,
            )
            cat_id = [config["attribute"][i] - 3 for i in range(len(config["attribute"]))]
            if args.table3_only:
                fr, pr, table3_metrics = run_feature_selection(DR, FS, train_split, val_split, cat_id)
                table3_rows.append(table3_diagnostic_row(
                    config, args, raw_path, raw_md5, raw_file_size, split_info, max_case_length, fr, pr, table3_metrics, log_path
                ))
                selected_rows = selected_feature_rows(config, args, fr, DR.State)
                print("[Original:Table3] --table3-only enabled: skip CBOW, P.NoFill, P.changeLen, M.trian and M.trianT.", flush=True)
                return metric_rows, selected_rows, table3_rows

            datafr, fr, pr, categorical_dims, _mat_path, table3_metrics = build_datafr(
                DR, P, FS, w2v, train_split, val_split, cat_id, config["attribute"], output_dir, config["eventlog"],
                embedding_scope=args.embedding_scope, run_tag=run_tag,
            )
            table3_rows.append(table3_diagnostic_row(
                config, args, raw_path, raw_md5, raw_file_size, split_info, max_case_length, fr, pr, table3_metrics, log_path
            ))
            cbow_dim = int(np.asarray(datafr["0"]).shape[1]) if "0" in datafr else ""

            feature = datafr["index"][-1]
            train_batch = P.NoFill(DR.Train.copy(), feature, -1, args.batch_size)
            test_x, test_y = P.changeLen(DR.Test.copy(), feature, -1, 1)

            if "LSTM_FPM" in models:
                print("[Original:Table4] Train LSTM FPM | original trian(... method='rnn', input_size=-2)", flush=True)
                _model, metric = M.trian(
                    train_batch, test_x, test_y, args.epochs, 2, -2, 32, 1, "rnn", datafr, isEarly=0
                )
                metric_rows.append(package_metric_row(
                    config, "LSTM_FPM", metric, args, raw_path, raw_md5, raw_file_size, split_info, fr, pr,
                    categorical_dims, cbow_dim, max_case_length, time.time() - start, log_path
                ))

            if "Transformer_FPM" in models:
                print("[Original:Table4] Train Transformer FPM | original trianT(... method='tran')", flush=True)
                _model, metric, _count = M.trianT(
                    train_batch, test_x, test_y, args.epochs, 2, max_case_length, "tran", datafr
                )
                metric_rows.append(package_metric_row(
                    config, "Transformer_FPM", metric, args, raw_path, raw_md5, raw_file_size, split_info, fr, pr,
                    categorical_dims, cbow_dim, max_case_length, time.time() - start, log_path
                ))

            if "LSTM_ActivityOnly_Approx" in models or "Transformer_ActivityOnly_Approx" in models:
                set_repro_seed(args.seed)
                datafr_activity, activity_feature_indices, categorical_dims_activity = build_datafr_activity_only(
                    DR, P, w2v, output_dir, config["eventlog"], datafr_mode="mat",
                    embedding_scope=args.embedding_scope, run_tag=run_tag,
                )
                cbow_dim_activity = int(np.asarray(datafr_activity["0"]).shape[1]) if "0" in datafr_activity else ""
                feature_activity = datafr_activity["index"][-1]
                fr_activity = [
                    "",
                    [DR.header[i] for i in activity_feature_indices],
                    activity_feature_indices,
                ]

                if "LSTM_ActivityOnly_Approx" in models:
                    model_name, metric = run_paper_lstm_baseline(
                        DR, P, M, args, datafr_activity, feature_activity
                    )
                    metric_rows.append(package_metric_row(
                        config, model_name, metric, args, raw_path, raw_md5, raw_file_size, split_info, fr_activity, "ActivityOnly",
                        categorical_dims_activity, cbow_dim_activity, max_case_length, time.time() - start, log_path
                    ))

                if "Transformer_ActivityOnly_Approx" in models:
                    model_name, metric = run_paper_transformer_baseline(
                        DR, P, M, args, datafr_activity, feature_activity, max_case_length
                    )
                    metric_rows.append(package_metric_row(
                        config, model_name, metric, args, raw_path, raw_md5, raw_file_size, split_info, fr_activity, "ActivityOnly",
                        categorical_dims_activity, cbow_dim_activity, max_case_length, time.time() - start, log_path
                    ))

            if "LSTM_AllFeatures_Diagnostic" in models or "Transformer_AllFeatures_Diagnostic" in models:
                set_repro_seed(args.seed)
                datafr_all, all_feature_indices, categorical_dims_all = build_datafr_all_features(
                    DR, P, w2v, config["attribute"], output_dir, config["eventlog"], datafr_mode="mat",
                    embedding_scope=args.embedding_scope, run_tag=run_tag,
                )
                cbow_dim_all = int(np.asarray(datafr_all["0"]).shape[1]) if "0" in datafr_all else ""
                feature_all = datafr_all["index"][-1]
                train_batch_all = P.NoFill(DR.Train.copy(), feature_all, -1, args.batch_size)
                test_x_all, test_y_all = P.changeLen(DR.Test.copy(), feature_all, -1, 1)
                fr_all = [
                    "",
                    [DR.header[i] for i in all_feature_indices],
                    all_feature_indices,
                ]

                if "LSTM_AllFeatures_Diagnostic" in models:
                    print("[Diagnostic] Train LSTM AllFeatures | all original input features", flush=True)
                    _model, metric = M.trian(
                        train_batch_all, test_x_all, test_y_all, args.epochs, 2, -2, 32, 1, "rnn", datafr_all, isEarly=0
                    )
                    metric_rows.append(package_metric_row(
                        config, "LSTM_AllFeatures_Diagnostic", metric, args, raw_path, raw_md5, raw_file_size, split_info, fr_all, "AllFeatures_Diagnostic",
                        categorical_dims_all, cbow_dim_all, max_case_length, time.time() - start, log_path
                    ))

                if "Transformer_AllFeatures_Diagnostic" in models:
                    print("[Diagnostic] Train Transformer AllFeatures | all original input features", flush=True)
                    _model, metric, _count = M.trianT(
                        train_batch_all, test_x_all, test_y_all, args.epochs, 2, max_case_length, "tran", datafr_all
                    )
                    metric_rows.append(package_metric_row(
                        config, "Transformer_AllFeatures_Diagnostic", metric, args, raw_path, raw_md5, raw_file_size, split_info, fr_all, "AllFeatures_Diagnostic",
                        categorical_dims_all, cbow_dim_all, max_case_length, time.time() - start, log_path
                    ))

            selected_rows = selected_feature_rows(config, args, fr, DR.State)

    return metric_rows, selected_rows, table3_rows


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def format_float(value, digits=4):
    try:
        if value == "" or value is None:
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def print_table(title, rows, columns):
    if not rows:
        print(f"\n{title}: no rows", flush=True)
        return

    str_rows = []
    for row in rows:
        item = {}
        for key, header in columns:
            value = row.get(key, "")
            if key in {
                "mae_day",
                "paper_reference_mae_day",
                "paper_delta_mae_day",
                "elapsed",
                "mean_mae_day",
                "std_mae_day",
                "min_mae_day",
                "max_mae_day",
                "mean_delta_day",
                "activity_mae_day",
                "all_features_mae_day",
                "efc_mae_day",
                "efc_feature_count",
                "prefix_mae_day",
                "activity_mae_day_mean",
                "activity_mae_day_min",
                "activity_mae_day_max",
                "all_features_mae_day_mean",
                "all_features_mae_day_min",
                "all_features_mae_day_max",
                "efc_mae_day_mean",
                "efc_mae_day_min",
                "efc_mae_day_max",
                "efc_feature_count_mean",
                "efc_feature_count_min",
                "efc_feature_count_max",
                "prefix_mae_day_mean",
                "prefix_mae_day_min",
                "prefix_mae_day_max",
                "best_efc_mae_day",
                "best_prefix_mae_day",
            }:
                value = format_float(value)
            item[key] = str(value)
        str_rows.append(item)

    widths = {}
    for key, header in columns:
        widths[key] = max(len(header), *(len(row[key]) for row in str_rows))

    print("\n" + title, flush=True)
    print("=" * (sum(widths.values()) + 3 * len(columns) + 1), flush=True)

    header_line = "| " + " | ".join(
        header.ljust(widths[key]) for key, header in columns
    ) + " |"
    sep_line = "|-" + "-|-".join(
        "-" * widths[key] for key, _ in columns
    ) + "-|"

    print(header_line, flush=True)
    print(sep_line, flush=True)

    for row in str_rows:
        print("| " + " | ".join(
            row[key].ljust(widths[key]) for key, _ in columns
        ) + " |", flush=True)


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


def main():
    args = parse_args()
    patch_numpy_legacy_aliases()
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    if args.list_datasets:
        print("Available paper Table4 datasets:")
        for name in PAPER_DATASETS:
            print(f"  {name}")
        return

    if not args.datasets and not args.paper_datasets and not args.run_all_paper and not args.default_all_paper:
        print(
            "[Info] No dataset option provided. Defaulting to BPIC2015_1. "
            "Use --run-all-paper or --paper-datasets to run all paper datasets.",
            flush=True,
        )

    paper_mode = args.paper_datasets or args.run_all_paper or (
        args.default_all_paper and not args.datasets
    )
    configs = selected_configs(args.datasets, paper_datasets=paper_mode)
    models = normalize_models(args.models)
    seeds = parse_int_list(args.seed_sweep, args.seed, "seed-sweep")
    selector_seed_sweep_enabled = bool(args.selector_seed_sweep.strip())
    if selector_seed_sweep_enabled:
        selector_seeds = parse_int_list(args.selector_seed_sweep, args.selector_seed, "selector-seed-sweep")
        print(
            "[Diagnostic] selector_seed_sweep is enabled. This is a sensitivity/diagnostic run, "
            "not the strict reproduction main run.",
            flush=True,
        )
    else:
        if args.experiment_mode in ("main", "sensitivity") and args.selector_seed != 20:
            print(
                f"[Warning] experiment-mode={args.experiment_mode} requires selector_seed=20 for paper reproduction. "
                f"Ignoring --selector-seed {args.selector_seed}.",
                flush=True,
            )
            args.selector_seed = 20
        selector_seeds = [args.selector_seed]

    if args.split_sweep:
        split_modes = ["divi", "time_year"]
        print("[Info] --split-sweep enabled. Running both divi and time_year.", flush=True)
    elif args.experiment_mode == "main":
        split_modes = ["divi"]
    elif args.experiment_mode == "sensitivity":
        split_modes = ["time_year"]
    elif args.experiment_mode == "both":
        split_modes = ["divi", "time_year"]
    else:
        raise ValueError(f"Unsupported experiment-mode: {args.experiment_mode}")
    dataset_names = [config["dataset"] for config in configs]
    output_dir = os.path.join(workspace_dir, args.output_dir)

    print(f"Selected datasets: {dataset_names}", flush=True)
    print(f"Selected models: {models}", flush=True)
    print(f"Selected seeds: {seeds}", flush=True)
    print(f"Selected selector seeds: {selector_seeds}", flush=True)
    print(f"Selected split modes: {split_modes}", flush=True)
    print(f"Embedding scope: {args.embedding_scope}", flush=True)
    print("\nRun Plan", flush=True)
    print("=" * 80, flush=True)
    print(f"Experiment mode: {args.experiment_mode}", flush=True)
    print(f"Embedding scope: {args.embedding_scope}", flush=True)
    print("Strict main setting: split=divi, selector_seed=20, embedding_scope=original_all_data", flush=True)
    print("Sensitivity setting: split=time_year, selector_seed=20", flush=True)
    print(f"Datasets: {dataset_names}", flush=True)
    print(f"Models: {models}", flush=True)
    print(f"Split modes: {split_modes}", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print(f"Selector seeds: {selector_seeds}", flush=True)
    if not selector_seed_sweep_enabled:
        print("Strict selector seed: 20", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Table3 only: {args.table3_only}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    print_dependency_versions()
    modules = import_original_modules(workspace_dir)
    os.makedirs(output_dir, exist_ok=True)

    metric_rows = []
    feature_rows = []
    table3_rows = []
    for split_mode in split_modes:
        args.split_mode = split_mode
        for selector_seed in selector_seeds:
            args.selector_seed = selector_seed
            for seed in seeds:
                args.seed = seed
                for config in configs:
                    print(
                        f"\n=== Original Table4 FPM: {config['dataset']} | split={split_mode} | seed={seed} | selector_seed={selector_seed} ===",
                        flush=True,
                    )
                    try:
                        rows, selected, table3 = run_dataset(config, args, models, workspace_dir, modules)
                        metric_rows.extend(rows)
                        feature_rows.extend(selected)
                        table3_rows.extend(table3)
                        print(
                            f"OK | dataset={config['dataset']} | split={split_mode} | seed={seed} | selector_seed={selector_seed} | rows={len(rows)}",
                            flush=True,
                        )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        log_path = os.path.join(
                            output_dir,
                            "logs",
                            f"{config['dataset']}_{args.split_mode}_seed{args.seed}_selector{args.selector_seed}_{args.embedding_scope}_mt{args.max_traces}.txt",
                        )
                        raw_path = os.path.join(workspace_dir, "dataset", config["raw_file"])
                        raw_md5 = file_md5(raw_path) if os.path.exists(raw_path) else ""
                        raw_file_size = file_size(raw_path) if os.path.exists(raw_path) else ""
                        versions = dependency_versions()
                        os.makedirs(os.path.dirname(log_path), exist_ok=True)
                        with open(log_path, "a", encoding="utf-8") as log_file:
                            log_file.write(traceback.format_exc())
                        for model in models:
                            metric_rows.append({
                                "dataset": config["dataset"],
                                "eventlog": config["eventlog"],
                                "experiment_mode": args.experiment_mode,
                                "model": model,
                                "status": "FAIL",
                                "error": error,
                                "epochs": args.epochs,
                                "batch_size": args.batch_size,
                                "seed": args.seed,
                                "selector_seed": args.selector_seed,
                                "split_mode": args.split_mode,
                                "max_traces": args.max_traces,
                                "embedding_scope": args.embedding_scope,
                                "run_type": "full" if args.max_traces == 0 else "smoke_test",
                                "is_exact_paper_baseline": "false",
                                "baseline_note": "",
                                "raw_file": raw_path,
                                "raw_md5": raw_md5,
                                "raw_file_size": raw_file_size,
                                "python_version": versions["python_version"],
                                "numpy_version": versions["numpy_version"],
                                "scipy_version": versions["scipy_version"],
                                "sklearn_version": versions["sklearn_version"],
                                "lightgbm_version": versions["lightgbm_version"],
                                "torch_version": versions["torch_version"],
                                "pandas_version": versions["pandas_version"],
                                "log": log_path,
                                "source_repo": SOURCE_REPO,
                            })
                        print(
                            f"FAIL | dataset={config['dataset']} | split={split_mode} | seed={seed} | selector_seed={selector_seed} | {error}",
                            flush=True,
                        )

    metrics_path = os.path.join(output_dir, "table4_fpm_metrics.csv")
    selected_path = os.path.join(output_dir, "table4_fpm_selected_features.csv")
    table3_path = os.path.join(output_dir, "table3_diagnostics.csv")
    table3_summary_path = os.path.join(output_dir, "table3_diagnostics_summary.csv")
    seed_summary_path = os.path.join(output_dir, "table4_fpm_seed_summary.csv")
    seed_summary = summarize_seed_results(metric_rows)
    table3_summary = summarize_table3_diagnostics(table3_rows)
    upsert_rows(
        metrics_path,
        metric_rows,
        METRIC_FIELDS,
        ["dataset", "model", "experiment_mode", "split_mode", "seed", "selector_seed", "epochs", "batch_size", "max_traces", "embedding_scope"],
        fresh=args.fresh,
    )
    upsert_rows(
        selected_path,
        feature_rows,
        SELECTED_FIELDS,
        ["dataset", "split_mode", "seed", "selector_seed", "rank", "embedding_scope", "max_traces"],
        fresh=args.fresh,
    )
    upsert_rows(
        table3_path,
        table3_rows,
        TABLE3_FIELDS,
        ["dataset", "experiment_mode", "split_mode", "seed", "selector_seed", "embedding_scope", "max_traces"],
        fresh=args.fresh,
    )
    write_csv_rows(seed_summary_path, seed_summary, SEED_SUMMARY_FIELDS)
    write_csv_rows(table3_summary_path, table3_summary, TABLE3_SUMMARY_FIELDS)
    if args.fresh:
        print("[Info] --fresh enabled: CSV rows are rebuilt for this run.", flush=True)
    print(f"\nSaved metrics: {metrics_path}", flush=True)
    print(f"Saved selected features: {selected_path}", flush=True)
    print(f"Saved Table3 diagnostics: {table3_path}", flush=True)
    print(f"Saved Table3 diagnostics summary: {table3_summary_path}", flush=True)
    print(f"Saved seed summary: {seed_summary_path}", flush=True)

    print_table(
        title="Table 4 FPM Reproduction Summary",
        rows=metric_rows,
        columns=[
            ("dataset", "Dataset"),
            ("experiment_mode", "Exp"),
            ("model", "Model"),
            ("split_mode", "Split"),
            ("seed", "Seed"),
            ("selector_seed", "SelSeed"),
            ("epochs", "Epochs"),
            ("mae_day", "MAE(Day)"),
            ("paper_reference_mae_day", "Paper"),
            ("paper_delta_mae_day", "Delta"),
            ("selected_feature_count", "Fea#"),
            ("status", "Status"),
        ],
    )

    print_table(
        title="Selected Feature Combination Summary",
        rows=metric_rows,
        columns=[
            ("dataset", "Dataset"),
            ("experiment_mode", "Exp"),
            ("split_mode", "Split"),
            ("seed", "Seed"),
            ("selector_seed", "SelSeed"),
            ("model", "Model"),
            ("selected_feature_count", "Fea#"),
            ("selected_features", "Selected Features"),
            ("prefix_features", "Prefix"),
        ],
    )

    print_table(
        title="Table 3 Diagnostics Summary",
        rows=table3_summary,
        columns=[
            ("dataset", "Dataset"),
            ("experiment_mode", "Exp"),
            ("split_mode", "Split"),
            ("selector_seed", "SelSeed"),
            ("n", "N"),
            ("activity_mae_day", "Activity"),
            ("all_features_mae_day", "All"),
            ("efc_mae_day", "EFC"),
            ("efc_mae_day_min", "EFCMin"),
            ("efc_feature_count", "Fea#"),
            ("prefix_mae_day", "PrefixMAE"),
            ("prefix_mae_day_min", "PrefixMin"),
            ("prefix_features", "Prefix"),
        ],
    )

    failed_rows = [row for row in metric_rows if row.get("status") != "OK"]
    if failed_rows:
        print_table(
            title="Failed Runs",
            rows=failed_rows,
            columns=[
                ("dataset", "Dataset"),
                ("experiment_mode", "Exp"),
                ("model", "Model"),
                ("split_mode", "Split"),
                ("seed", "Seed"),
                ("selector_seed", "SelSeed"),
                ("status", "Status"),
                ("error", "Error"),
                ("log", "Log"),
            ],
        )

    print_table(
        title="Seed Sweep Summary",
        rows=seed_summary,
        columns=[
            ("dataset", "Dataset"),
            ("experiment_mode", "Exp"),
            ("model", "Model"),
            ("split_mode", "Split"),
            ("selector_seed", "SelSeed"),
            ("n", "N"),
            ("mean_mae_day", "Mean"),
            ("std_mae_day", "Std"),
            ("min_mae_day", "Min"),
            ("max_mae_day", "Max"),
            ("paper_reference_mae_day", "Paper"),
            ("mean_delta_day", "Delta"),
        ],
    )

    print_recommended_commands()


if __name__ == "__main__":
    main()
