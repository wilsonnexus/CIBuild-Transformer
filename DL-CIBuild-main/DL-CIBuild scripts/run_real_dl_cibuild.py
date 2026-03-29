import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import LSTM_Tuner


PROJECT_ROOT = Path(r"C:\Users\Owner\Documents\Audacity\CIBuild-Transformer")
DATASET_DIR = PROJECT_ROOT / "data" / "dl_cibuild"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPEATS = 1
TUNER = "ga"


def online_validation_folds(dataset: pd.DataFrame):
    n = len(dataset)
    folds = []
    train_fracs = [0.5, 0.7, 0.9]   

    for frac in train_fracs:
        split_idx = max(30, int(n * frac))
        if split_idx < n:
            train_set = dataset.iloc[:split_idx].copy()
            test_set = dataset.iloc[split_idx:].copy()
            folds.append((train_set, test_set))

    return folds


def best_threshold_from_train_probs(train_probs, y_train):
    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.arange(0.05, 0.96, 0.05):
        y_pred = (train_probs >= thr).astype(int)
        f1 = f1_score(y_train, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr


def buildfast_style_cost(y_true, y_prob, durations=None, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    if durations is None:
        durations = np.ones(len(y_true), dtype=float)
    else:
        durations = np.asarray(durations, dtype=float)

    true_pass_pred_pass = (y_true == 0) & (y_pred == 0)
    true_fail_pred_pass = (y_true == 1) & (y_pred == 0)

    benefit_hours = float(durations[true_pass_pred_pass].sum())
    cost_hours = float(durations[true_fail_pred_pass].sum())
    gain_hours = float(benefit_hours - cost_hours)

    return {
        "benefit_hours": benefit_hours,
        "cost_hours": cost_hours,
        "gain_hours": gain_hours,
        "flagged_builds": int((y_pred == 1).sum()),
    }


def evaluate_project(file_path: Path):
    dataset = pd.read_csv(file_path)

    if "gh_build_started_at" in dataset.columns:
        dataset["gh_build_started_at"] = pd.to_datetime(dataset["gh_build_started_at"], errors="coerce")
        dataset = dataset.sort_values("gh_build_started_at").reset_index(drop=True)

    if "build_Failed" not in dataset.columns:
        raise ValueError(f"{file_path.name} does not contain build_Failed column.")

    if "tr_duration" in dataset.columns:
        duration_col = "tr_duration"
    elif "build_duration" in dataset.columns:
        duration_col = "build_duration"
    else:
        duration_col = None

    folds = online_validation_folds(dataset)
    rows = []

    for fold_idx, (train_set, test_set) in enumerate(folds, start=1):
        for rep in range(1, N_REPEATS + 1):
            entry_train = LSTM_Tuner.evaluate_tuner(TUNER, train_set)
            best_params = entry_train["params"]
            best_model = entry_train["model"]

            X_train, y_train = LSTM_Tuner.train_preprocess(train_set, best_params["time_step"])
            X_test, y_test = LSTM_Tuner.test_preprocess(train_set, test_set, best_params["time_step"])

            if best_model is None or len(X_train) == 0 or len(X_test) == 0:
                continue

            train_prob = best_model.predict(X_train, verbose=0).reshape(-1)
            y_train = np.asarray(y_train).astype(int).reshape(-1)
            best_thr = best_threshold_from_train_probs(train_prob, y_train)

            test_prob = best_model.predict(X_test, verbose=0).reshape(-1)
            y_test = np.asarray(y_test).astype(int).reshape(-1)
            y_pred = (test_prob >= best_thr).astype(int)

            if duration_col is not None:
                durations = test_set[duration_col].iloc[:len(y_test)].fillna(1.0).astype(float).values
            else:
                durations = np.ones(len(y_test), dtype=float)

            cb = buildfast_style_cost(y_test, test_prob, durations=durations, threshold=best_thr)

            row = {
                "project_file": file_path.name,
                "fold": fold_idx,
                "iter": rep,
                "threshold": best_thr,
                "precision_fail": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
                "recall_fail": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
                "f1_fail": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else np.nan,
                "pr_auc": float(average_precision_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else np.nan,
                **cb,
                "test_rows": int(len(y_test)),
            }
            rows.append(row)

    return rows


def main():
    all_rows = []
    for csv_file in sorted(DATASET_DIR.glob("*.csv")):
        print(f"Processing {csv_file.name}")
        try:
            rows = evaluate_project(csv_file)
            all_rows.extend(rows)
        except Exception as e:
            print(f"Skipping {csv_file.name} because of error: {e}")

    if not all_rows:
        raise RuntimeError("No DL-CIBuild rows were produced.")

    per_fold_df = pd.DataFrame(all_rows)
    per_fold_path = OUTPUT_DIR / "dl_cibuild_original_per_fold.csv"
    per_fold_df.to_csv(per_fold_path, index=False)

    # paper-style: use median over repeated stochastic runs
    summary = {
        "model": "DL-CIBuild_common10",
        "precision_fail": float(per_fold_df["precision_fail"].median()),
        "recall_fail": float(per_fold_df["recall_fail"].median()),
        "f1_fail": float(per_fold_df["f1_fail"].median()),
        "accuracy": float(per_fold_df["accuracy"].median()),
        "roc_auc": float(per_fold_df["roc_auc"].median()),
        "pr_auc": float(per_fold_df["pr_auc"].median()),
        "benefit_hours": float(per_fold_df["benefit_hours"].sum()),
        "cost_hours": float(per_fold_df["cost_hours"].sum()),
        "gain_hours": float(per_fold_df["gain_hours"].sum()),
        "flagged_builds": int(per_fold_df["flagged_builds"].sum()),
        "precision_macro": float(per_fold_df["precision_macro"].median()),
        "recall_macro": float(per_fold_df["recall_macro"].median()),
        "f1_macro": float(per_fold_df["f1_macro"].median()),
        "projects_processed": int(per_fold_df["project_file"].nunique()),
    }

    summary_df = pd.DataFrame([summary])
    summary_csv = OUTPUT_DIR / "dl_cibuild_original_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_json = OUTPUT_DIR / "dl_cibuild_original_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(per_fold_path)
    print(summary_csv)
    print(summary_json)
    print(summary)


if __name__ == "__main__":
    main()