import os
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(r"C:\Users\Owner\Documents\Audacity\CIBuild-Transformer")
BUILDFAST_DATA_DIR = PROJECT_ROOT / "data" / "buildfast"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Try the paper-style folder first, but fall back to the buildfast root if needed
PROJECTS_DIR = BUILDFAST_DATA_DIR / "20_projects"
if not PROJECTS_DIR.exists():
    PROJECTS_DIR = BUILDFAST_DATA_DIR

# FAIR COMPARISON FILTER: use the same DL-CIBuild projects when possible
COMMON_PROJECTS_DIR = PROJECT_ROOT / "data" / "dl_cibuild"

COMMON_PROJECT_ALIASES = {
    "cloudify": ["cloudify"],
    "graylog2-server": ["graylog2-server", "graylog2", "graylog"],
    "jackrabbit-oak": ["jackrabbit-oak", "oak"],
    "jruby": ["jruby"],
    "metasploit-framework": ["metasploit-framework", "metasploit"],
    "open-build-service": ["open-build-service", "openbuildservice", "obs"],
    "openproject": ["openproject"],
    "rails": ["rails"],
    "ruby": ["ruby"],
    "sonarqube": ["sonarqube", "sonar"],
}

COMMON_PROJECTS = list(COMMON_PROJECT_ALIASES.keys())

def normalize_project_name(path_or_name):
    s = str(path_or_name).strip().lower().replace("\\", "/")
    s = s.split("/")[-1]
    if s.endswith(".csv"):
        s = s[:-4]
    return s

def canonical_project_name(path_or_name):
    s = normalize_project_name(path_or_name)
    for canon, aliases in COMMON_PROJECT_ALIASES.items():
        for alias in aliases:
            if alias in s:
                return canon
    return None


# ============================================================
# HELPERS
# ============================================================
def get_listdir(path: Path):
    return sorted(str(p) for p in Path(path).glob("*.csv"))


def fail_rate_diff(new_data: pd.DataFrame) -> pd.DataFrame:
    if "fail_ratio_pr" not in new_data.columns:
        new_data["fail_ratio_diff"] = 0.0
        return new_data

    vals = new_data["fail_ratio_pr"].fillna(0).astype(float).values
    diffs = [0.0]
    for i in range(1, len(vals)):
        prev_v = vals[i - 1]
        curr_v = vals[i]
        if prev_v == 0 and curr_v == 0:
            diffs.append(0.0)
        elif prev_v == 0 and curr_v > 0:
            diffs.append(100.0)
        elif prev_v == 0 and curr_v < 0:
            diffs.append(-100.0)
        else:
            diffs.append(100.0 * (curr_v - prev_v) / prev_v)
    new_data = new_data.copy()
    new_data["fail_ratio_diff"] = diffs
    return new_data


def f_classif_feature(X_pass: pd.DataFrame, y_pass: pd.Series, pnum: int):
    X_tmp = X_pass.copy()
    keep_extra = []

    for c in ["fail_ratio_diff", "commiter_exp"]:
        if c in X_tmp.columns:
            X_tmp = X_tmp.drop(columns=[c])
            if c == "fail_ratio_diff":
                keep_extra.append(c)

    k = min(pnum, X_tmp.shape[1]) if X_tmp.shape[1] > 0 else 0
    if k == 0:
        selected = []
    else:
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X_tmp, y_pass)
        selected = list(X_tmp.columns[selector.get_support(indices=True)])

    if "pr_status" in X_pass.columns and "pr_status" not in selected:
        selected.append("pr_status")
    for c in keep_extra:
        if c in X_pass.columns and c not in selected:
            selected.append(c)

    return selected


def chi2_feature(X_fail: pd.DataFrame, y_fail: pd.Series, fnum: int):
    X_tmp = X_fail.copy()
    keep_extra = []

    for c in ["fail_ratio_diff", "commiter_exp"]:
        if c in X_tmp.columns:
            X_tmp = X_tmp.drop(columns=[c])
            if c == "fail_ratio_diff":
                keep_extra.append(c)

    # chi2 requires nonnegative values
    X_tmp = X_tmp.fillna(0)
    X_tmp[X_tmp < 0] = 0

    k = min(fnum, X_tmp.shape[1]) if X_tmp.shape[1] > 0 else 0
    if k == 0:
        selected = []
    else:
        selector = SelectKBest(chi2, k=k)
        selector.fit(X_tmp, y_fail)
        selected = list(X_tmp.columns[selector.get_support(indices=True)])

    if "pr_status" in X_fail.columns and "pr_status" not in selected:
        selected.append("pr_status")
    for c in keep_extra:
        if c in X_fail.columns and c not in selected:
            selected.append(c)

    return selected


def feature_selection2(new_data_pass, new_data_fail, selcect_pass="f_classif", pnum=25, select_fail="chi2", fnum=30):
    y_pass = new_data_pass["now_label"]
    X_pass = new_data_pass.drop(columns=["now_label"], errors="ignore")

    y_fail = new_data_fail["now_label"]
    X_fail = new_data_fail.drop(columns=["now_label"], errors="ignore")

    if selcect_pass == "f_classif":
        new_features_pass = f_classif_feature(X_pass, y_pass, pnum)
    else:
        new_features_pass = list(X_pass.columns)

    if select_fail == "chi2":
        new_features_fail = chi2_feature(X_fail, y_fail, fnum)
    else:
        new_features_fail = list(X_fail.columns)

    return new_features_pass, new_features_fail


def save_time(y_pred_collect, y_test_collect, duration_collect):
    y_pred = np.asarray(y_pred_collect).astype(int)
    y_true = np.asarray(y_test_collect).astype(int)
    durations = np.asarray(duration_collect, dtype=float)

    # BuildFast-style practical framing:
    # predict 0 => can be skipped / treated as passing
    # benefit = correctly predicted passing builds
    # cost    = failed builds incorrectly treated as passing

    true_pass_pred_pass = (y_true == 0) & (y_pred == 0)
    true_fail_pred_pass = (y_true == 1) & (y_pred == 0)

    benefit_hours = float(durations[true_pass_pred_pass].sum())
    cost_hours = float(durations[true_fail_pred_pass].sum())
    gain_hours = float(benefit_hours - cost_hours)
    flagged_builds = int((y_pred == 1).sum())

    return {
        "benefit_hours": benefit_hours,
        "cost_hours": cost_hours,
        "gain_hours": gain_hours,
        "flagged_builds": flagged_builds,
    }


def preprocess_project(new_data: pd.DataFrame) -> pd.DataFrame:
    new_data = new_data.copy()

    # duration kept for cost-benefit before dropping from training features
    if "now_duration" not in new_data.columns:
        new_data["now_duration"] = 1.0

    new_data = fail_rate_diff(new_data)

    noeach_commit = [
        "import_change_count", "signature", "deletesignature", "addsignature",
        "methodbody", "addmethodbody", "deletemethodbody",
        "fieldchange", "addfieldchange", "deletefieldchange",
        "classchange", "addclasschange", "deleteclasschange",
        "add_import", "deleteimport", "prev_modified"
    ]
    new_data = new_data.drop(columns=[c for c in noeach_commit if c in new_data.columns], errors="ignore")

    detail_info = [
        "addmethod", "deletemethod", "cmt_add_methodcount",
        "eachsignature", "eachdeletesignature", "eachaddsignature",
        "eachmethodbody", "eachaddmethodbody", "eachdeletemethodbody"
    ]

    if "addmethod" in new_data.columns and "deletemethod" in new_data.columns:
        new_data["sum_method"] = new_data["addmethod"] + new_data["deletemethod"]
    if all(c in new_data.columns for c in ["eachsignature", "eachdeletesignature", "eachaddsignature"]):
        new_data["eachsumsignature"] = (
            new_data["eachsignature"] + new_data["eachdeletesignature"] + new_data["eachaddsignature"]
        )
    if all(c in new_data.columns for c in ["eachmethodbody", "eachaddmethodbody", "eachdeletemethodbody"]):
        new_data["eachsummethodbody"] = (
            new_data["eachmethodbody"] + new_data["eachaddmethodbody"] + new_data["eachdeletemethodbody"]
        )

    new_data = new_data.drop(columns=[c for c in detail_info if c in new_data.columns], errors="ignore")

    # Original script dropped these before modeling
    new_data = new_data.drop(
        columns=[c for c in ["gaussian", "pr_test_assert", "pr_other_error", "now_is_pr"] if c in new_data.columns],
        errors="ignore"
    )

    return new_data


def run_buildfast():
    all_buildfast_files = get_listdir(PROJECTS_DIR)

    print("PROJECTS_DIR being used:", PROJECTS_DIR)
    print("Total BuildFast CSV files found:", len(all_buildfast_files))
    print("Sample BuildFast file names:", [Path(fp).name for fp in all_buildfast_files[:10]])
    print("Common DL-CIBuild projects:", COMMON_PROJECTS)

    filtered_file_list = []
    for fp in all_buildfast_files:
        canon = canonical_project_name(fp)
        if canon is not None:
            filtered_file_list.append(fp)

    matched_buildfast_projects = sorted({
        canonical_project_name(fp)
        for fp in filtered_file_list
        if canonical_project_name(fp) is not None
    })
    print("Matched BuildFast common projects:", matched_buildfast_projects)

    if len(filtered_file_list) < 10:
        print("WARNING: Fewer than 10 BuildFast files matched the common-project aliases.")
        print("Falling back to all BuildFast CSV files.")
        file_list = all_buildfast_files
        model_name = "BuildFast_original"
    else:
        file_list = filtered_file_list
        model_name = "BuildFast_common10"

    print("BuildFast files selected:", len(file_list))
    print("Selected BuildFast file names:", [Path(fp).name for fp in file_list[:10]])

    if not file_list:
        raise FileNotFoundError(f"No CSV files found in {PROJECTS_DIR}")

    per_project_rows = []

    for file_path in file_list:
        print("=" * 80)
        print("Processing:", os.path.basename(file_path))

        new_data = pd.read_csv(file_path, low_memory=False)
        new_data = preprocess_project(new_data)

        required_cols = ["last_label", "now_label", "now_build_id", "build_id"]
        missing = [c for c in required_cols if c not in new_data.columns]
        if missing:
            print(f"Skipping {os.path.basename(file_path)} because missing required columns: {missing}")
            continue

        duration_series = new_data["now_duration"].fillna(1.0).astype(float).copy()

        # Preserve id/label columns while scaling only numeric features
        keep_cols = [c for c in ["pr_status", "last_label", "now_label", "id", "now_build_id", "build_id"] if c in new_data.columns]
        scale_cols = [c for c in new_data.columns if c not in keep_cols + ["now_duration"]]

        scaled_part = new_data[scale_cols].copy()
        scaled_part = scaled_part.fillna(0)

        scaler = preprocessing.MinMaxScaler()
        if scaled_part.shape[1] > 0:
            scaled_part = pd.DataFrame(
                scaler.fit_transform(scaled_part),
                columns=scale_cols,
                index=new_data.index
            )
        else:
            scaled_part = pd.DataFrame(index=new_data.index)

        rebuilt = pd.concat(
            [scaled_part, new_data[keep_cols].copy(), duration_series.rename("now_duration")],
            axis=1
        )

        # Use last 20% as test, same as original train.py logic
        test_size = max(1, int(np.ceil(rebuilt.shape[0] / 5)))
        test_data = rebuilt.tail(test_size).copy()
        train_data = rebuilt.drop(index=test_data.index).copy()

        y_test = test_data["now_label"].astype(int)
        x_test = test_data.drop(columns=["now_label"]).copy()

        # Pass/fail routing datasets from the original logic
        new_data_fail = rebuilt[
            (rebuilt["last_label"] == 0) | ((rebuilt["last_label"] == 1) & (rebuilt["now_label"] == 0))
        ].copy()

        new_data_pass = rebuilt[
            (rebuilt["last_label"] == 1) | ((rebuilt["last_label"] == 0) & (rebuilt["now_label"] == 0))
        ].copy()

        drop_fail = ["now_build_id", "id", "build_id"]
        drop_pass = ["now_build_id", "build_id", "log_src_files", "log_src_files_in",
                     "log_test_files", "log_test_files_in", "pr_compile_error",
                     "pr_test_exception", "id"]

        new_data_fail = new_data_fail.drop(columns=[c for c in drop_fail if c in new_data_fail.columns], errors="ignore")
        new_data_pass = new_data_pass.drop(columns=[c for c in drop_pass if c in new_data_pass.columns], errors="ignore")

        X_pass_new = train_data[
            (train_data["last_label"] == 1) | ((train_data["last_label"] == 0) & (train_data["now_label"] == 0))
        ].copy()

        X_fail_new = train_data[
            (train_data["last_label"] == 0) | ((train_data["last_label"] == 1) & (train_data["now_label"] == 0))
        ].copy()

        new_features_pass, new_features_fail = feature_selection2(
            new_data_pass,
            new_data_fail,
            selcect_pass="f_classif",
            pnum=25,
            select_fail="chi2",
            fnum=30
        )

        y_pass = X_pass_new["now_label"].astype(int)
        X_pass = X_pass_new.drop(columns=["now_label"], errors="ignore")
        X_pass = X_pass[[c for c in new_features_pass if c in X_pass.columns]]

        y_fail = X_fail_new["now_label"].astype(int)
        X_fail = X_fail_new.drop(columns=["now_label"], errors="ignore")
        X_fail = X_fail[[c for c in new_features_fail if c in X_fail.columns]]

        RF_pass = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )

        RF_fail = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )

        RF_pass.fit(X_pass, y_pass)
        RF_fail.fit(X_fail, y_fail)

        y_pred_collect = []
        predict_proba = []

        duration_collect = duration_series.loc[y_test.index].values

        for idx in x_test.index:
            test_line = x_test.loc[[idx]].copy()
            last_label_value = int(test_line["last_label"].values[0])

            if last_label_value == 0:
                use_cols = [c for c in new_features_fail if c in test_line.columns]
                test_line_model = test_line[use_cols]
                pred = RF_fail.predict(test_line_model)[0]
                prob = RF_fail.predict_proba(test_line_model)[:, 1][0]
            else:
                use_cols = [c for c in new_features_pass if c in test_line.columns]
                test_line_model = test_line[use_cols]
                pred = RF_pass.predict(test_line_model)[0]
                prob = RF_pass.predict_proba(test_line_model)[:, 1][0]

            y_pred_collect.append(int(pred))
            predict_proba.append(float(prob))

        y_test_collect = y_test.values.astype(int)
        y_pred_collect = np.asarray(y_pred_collect).astype(int)
        predict_proba = np.asarray(predict_proba).astype(float)

        time_stats = save_time(y_pred_collect, y_test_collect, duration_collect)

        project_row = {
            "project_file": os.path.basename(file_path),
            "precision_fail": float(precision_score(y_test_collect, y_pred_collect, pos_label=1, zero_division=0)),
            "recall_fail": float(recall_score(y_test_collect, y_pred_collect, pos_label=1, zero_division=0)),
            "f1_fail": float(f1_score(y_test_collect, y_pred_collect, pos_label=1, zero_division=0)),
            "accuracy": float(accuracy_score(y_test_collect, y_pred_collect)),
            "precision_macro": float(precision_score(y_test_collect, y_pred_collect, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test_collect, y_pred_collect, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test_collect, y_pred_collect, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test_collect, predict_proba)) if len(np.unique(y_test_collect)) > 1 else np.nan,
            "pr_auc": float(average_precision_score(y_test_collect, predict_proba)) if len(np.unique(y_test_collect)) > 1 else np.nan,
            **time_stats,
            "test_rows": int(len(y_test_collect)),
            "fail_rate_test": float(np.mean(y_test_collect)),
        }

        print(project_row)
        per_project_rows.append(project_row)

    if not per_project_rows:
        raise RuntimeError("No BuildFast projects were processed successfully.")

    per_project_df = pd.DataFrame(per_project_rows)
    per_project_csv = OUTPUT_DIR / "buildfast_original_per_project.csv"
    per_project_df.to_csv(per_project_csv, index=False)

    summary = {
        "model": model_name,
        "precision_fail": float(per_project_df["precision_fail"].mean()),
        "recall_fail": float(per_project_df["recall_fail"].mean()),
        "f1_fail": float(per_project_df["f1_fail"].mean()),
        "accuracy": float(per_project_df["accuracy"].mean()),
        "roc_auc": float(per_project_df["roc_auc"].mean()),
        "pr_auc": float(per_project_df["pr_auc"].mean()),
        "benefit_hours": float(per_project_df["benefit_hours"].sum()),
        "cost_hours": float(per_project_df["cost_hours"].sum()),
        "gain_hours": float(per_project_df["gain_hours"].sum()),
        "flagged_builds": int(per_project_df["flagged_builds"].sum()),
        "precision_macro": float(per_project_df["precision_macro"].mean()),
        "recall_macro": float(per_project_df["recall_macro"].mean()),
        "f1_macro": float(per_project_df["f1_macro"].mean()),
        "projects_processed": int(per_project_df["project_file"].nunique()),
    }

    summary_csv = OUTPUT_DIR / "buildfast_original_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    summary_json = OUTPUT_DIR / "buildfast_original_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("BuildFast summary saved to:")
    print(summary_csv)
    print(summary_json)
    print(per_project_csv)
    print(summary)

    return summary


if __name__ == "__main__":
    run_buildfast()