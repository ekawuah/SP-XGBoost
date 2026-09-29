#!/usr/bin/env python3
"""
00_CLOSEOUT_ERNEST_AWUAH_20260920.py

Close-out verification for SP-XGBoost pipeline.
Run this file in your own environment. Paste back the VERDICT TABLE.

THIS SCRIPT IS AN EXEMPLAR. IT IS NOT A FORCED SOLUTION.
CHECK EVERY COLUMN NAME, FILE PATH, OBJECT NAME, SEED, AND LIBRARY VERSION
BEFORE RUNNING. MODIFY IT WHERE YOUR STUDY CONTEXT REQUIRES MODIFICATION.
YOU OWN THE FINAL VERSION. DO NOT RUN IT BLINDLY.

It produces EVIDENCE, not outcomes. It can come back with a result you did
not want. That is what makes it worth running. Report what returns.

THIS IS THE LAST ENGINEERING PASS. AFTER THIS SCRIPT RETURNS, THE NUMBERS
ARE THE NUMBERS AND THE WRITING BEGINS.
"""

# =============================================================================
# ENV DETECT
# =============================================================================
import sys
import os

try:
    import google.colab  # noqa: F401
    ENV = "colab"
except ImportError:
    ENV = "local"

print("ENVIRONMENT:", ENV)

# =============================================================================
# CONFIG — <<< FILL IN: paths and filenames from your repository >>>
# =============================================================================

# --- Repository root (auto-detected relative to script location) ---
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --- ONE data directory constant ---
DATA_DIR = os.path.join(REPO_DIR, "data", "processed")
# On Colab, if mounted, adjust if needed: "/content/drive/MyDrive/SP-XGBoost/data/processed"

# --- Data files ---
PATH_TRAIN = os.path.join(DATA_DIR, "train_binary_shap.csv")
PATH_TEST  = os.path.join(DATA_DIR, "test_binary_shap.csv")
PATH_FULL  = os.path.join(REPO_DIR, "data", "field", "SPhealth_student_data_v2.csv")

# --- Column name assertions ---
TARGET_COL     = "RISK_BINARY"        # Target column name
GPA_S1_COL     = "GPA_S1"             # GPA Semester 1 column name
STUDENT_ID_COL = "SID"                # Student identifier column

# --- Locked hyperparameters (from M12) ---
SP_XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.04133713629478509,
    "min_child_weight": 1,
    "subsample": 0.8493145117240506,
    "colsample_bytree": 0.8332908615865308,
    "gamma": 4.101525122344799,
    "reg_alpha": 0.004811499595716167,
    "reg_lambda": 0.012132398817719065,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
}

# --- The 49 SHAP-retained predictors (from M17) ---
features_json_file = os.path.join(REPO_DIR, "results", "sp_xgboost_final_shap_features.json")
if os.path.exists(features_json_file):
    import json
    with open(features_json_file, "r") as f:
        FEATURES_49 = json.load(f)
else:
    FEATURES_49 = [
        "GPA_S1", "Persists when difficult", "Takes rest breaks", "CA_AVG",
        "Manages time effectively", "ATT_RATE", "Confident in clinical skills",
        "Anxious about assessments", "EXAM_AVG", "Reviews notes within 24h",
        "CLIN_AVG", "Participates in clubs", "Sense of belonging",
        "Confident to complete prog", "LAB_AVG", "Lecturers approachable",
        "Concentration in self-study", "Schedule conflicts", "Hopeless/unmotivated",
        "Balanced diet", "Sleep difficulty", "Adequate advisory support",
        "Sleep_hrs", "Programme prepares for career", "Considered break",
        "Adequate supervision", "Sleep affects concentration", "Rotations affect performance",
        "Regular exercise", "Uses library regularly", "Burnt out",
        "Emotionally supported", "Early intervention provided", "Participates in class",
        "Self_risk_percep", "Understands content pre-exam", "ASSIGN_LATE",
        "Financial_diff", "Sets academic goals", "Self-motivates",
        "Year_study", "Completes readings", "Concerns taken seriously",
        "Employment_hrs", "Programme_ENT", "Can improve performance",
        "Seeks help when stuck", "Prepared for clinical assess", "Study_hrs_day"
    ]

# --- Canonical artifact output path ---
ARTIFACT_DIR = os.path.join(REPO_DIR, "artifacts", "canonical")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# =============================================================================
# VERSION GUARD — H4
# =============================================================================
import importlib
import numpy as np

def _ver(pkg):
    try:
        return importlib.import_module(pkg).__version__
    except Exception:
        return "NOT INSTALLED"

VERSIONS = {
    "numpy": _ver("numpy"),
    "pandas": _ver("pandas"),
    "sklearn": _ver("sklearn"),
    "xgboost": _ver("xgboost"),
    "shap": _ver("shap"),
    "scipy": _ver("scipy"),
}

print("=" * 70)
print("VERSION GUARD")
print("=" * 70)
for k, v in VERSIONS.items():
    print(f"  {k:<12}: {v}")

# Assert critical versions
assert VERSIONS["xgboost"] != "NOT INSTALLED", "xgboost not installed"
assert VERSIONS["sklearn"] != "NOT INSTALLED", "scikit-learn not installed"
assert VERSIONS["shap"] != "NOT INSTALLED", "shap not installed"

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import warnings
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
)
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# PREFLIGHT
# =============================================================================
print()
print("=" * 70)
print("PREFLIGHT")
print("=" * 70)

# Load data
df_train = pd.read_csv(PATH_TRAIN)
df_test  = pd.read_csv(PATH_TEST)

print(f"  Training rows: {len(df_train)}  (expected 468)")
print(f"  Test rows:     {len(df_test)}  (expected 117)")
print(f"  Training cols: {df_train.shape[1]}")
print(f"  Test cols:     {df_test.shape[1]}")

# Assertions
assert len(df_train) == 468, f"Expected 468 training rows, got {len(df_train)}"
assert len(df_test) == 117, f"Expected 117 test rows, got {len(df_test)}"

# Target column check
assert TARGET_COL in df_train.columns, f"Target column '{TARGET_COL}' not in training data"
assert TARGET_COL in df_test.columns, f"Target column '{TARGET_COL}' not in test data"

# Target distribution
y_train = df_train[TARGET_COL].values
y_test  = df_test[TARGET_COL].values

print(f"  Train positive rate: {y_train.mean():.4f}")
print(f"  Test positive rate:  {y_test.mean():.4f}")

assert abs(y_train.mean() - 0.4786) < 0.01, f"Train positive rate mismatch: {y_train.mean():.4f}"
assert abs(y_test.mean() - 0.4786) < 0.02, f"Test positive rate mismatch: {y_test.mean():.4f}"

# Missing value check — F3
train_missing = df_train.isnull().sum().sum()
test_missing  = df_test.isnull().sum().sum()
print(f"  Missing values (train): {train_missing}")
print(f"  Missing values (test):  {test_missing}")
assert train_missing == 0, f"Training data has {train_missing} missing values"
assert test_missing == 0, f"Test data has {test_missing} missing values"

# GPA_S1 column check
assert GPA_S1_COL in df_train.columns, f"GPA_S1 column '{GPA_S1_COL}' not found"

# Attach student identifier to df_test if not directly present in test_binary_shap
if STUDENT_ID_COL not in df_test.columns:
    raw_test_file = os.path.join(DATA_DIR, "test_binary.csv")
    if os.path.exists(raw_test_file):
        df_raw_test = pd.read_csv(raw_test_file)
        if STUDENT_ID_COL in df_raw_test.columns:
            df_test[STUDENT_ID_COL] = df_raw_test[STUDENT_ID_COL].values
    if STUDENT_ID_COL not in df_test.columns:
        df_test[STUDENT_ID_COL] = [f"SID_{i+1:03d}" for i in range(len(df_test))]

# Feature availability check
available_features = [f for f in FEATURES_49 if f in df_train.columns]
missing_features = [f for f in FEATURES_49 if f not in df_train.columns]

print(f"  Features in FEATURES_49: {len(FEATURES_49)}")
print(f"  Features present in data: {len(available_features)}")
if missing_features:
    print(f"  MISSING FEATURES: {missing_features}")
    raise RuntimeError(f"Features missing from data: {missing_features}")

# =============================================================================
# HALF 1 — VERIFY (READ-ONLY, SAFE TO RUN IN ANY ORDER)
# =============================================================================

# -----------------------------------------------------------------------------
# BLOCK V1 — DETERMINISM CONFIRM
# -----------------------------------------------------------------------------
# CLOSES            : Item 1 (partial — confirm cause)
# OPEN ITEM         : Two SP-XGBoost artifacts diverge: ROC-AUC 0.8138 vs 0.8088
#                     with identical hyperparameters, data, and random_state=42.
# ANSWERS           : Is the refit deterministic with n_jobs=1?
# EXPECTED DIRECTION: NEUTRAL — this is a diagnostic, not a fix
# RUNTIME           : ~2 minutes (2 fits × 468 rows)
# IF IT RETURNS
# SOMETHING BAD     : If the two fits still differ with n_jobs=1, the cause is
#                     not threading — investigate feature column order.
# PASTE OUTPUT WHERE: Methods M14 reproducibility section
# -----------------------------------------------------------------------------

print()
print("=" * 70)
print("BLOCK V1 — DETERMINISM CONFIRM")
print("=" * 70)

X_train = df_train[FEATURES_49].values
X_test  = df_test[FEATURES_49].values

# Fit 1: n_jobs=-1 (current setup, as in notebook)
model_a = XGBClassifier(**SP_XGB_PARAMS, n_jobs=-1)
model_a.fit(X_train, y_train)
pred_a = model_a.predict_proba(X_test)[:, 1]
auc_a = roc_auc_score(y_test, pred_a)

# Fit 2: n_jobs=1 (single-threaded)
model_b = XGBClassifier(**SP_XGB_PARAMS, n_jobs=1)
model_b.fit(X_train, y_train)
pred_b = model_b.predict_proba(X_test)[:, 1]
auc_b = roc_auc_score(y_test, pred_b)

print(f"  Fit 1 (n_jobs=-1): ROC-AUC = {auc_a:.10f}")
print(f"  Fit 2 (n_jobs=1):  ROC-AUC = {auc_b:.10f}")
print(f"  Difference:        {abs(auc_a - auc_b):.10e}")

# Predictions equality check
preds_identical = np.array_equal(pred_a, pred_b)
print(f"  Predictions identical: {preds_identical}")

# Hash the predictions for a stable identity
hash_a = hashlib.sha256(pred_a.tobytes()).hexdigest()[:16]
hash_b = hashlib.sha256(pred_b.tobytes()).hexdigest()[:16]
print(f"  Pred hash (n_jobs=-1): {hash_a}")
print(f"  Pred hash (n_jobs=1):  {hash_b}")

if not preds_identical:
    print()
    print("  DETERMINISM CONFIRMED: n_jobs > 1 causes divergent results.")
    print("  Cause: multi-threaded histogram binning (tree_method='hist').")
    print("  Next: set n_jobs=1 in all final model fits and re-run K1.")
else:
    print()
    print("  DETERMINISM CONFIRMED: n_jobs does not affect this fit.")
    print("  Cause: likely feature column order or a preprocessing artifact")
    print("  rebuilt differently. Investigate notebook 07 vs notebook 12.")

# Also test with tree_method='exact'
model_c = XGBClassifier(**SP_XGB_PARAMS, n_jobs=1, tree_method="exact")
model_c.fit(X_train, y_train)
pred_c = model_c.predict_proba(X_test)[:, 1]
auc_c = roc_auc_score(y_test, pred_c)
print(f"  Fit 3 (n_jobs=1, exact): ROC-AUC = {auc_c:.10f}")

# -----------------------------------------------------------------------------
# BLOCK V2 — PAIRED-PREDICTION CSV CONSTRUCTION VERIFY
# -----------------------------------------------------------------------------
# CLOSES            : Item 2
# OPEN ITEM         : SP-XGBoost and Random Forest both report ROC-AUC
#                     0.80884075 at seed 42, but confusion matrices differ.
#                     DeLong gives z=0.0000, p=1.0000.
# ANSWERS           : Is the identical AUC a genuine coincidence or a data-
#                     pairing bug in the paired-predictions CSV?
# EXPECTED DIRECTION: NEUTRAL
# RUNTIME           : ~1 minute
# IF IT RETURNS
# SOMETHING BAD     : If the prediction arrays are identical, the CSV
#                     construction has a bug — predictions are being copied.
# PASTE OUTPUT WHERE: R4 Statistical Comparison
# -----------------------------------------------------------------------------

print()
print("=" * 70)
print("BLOCK V2 — PAIRED-PREDICTION CSV VERIFY")
print("=" * 70)

# Paired predictions CSV path (from notebook 15 statistics)
PATH_PAIRED = os.path.join(
    REPO_DIR, "results", "notebook15_statistics",
    "sp_xgboost_vs_random_forest_seed42_paired_predictions.csv"
)

# Identify the prediction columns
COL_SP_PRED = "SP_XGBoost_Probability"
COL_RF_PRED = "Random_Forest_Probability"

# Check if the file exists; if not, skip with a clear message
if os.path.exists(PATH_PAIRED):
    df_paired = pd.read_csv(PATH_PAIRED)
    print(f"  Paired CSV loaded: {len(df_paired)} rows, {df_paired.shape[1]} columns")
    print(f"  Columns: {list(df_paired.columns)}")

    if COL_SP_PRED in df_paired.columns and COL_RF_PRED in df_paired.columns:
        sp_preds = df_paired[COL_SP_PRED].values
        rf_preds = df_paired[COL_RF_PRED].values

        # Are the raw prediction arrays identical?
        arrays_identical = np.array_equal(sp_preds, rf_preds)
        print(f"  SP vs RF prediction arrays identical: {arrays_identical}")

        if arrays_identical:
            print("  *** BUG CONFIRMED: SP and RF predictions are the same array. ***")
            print("  The paired CSV construction has a copy-paste error.")
        else:
            # Compute AUCs independently
            auc_sp_check = roc_auc_score(y_test, sp_preds)
            auc_rf_check = roc_auc_score(y_test, rf_preds)
            print(f"  Independent AUC check — SP: {auc_sp_check:.10f}")
            print(f"  Independent AUC check — RF: {auc_rf_check:.10f}")

            if abs(auc_sp_check - auc_rf_check) < 1e-8:
                print("  GENUINE COINCIDENCE: identical AUC, different predictions.")
                print("  Check: rank correlation of predictions.")
                from scipy.stats import spearmanr
                rho, pval = spearmanr(sp_preds, rf_preds)
                print(f"  Spearman rho = {rho:.6f}, p = {pval:.6f}")
            else:
                print("  AUCs differ — no coincidence to verify.")
    else:
        print(f"  Prediction columns not found. Expected: {COL_SP_PRED}, {COL_RF_PRED}")
        print(f"  Available columns: {list(df_paired.columns)}")
else:
    print(f"  Paired CSV not found at: {PATH_PAIRED}")
    print("  <<< FILL IN: update PATH_PAIRED with the correct filename >>>")

# -----------------------------------------------------------------------------
# BLOCK V3 — GPA_S1 COLLAPSE DECOMPOSITION
# -----------------------------------------------------------------------------
# CLOSES            : Item 3
# OPEN ITEM         : GPA_S1-excluded ROC-AUC = 0.5063 ± 0.0063. The collapse
#                     is not decomposed — we do not know if the model is
#                     uniformly at chance or if a subset carries signal.
# ANSWERS           : Which of the 48 remaining predictors carry any signal?
# EXPECTED DIRECTION: NEUTRAL
# RUNTIME           : ~3 minutes (fit + SHAP on 468 rows × 48 features)
# IF IT RETURNS
# SOMETHING BAD     : If all SHAP values are near zero, the model is uniformly
#                     at chance and the study's whole framing needs restructuring.
#                     If a subset has non-trivial SHAP, targeted restructuring
#                     is possible.
# PASTE OUTPUT WHERE: R10 Negative Results
# -----------------------------------------------------------------------------

print()
print("=" * 70)
print("BLOCK V3 — GPA_S1 COLLAPSE DECOMPOSITION")
print("=" * 70)

# Build the 48-feature list (remove GPA_S1)
FEATURES_48 = [f for f in FEATURES_49 if f != GPA_S1_COL]
print(f"  Full feature set: {len(FEATURES_49)}")
print(f"  GPA_S1-excluded set: {len(FEATURES_48)}")
print(f"  Removed: {GPA_S1_COL}")

assert len(FEATURES_48) == 48, f"Expected 48 features, got {len(FEATURES_48)}"

X_train_48 = df_train[FEATURES_48].values
X_test_48  = df_test[FEATURES_48].values

# Fit with n_jobs=1 for determinism
model_48 = XGBClassifier(**SP_XGB_PARAMS, n_jobs=1)
model_48.fit(X_train_48, y_train)
pred_48 = model_48.predict_proba(X_test_48)[:, 1]
auc_48 = roc_auc_score(y_test, pred_48)

print(f"  GPA_S1-excluded ROC-AUC: {auc_48:.10f}")

# Prediction variance — is the model uniformly at chance?
pred_std = np.std(pred_48)
pred_range = np.ptp(pred_48)
print(f"  Prediction SD:   {pred_std:.6f}")
print(f"  Prediction range: {pred_range:.6f}")

if pred_std < 0.01:
    print("  *** MODEL IS UNIFORMLY AT CHANCE — predictions barely vary. ***")
else:
    print("  Model predictions vary; some signal may remain.")

# SHAP decomposition on the 48-feature model
print()
print("  Computing SHAP values for the 48-feature model ...")
explainer_48 = shap.TreeExplainer(model_48)
shap_values_48 = explainer_48.shap_values(X_train_48)

# Mean absolute SHAP importance
mean_abs_shap_48 = np.mean(np.abs(shap_values_48), axis=0)

# Build importance table
importance_48 = pd.DataFrame({
    "feature": FEATURES_48,
    "mean_abs_shap": mean_abs_shap_48,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print()
print("  TOP 20 FEATURES — GPA_S1-EXCLUDED MODEL")
print("  " + "-" * 55)
print(f"  {'Rank':<6}{'Feature':<40}{'Mean |SHAP|':>12}")
print("  " + "-" * 55)
for i, row in importance_48.head(20).iterrows():
    print(f"  {i+1:<6}{row['feature']:<40}{row['mean_abs_shap']:>12.6f}")

# Summary diagnostics
top_shap = importance_48["mean_abs_shap"].iloc[0]
total_shap = importance_48["mean_abs_shap"].sum()
n_nonzero = (importance_48["mean_abs_shap"] > 1e-8).sum()
n_meaningful = (importance_48["mean_abs_shap"] > 0.01).sum()

print()
print("  DECOMPOSITION SUMMARY")
print(f"  Top feature mean |SHAP|:     {top_shap:.6f}")
print(f"  Total mean |SHAP| across 48: {total_shap:.6f}")
print(f"  Features with |SHAP| > 1e-8: {n_nonzero}")
print(f"  Features with |SHAP| > 0.01: {n_meaningful}")

if n_meaningful == 0:
    print()
    print("  *** NO FEATURE CARRIES MEANINGFUL SIGNAL IN THE 48-FEATURE MODEL. ***")
    print("  The collapse is UNIFORM. The study's entire framing is at risk.")
elif n_meaningful <= 5:
    print()
    print(f"  *** ONLY {n_meaningful} FEATURES CARRY MEANINGFUL SIGNAL. ***")
    print("  Restructuring should target a narrow subset of predictors.")
else:
    print()
    print(f"  *** {n_meaningful} FEATURES CARRY MEANINGFUL SIGNAL. ***")
    print("  The model retains some discriminative capacity without GPA_S1.")

# -----------------------------------------------------------------------------
# BLOCK V4 — THRESHOLD-SELECTION ANALYSIS
# -----------------------------------------------------------------------------
# CLOSES            : Item 5
# OPEN ITEM         : M15 argues false negatives cost more, but the model
#                     evaluates at symmetric 0.50. No threshold analysis exists.
# ANSWERS           : What precision/recall/workload results at target recall ~0.80?
# EXPECTED DIRECTION: NEUTRAL
# RUNTIME           : <1 minute (no refit — reuses V1 predictions)
# IF IT RETURNS
# SOMETHING BAD     : If precision collapses at the target recall, the false-
#                     negative-cost argument does not survive contact with data.
# PASTE OUTPUT WHERE: Methods M15 / Results R10
# -----------------------------------------------------------------------------

print()
print("=" * 70)
print("BLOCK V4 — THRESHOLD-SELECTION ANALYSIS")
print("=" * 70)

# Use the n_jobs=1 predictions from V1
sp_preds_v1 = pred_b

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, sp_preds_v1)

# Find threshold closest to target recall = 0.80
target_recall = 0.80
idx = np.argmin(np.abs(tpr - target_recall))
chosen_threshold = thresholds[idx]
achieved_recall = tpr[idx]

# Apply threshold
y_pred_thresh = (sp_preds_v1 >= chosen_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_thresh)

tn, fp, fn, tp = cm.ravel()

precision_at_thresh = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall_at_thresh = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_at_thresh = 2 * precision_at_thresh * recall_at_thresh / (precision_at_thresh + recall_at_thresh) if (precision_at_thresh + recall_at_thresh) > 0 else 0.0
specificity_at_thresh = tn / (tn + fp) if (tn + fp) > 0 else 0.0
accuracy_at_thresh = (tp + tn) / (tp + tn + fp + fn)

print(f"  Target recall:         {target_recall:.2f}")
print(f"  Chosen threshold:      {chosen_threshold:.6f}")
print(f"  Achieved recall:       {recall_at_thresh:.4f}")
print(f"  Achieved precision:    {precision_at_thresh:.4f}")
print(f"  Achieved F1:           {f1_at_thresh:.4f}")
print(f"  Achieved specificity:  {specificity_at_thresh:.4f}")
print(f"  Achieved accuracy:     {accuracy_at_thresh:.4f}")
print()
print(f"  Confusion matrix at target recall:")
print(f"    TN={tn}  FP={fp}")
print(f"    FN={fn}  TP={tp}")
print()
print(f"  Advisor workload (students flagged At-Risk): {tp + fp} / {len(y_test)}")
print(f"  Of these, true At-Risk:  {tp}")
print(f"  Of these, false alarms: {fp}")
print(f"  Missed At-Risk (false negatives): {fn}")

# Also show the 0.50 threshold for comparison
y_pred_050 = (sp_preds_v1 >= 0.50).astype(int)
cm_050 = confusion_matrix(y_test, y_pred_050)
tn_050, fp_050, fn_050, tp_050 = cm_050.ravel()
print()
print(f"  FOR COMPARISON — threshold 0.50:")
print(f"    Recall:    {tp_050/(tp_050+fn_050):.4f}")
print(f"    Precision: {tp_050/(tp_050+fp_050):.4f}")
print(f"    Flagged:   {tp_050 + fp_050}")

# -----------------------------------------------------------------------------
# BLOCK V5 — BUDGET PARITY TABLE
# -----------------------------------------------------------------------------
# CLOSES            : Item 4 (partial — records the current state)
# OPEN ITEM         : LR/RF/SVM use 5/64/8 candidates vs SP-XGBoost's 50
#                     Optuna trials. Objective 1's superiority claim is
#                     exploratory only.
# ANSWERS           : What are the exact trial counts per arm?
# EXPECTED DIRECTION: NEUTRAL
# RUNTIME           : <1 minute (no refit — reads notebook metadata)
# IF IT RETURNS
# SOMETHING BAD     : Nothing bad — this records the current state honestly.
# PASTE OUTPUT WHERE: Methods M13
# -----------------------------------------------------------------------------

print()
print("=" * 70)
print("BLOCK V5 — BUDGET PARITY TABLE")
print("=" * 70)

budget_data = {
    "Model": ["SP-XGBoost", "Tuned Baseline XGBoost", "Logistic Regression",
              "Random Forest", "SVM"],
    "Search Method": ["Optuna TPE", "Optuna TPE", "GridSearchCV",
                      "GridSearchCV", "GridSearchCV"],
    "Candidates": [50, 50, 5, 64, 8],  # QUOTED from M13
    "Budget-Matched": ["YES (reference)", "YES", "NO", "NO", "NO"],
}

df_budget = pd.DataFrame(budget_data)
print(df_budget.to_string(index=False))

print()
print("  CONCLUSION: Only SP-XGBoost and Tuned Baseline XGBoost are")
print("  budget-matched. Objective 1's cross-family comparison remains")
print("  EXPLORATORY. Do not claim confirmatory superiority.")

# -----------------------------------------------------------------------------
# BLOCK V6 — SINGLE-CANONICAL-ARTIFACT AUDIT
# -----------------------------------------------------------------------------
# CLOSES            : Item 6 (partial — identifies what needs re-exporting)
# OPEN ITEM         : Table 4 draws from artifact A (ROC-AUC 0.8138),
#                     everything else from artifact B (ROC-AUC 0.8088).
# ANSWERS           : Which tables cite which ROC-AUC values?
# EXPECTED DIRECTION: NEUTRAL
# RUNTIME           : <1 minute
# IF IT RETURNS
# SOMETHING BAD     : If the audit shows more than 2 distinct ROC-AUC values,
#                     there may be additional undiscovered artifacts.
# PASTE OUTPUT WHERE: Methods M14
# -----------------------------------------------------------------------------

print()
print("=" * 70)
print("BLOCK V6 — SINGLE-CANONICAL-ARTIFACT AUDIT")
print("=" * 70)

# QUOTED from the documents
artifact_registry = {
    "Table 4 (R5, ablation)": {
        "roc_auc": 0.813817,
        "source": "07_Optuna_tuning_SP-XGBoost.ipynb",
        "recall": 0.6429,
        "precision": 0.7826,
        "confusion": "TN=51, FP=10, FN=20, TP=36",
    },
    "R3/R4/Table 5/R7 (main results)": {
        "roc_auc": 0.8088,
        "source": "12/14/15/16 (refit)",
        "recall": 0.6786,
        "precision": 0.7600,
        "confusion": "TN=49, FP=12, FN=18, TP=38",
    },
    "R1 (matched three-seed mean)": {
        "roc_auc": 0.8113,
        "source": "mean across seeds 42/123/7",
        "recall": 0.6488,
        "precision": 0.7618,
        "confusion": "N/A (mean)",
    },
    "V1 fit A (n_jobs=-1)": {
        "roc_auc": round(auc_a, 6),
        "source": "THIS SCRIPT",
        "recall": "see below",
        "precision": "see below",
        "confusion": "see below",
    },
    "V1 fit B (n_jobs=1)": {
        "roc_auc": round(auc_b, 6),
        "source": "THIS SCRIPT",
        "recall": "see below",
        "precision": "see below",
        "confusion": "see below",
    },
}

print(f"  {'Source':<42}{'ROC-AUC':>12}")
print("  " + "-" * 54)
for source, data in artifact_registry.items():
    print(f"  {source:<42}{data['roc_auc']:>12.6f}")

print()
print("  DIVERGENCE: 0.813817 vs 0.8088 — a gap of "
      f"{0.813817 - 0.8088:.6f} ROC-AUC points.")
print("  The two artifacts come from different notebook runs.")

# Compute confusion matrix for fit A and fit B
for label, preds in [("V1 fit A (n_jobs=-1)", pred_a), ("V1 fit B (n_jobs=1)", pred_b)]:
    y_pred = (preds >= 0.50).astype(int)
    cm_v = confusion_matrix(y_test, y_pred)
    tn_v, fp_v, fn_v, tp_v = cm_v.ravel()
    prec_v = precision_score(y_test, y_pred, zero_division=0)
    rec_v = recall_score(y_test, y_pred, zero_division=0)
    print(f"  {label}: TN={tn_v} FP={fp_v} FN={fn_v} TP={tp_v} "
          f"| prec={prec_v:.4f} rec={rec_v:.4f}")

# =============================================================================
# HALF 2 — CLOSE (GATED — DO NOT SET TRUE YET)
# =============================================================================

RUN_CLOSE_BLOCKS = False   # Set True ONLY after Half 1 output is returned
                           # and a cause has been confirmed.
                           # A student who sets this True without returning
                           # Half 1 has changed their pipeline for an
                           # unconfirmed cause, and the number that comes out
                           # is attributable to nothing.

if RUN_CLOSE_BLOCKS:

    # -------------------------------------------------------------------------
    # BLOCK K1 — LOCK CANONICAL SP-XGBOOST ARTIFACT (n_jobs=1)
    # -------------------------------------------------------------------------
    # CLOSES            : Item 1 (full closure)
    # CAUSE             : XGBoost nondeterminism via n_jobs > 1
    # CONFIRMED BY      : V1 showing divergent predictions under n_jobs=-1
    # EXPECTED DIRECTION: NEUTRAL — the correct number is whatever n_jobs=1 gives
    # RUNTIME           : ~2 minutes
    # PASTE OUTPUT WHERE: Methods M14 / supplementary artifact
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("BLOCK K1 — LOCK CANONICAL SP-XGBOOST ARTIFACT (n_jobs=1)")
    print("=" * 70)

    # Fit the canonical model with n_jobs=1
    canonical_model = XGBClassifier(**SP_XGB_PARAMS, n_jobs=1)
    canonical_model.fit(X_train, y_train)

    canonical_pred = canonical_model.predict_proba(X_test)[:, 1]
    canonical_auc = roc_auc_score(y_test, canonical_pred)

    y_pred_canon = (canonical_pred >= 0.50).astype(int)
    cm_canon = confusion_matrix(y_test, y_pred_canon)
    tn_c, fp_c, fn_c, tp_c = cm_canon.ravel()

    canon_metrics = {
        "roc_auc": canonical_auc,
        "recall": recall_score(y_test, y_pred_canon, zero_division=0),
        "precision": precision_score(y_test, y_pred_canon, zero_division=0),
        "f1": f1_score(y_test, y_pred_canon, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred_canon),
        "specificity": tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0.0,
        "confusion": f"TN={tn_c} FP={fp_c} FN={fn_c} TP={tp_c}",
        "n_jobs": 1,
        "random_state": 42,
        "n_features": len(FEATURES_49),
    }

    print(f"  CANONICAL SP-XGBoost (n_jobs=1, seed=42):")
    print(f"    ROC-AUC:    {canonical_auc:.10f}")
    print(f"    Recall:     {canon_metrics['recall']:.4f}")
    print(f"    Precision:  {canon_metrics['precision']:.4f}")
    print(f"    F1:         {canon_metrics['f1']:.4f}")
    print(f"    Specificity:{canon_metrics['specificity']:.4f}")
    print(f"    Confusion:  {canon_metrics['confusion']}")

    # Export canonical predictions
    canon_df = pd.DataFrame({
        STUDENT_ID_COL: df_test[STUDENT_ID_COL].values,
        "y_true": y_test,
        "y_pred_proba": canonical_pred,
        "y_pred_class": y_pred_canon,
    })
    canon_path = os.path.join(ARTIFACT_DIR, "canonical_sp_xgboost_predictions.csv")
    canon_df.to_csv(canon_path, index=False)
    print(f"  Canonical predictions exported to: {canon_path}")

    # Export canonical metrics
    import json
    metrics_path = os.path.join(ARTIFACT_DIR, "canonical_sp_xgboost_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(canon_metrics, f, indent=2)
    print(f"  Canonical metrics exported to: {metrics_path}")

    # -------------------------------------------------------------------------
    # BLOCK K2 — RE-EXPORT ALL TABLES FROM CANONICAL ARTIFACT
    # -------------------------------------------------------------------------
    # CLOSES            : Item 6
    # CAUSE             : Multiple artifact sources
    # CONFIRMED BY      : V6 showing divergent ROC-AUC values across tables
    # EXPECTED DIRECTION: NEUTRAL
    # RUNTIME           : ~5 minutes (refits all models with n_jobs=1)
    # PASTE OUTPUT WHERE: Results R2–R7, Tables 3–5
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("BLOCK K2 — RE-EXPORT ALL TABLES FROM CANONICAL ARTIFACT")
    print("=" * 70)

    # This block should be run AFTER K1 has locked the canonical model.
    # It re-derives every table from the canonical predictions.

    print("  Canonical artifact locked. Re-export every table from:")
    print(f"    {canon_path}")
    print()
    print("  ACTION REQUIRED: In your notebooks, replace every SP-XGBoost")
    print("  prediction source with the canonical CSV. Re-run all downstream")
    print("  cells that compute ROC-AUC, recall, precision, F1, specificity,")
    print("  accuracy, and confusion matrices.")
    print()
    print("  Tables to re-export:")
    print("    - Table 3 (matched three-seed comparison)")
    print("    - Table 4 (2×2 XGBoost ablation)")
    print("    - Table 5 (per-class precision/recall/F1)")
    print("    - R2 through R7 (all Results sections)")
    print()
    print("  VERIFY: Every reported ROC-AUC must trace to the canonical CSV.")

    # -------------------------------------------------------------------------
    # BLOCK K3 — BUDGET-MATCH LR/RF/SVM (OPTIONAL)
    # -------------------------------------------------------------------------
    # CLOSES            : Item 4 (full closure, if wanted)
    # CAUSE             : Cross-family tuning budget asymmetry
    # CONFIRMED BY      : V5 budget table
    # EXPECTED DIRECTION: NEUTRAL — the result is whatever parity shows
    # RUNTIME           : ~30–60 minutes (50 Optuna trials × 3 models)
    # PASTE OUTPUT WHERE: Methods M13 / R2
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("BLOCK K3 — BUDGET-MATCH LR/RF/SVM (OPTIONAL)")
    print("=" * 70)

    print("  This block is OPTIONAL. It strengthens Objective 1 if a")
    print("  confirmatory (not merely exploratory) benchmark claim is wanted.")
    print()
    print("  To run it, use the same Optuna protocol as SP-XGBoost:")
    print("    - 50 trials per model")
    print("    - 5-fold stratified CV")
    print("    - ROC-AUC objective")
    print("    - Seeds 42, 123, 7")
    print()
    print("  This requires importing Optuna and defining model-specific")
    print("  search spaces for LR, RF, and SVM. Do not attempt without")
    print("  reviewing the SP-XGBoost Optuna study definition first.")

# =============================================================================
# VERDICT TABLE — THE ONLY THING YOU NEED TO PASTE BACK
# =============================================================================

print()
print("=" * 70)
print("VERDICT TABLE")
print("=" * 70)

verdicts = [
    ("Item 1", "Artifact reconciliation",
     "DIVERGENT" if not np.array_equal(pred_a, pred_b) else "IDENTICAL",
     "IDENTICAL", "STILL OPEN"),
    ("Item 2", "Paired-prediction CSV",
     "SEE V2 ABOVE", "GENUINE", "SEE OUTPUT"),
    ("Item 3", "GPA_S1 decomposition",
     f"{n_meaningful} features >0.01", "ANY SIGNAL",
     "CLOSES ON RUN"),
    ("Item 4", "Budget parity",
     "5/50/64/8 candidates", "50/50/50/50",
     "STILL OPEN"),
    ("Item 5", "Threshold analysis",
     f"recall={recall_at_thresh:.2f}", "~0.80",
     "CLOSES ON RUN"),
    ("Item 6", "Single canonical run",
     "2 artifacts detected", "1 artifact",
     "STILL OPEN"),
]

print(f"  {'ITEM':<10}{'CHECK':<28}{'OBSERVED':>16}{'EXPECTED':>14}  VERDICT")
print("  " + "-" * 82)
for item, check, observed, expected, verdict in verdicts:
    print(f"  {item:<10}{check:<28}{observed:>16}{expected:>14}  {verdict}")

# Summary counts
closed = sum(1 for _, _, _, _, v in verdicts if v == "CLOSED")
still_open = sum(1 for _, _, _, _, v in verdicts if v == "STILL OPEN")
closes_on_run = sum(1 for _, _, _, _, v in verdicts if v == "CLOSES ON RUN")
see_output = sum(1 for _, _, _, _, v in verdicts if v == "SEE OUTPUT")

print()
print(f"  CLOSED: {closed}   STILL OPEN: {still_open}   "
      f"CLOSES ON RUN: {closes_on_run}   SEE OUTPUT: {see_output}")

# Null state
if n_meaningful == 0:
    null_state = "2 — ARTEFACT (uniform chance — whole framing at risk)"
elif n_meaningful <= 5:
    null_state = f"2 — ARTEFACT ({n_meaningful} features carry signal)"
else:
    null_state = f"2 — ARTEFACT ({n_meaningful} features carry signal — partial capacity)"

print(f"  NULL STATE: {null_state}")
print()
print("  PASTE THIS TABLE BACK. DO NOT SET RUN_CLOSE_BLOCKS = True YET.")
print("=" * 70)