"""
Additional analysis for PUP/Condo retention evaluation

Part A: Performance by tenure (nt6, max_ntr_adw, high-tenure deep dive)
Part B: 4-model comparison (isolate window effect vs line effect)

Run after the existing evaluation notebook cells (assumes df_with, df_without,
training_df_with, training_df_without, bst_with, bst_without are already loaded).
"""

# %% ── Cell 1: Imports & helpers ──

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from classic_spl_ltv.jobs.retention.pipeline import load_model


def compute_slice_metrics(df, group_col, group_val):
    """Compute metrics for a subset of df where group_col == group_val."""
    subset = df[df[group_col] == group_val]
    y_true = subset["term_ind"]
    y_score = subset["model_prediction"]

    row = {
        group_col: int(group_val),
        "records": len(subset),
        "unique_policies": subset["policy_id"].nunique() if "policy_id" in subset.columns else np.nan,
        "actual_term_rate": y_true.mean(),
        "predicted_term_rate": y_score.mean(),
        "diff": y_true.mean() - y_score.mean(),
    }

    if y_true.nunique() > 1:
        y_pred = np.column_stack((1 - y_score, y_score))
        row["roc_auc"] = roc_auc_score(y_true, y_score)
        row["log_loss"] = log_loss(y_true, y_pred)
        row["avg_precision"] = average_precision_score(y_true, y_score)
    else:
        row["roc_auc"] = np.nan
        row["log_loss"] = np.nan
        row["avg_precision"] = np.nan

    return row


def metrics_by_column(df, group_col, label=""):
    """Compute metrics for each unique value of group_col."""
    results = []
    for val in sorted(df[group_col].unique()):
        results.append(compute_slice_metrics(df, group_col, val))
    result_df = pd.DataFrame(results)
    result_df["label"] = label
    return result_df


def score_with_model(df, bst, pred_col_name):
    """Score a DataFrame using a model, handling feature alignment.
    Returns the DataFrame with the new prediction column added."""
    model_features = bst.feature_name()
    missing = set(model_features) - set(df.columns)
    if missing:
        print(f"  WARNING: {len(missing)} features missing from data: {sorted(missing)[:10]}...")
        print(f"  Cannot score with this model. Skipping.")
        return None
    df = df.copy()
    df[pred_col_name] = bst.predict(df[model_features])
    return df


def compute_metrics_from_cols(df, y_true_col, y_score_col, label):
    """Compute metrics given column names."""
    y_true = df[y_true_col]
    y_score = df[y_score_col]
    y_pred = np.column_stack((1 - y_score, y_score))
    row = {
        "model": label,
        "records": len(df),
        "actual_term_rate": y_true.mean(),
        "predicted_term_rate": y_score.mean(),
        "diff": y_true.mean() - y_score.mean(),
    }
    if y_true.nunique() > 1:
        row["roc_auc"] = roc_auc_score(y_true, y_score)
        row["log_loss"] = log_loss(y_true, y_pred)
        row["avg_precision"] = average_precision_score(y_true, y_score)
    else:
        row["roc_auc"] = np.nan
        row["log_loss"] = np.nan
        row["avg_precision"] = np.nan
    return row


# ============================================================================
# PART A: TENURE ANALYSIS
# ============================================================================

# %% ── Cell 2: Slice 1 — Performance by nt6 bucket ──

print("=" * 80)
print("SLICE 1: Performance by nt6 (each exploded row's renewal interval)")
print("=" * 80)

nt6_with = metrics_by_column(df_with, "nt6", "with PUP/Condo")
nt6_without = metrics_by_column(df_without, "nt6", "without PUP/Condo")

print("\n--- With PUP/Condo ---")
print(nt6_with.to_string(index=False))
print("\n--- Without PUP/Condo ---")
print(nt6_without.to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Slice 1: Metrics by nt6 Bucket", fontsize=14, fontweight="bold")

for i, (metric, title) in enumerate([
    ("roc_auc", "ROC AUC"),
    ("avg_precision", "Avg Precision"),
    ("actual_term_rate", "Actual Term Rate"),
    ("log_loss", "Log Loss"),
]):
    ax = axes[i // 2, i % 2]
    ax.bar(nt6_with["nt6"] - 0.2, nt6_with[metric], width=0.4,
           label="With PUP/Condo", color="steelblue")
    ax.bar(nt6_without["nt6"] + 0.2, nt6_without[metric], width=0.4,
           label="Without PUP/Condo", color="indianred")
    ax.set_xlabel("nt6")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(range(10))
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# %% ── Cell 3: Slice 2 — Performance by max_ntr_adw (policy tenure) ──

print("\n" + "=" * 80)
print("SLICE 2: Performance by max_ntr_adw (policy-level tenure)")
print("=" * 80)

ntr_col = None
for candidate in ["max_ntr_adw", "starting_ntr_adw", "max_ntr", "ntr_adw"]:
    if candidate in df_with.columns:
        ntr_col = candidate
        break

if ntr_col is None:
    ntr_candidates = [c for c in df_with.columns
                      if any(x in c.lower() for x in ["ntr", "max_n", "renew", "current_nt"])]
    print(f"Could not find max_ntr_adw. Candidates found: {ntr_candidates}")
    print("Update ntr_col below and re-run.")
else:
    print(f"Using column: {ntr_col}")
    print(f"Unique values: {sorted(df_with[ntr_col].unique())}")

    ntr_with = metrics_by_column(df_with, ntr_col, "with PUP/Condo")
    ntr_without = metrics_by_column(df_without, ntr_col, "without PUP/Condo")

    print(f"\n--- With PUP/Condo (by {ntr_col}) ---")
    print(ntr_with.to_string(index=False))
    print(f"\n--- Without PUP/Condo (by {ntr_col}) ---")
    print(ntr_without.to_string(index=False))

    n_unique = df_with[ntr_col].nunique()
    if n_unique > 15:
        print(f"\n[Note: {ntr_col} has {n_unique} unique values — consider binning for charts]")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Slice 2: Metrics by {ntr_col} (Policy Tenure)", fontsize=14, fontweight="bold")

    for i, (metric, title) in enumerate([
        ("roc_auc", "ROC AUC"),
        ("avg_precision", "Avg Precision"),
        ("actual_term_rate", "Actual Term Rate"),
        ("log_loss", "Log Loss"),
    ]):
        ax = axes[i // 2, i % 2]
        ax.bar(ntr_with[ntr_col] - 0.2, ntr_with[metric], width=0.4,
               label="With PUP/Condo", color="steelblue")
        ax.bar(ntr_without[ntr_col] + 0.2, ntr_without[metric], width=0.4,
               label="Without PUP/Condo", color="indianred")
        ax.set_xlabel(ntr_col)
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

# %% ── Cell 4: Slice 3 — High-tenure deep dive ──

print("\n" + "=" * 80)
print("SLICE 3: High-Tenure Deep Dive")
print("=" * 80)

# 3a: Rows where nt6 >= 8
print("\n--- 3a: Rows where nt6 >= 8 (late renewal intervals only) ---\n")

for label, df in [("With PUP/Condo", df_with), ("Without PUP/Condo", df_without)]:
    subset = df[df["nt6"] >= 8]
    y_true = subset["term_ind"]
    y_score = subset["model_prediction"]
    print(f"  {label}:")
    print(f"    Records:             {len(subset):,}")
    print(f"    Unique policies:     {subset['policy_id'].nunique() if 'policy_id' in subset.columns else 'N/A':,}")
    print(f"    Actual term rate:    {y_true.mean():.4f}")
    print(f"    Predicted term rate: {y_score.mean():.4f}")
    print(f"    Diff:                {y_true.mean() - y_score.mean():.4f}")
    if y_true.nunique() > 1:
        print(f"    ROC AUC:             {roc_auc_score(y_true, y_score):.4f}")
        print(f"    Avg Precision:       {average_precision_score(y_true, y_score):.4f}")
    else:
        print(f"    ROC AUC:             N/A (single class)")
    print()

# 3b: ALL rows for policies with max_ntr_adw >= 8
if ntr_col and ntr_col in df_with.columns:
    print(f"--- 3b: ALL rows for policies where {ntr_col} >= 8 ---\n")

    for label, df in [("With PUP/Condo", df_with), ("Without PUP/Condo", df_without)]:
        subset = df[df[ntr_col] >= 8]
        y_true = subset["term_ind"]
        y_score = subset["model_prediction"]
        print(f"  {label}:")
        print(f"    Records:             {len(subset):,}")
        print(f"    Unique policies:     {subset['policy_id'].nunique() if 'policy_id' in subset.columns else 'N/A':,}")
        print(f"    Actual term rate:    {y_true.mean():.4f}")
        print(f"    Predicted term rate: {y_score.mean():.4f}")
        print(f"    Diff:                {y_true.mean() - y_score.mean():.4f}")
        if y_true.nunique() > 1:
            print(f"    ROC AUC:             {roc_auc_score(y_true, y_score):.4f}")
            print(f"    Avg Precision:       {average_precision_score(y_true, y_score):.4f}")
        else:
            print(f"    ROC AUC:             N/A (single class)")
        print()

    # 3c: High-tenure policies broken down by nt6 step
    print(f"--- 3c: Policies with {ntr_col} >= 8, metrics at EACH nt6 step ---")
    print("    (Caveat: survivorship bias — these policies survived to high tenure,")
    print("     so early nt6 rows are retrospectively known retentions)\n")

    high_tenure_with = df_with[df_with[ntr_col] >= 8]
    high_tenure_without = df_without[df_without[ntr_col] >= 8]

    ht_nt6_with = metrics_by_column(high_tenure_with, "nt6", "with PUP/Condo")
    ht_nt6_without = metrics_by_column(high_tenure_without, "nt6", "without PUP/Condo")

    print("  With PUP/Condo:")
    print(ht_nt6_with.to_string(index=False))
    print("\n  Without PUP/Condo:")
    print(ht_nt6_without.to_string(index=False))

# %% ── Cell 5: Calibration plots ──

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Calibration: Predicted vs Actual Term Rate", fontsize=14, fontweight="bold")

ax = axes[0]
for df_metrics, color, lbl in [
    (nt6_with, "steelblue", "With"),
    (nt6_without, "indianred", "Without"),
]:
    ax.scatter(df_metrics["predicted_term_rate"], df_metrics["actual_term_rate"],
              s=df_metrics["records"] / 5000, alpha=0.7, color=color, label=lbl)
    for _, row in df_metrics.iterrows():
        ax.annotate(f'{int(row["nt6"])}', (row["predicted_term_rate"], row["actual_term_rate"]),
                    fontsize=7, ha="center", va="bottom")
lims = [0, max(nt6_with["actual_term_rate"].max(), nt6_with["predicted_term_rate"].max()) * 1.15]
ax.plot(lims, lims, "--", color="gray", alpha=0.5)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("By nt6 bucket")
ax.legend(fontsize=8)

ax = axes[1]
if ntr_col and ntr_col in df_with.columns:
    for df_metrics, color, lbl in [
        (ntr_with, "steelblue", "With"),
        (ntr_without, "indianred", "Without"),
    ]:
        ax.scatter(df_metrics["predicted_term_rate"], df_metrics["actual_term_rate"],
                  s=df_metrics["records"] / 5000, alpha=0.7, color=color, label=lbl)
        for _, row in df_metrics.iterrows():
            ax.annotate(f'{int(row[ntr_col])}', (row["predicted_term_rate"], row["actual_term_rate"]),
                        fontsize=7, ha="center", va="bottom")
    ax.plot(lims, lims, "--", color="gray", alpha=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"By {ntr_col}")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "Column not found", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("By max_ntr_adw (N/A)")

ax = axes[2]
if ntr_col and ntr_col in df_with.columns:
    for df_metrics, color, lbl in [
        (ht_nt6_with, "steelblue", "With"),
        (ht_nt6_without, "indianred", "Without"),
    ]:
        ax.scatter(df_metrics["predicted_term_rate"], df_metrics["actual_term_rate"],
                  s=df_metrics["records"] / 2000, alpha=0.7, color=color, label=lbl)
        for _, row in df_metrics.iterrows():
            ax.annotate(f'nt6={int(row["nt6"])}', (row["predicted_term_rate"], row["actual_term_rate"]),
                        fontsize=7, ha="center", va="bottom")
    ax.plot(lims, lims, "--", color="gray", alpha=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"High-tenure ({ntr_col}>=8) by nt6")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

plt.tight_layout()
plt.show()

# %% ── Cell 6: Record count summary ──

print("\n" + "=" * 80)
print("RECORD COUNT SUMMARY — How thin is the tail?")
print("=" * 80)

for label, df in [("With PUP/Condo", df_with), ("Without PUP/Condo", df_without)]:
    print(f"\n--- {label} ---")
    counts = df.groupby("nt6").agg(
        records=("term_ind", "count"),
        unique_policies=("policy_id", "nunique") if "policy_id" in df.columns else ("term_ind", "count"),
        term_rate=("term_ind", "mean"),
        positives=("term_ind", "sum"),
    ).reset_index()
    counts["pct_of_total"] = (counts["records"] / counts["records"].sum() * 100).round(2)
    print(counts.to_string(index=False))


# ============================================================================
# PART B: 4-MODEL COMPARISON
#
# Design:
#   - Window effect: score SAME test set with old vs new model (same lines)
#   - Line effect: score SAME population (shared 4 lines only) with
#     "with" model vs "without" model (same window)
#
# This avoids conflating population differences with model differences.
# ============================================================================

# %% ── Cell 7: Load models, build common test set ──

print("\n" + "=" * 80)
print("PART B: 4-MODEL COMPARISON")
print("=" * 80)

# --- Model paths (UPDATE THESE) ---
model_path_old_with    = "TODO"  # old (full history), with PUP/Condo
model_path_old_without = "TODO"  # old (full history), without PUP/Condo
model_path_new_with    = "tmx-smsiweb/classic-specialty-ltv/prod/feature/B-2760706/retention/models/other_short_ret.mdl"  # UPDATE if needed
model_path_new_without = "tmx-smsiweb/classic-specialty-ltv/prod/feature/B-2779253/retention/models/other_short_ret.mdl"  # UPDATE if needed

# Uncomment once paths are set:
"""
# Load all 4 models
bst_old_with = load_model(model_path_old_with)
bst_old_without = load_model(model_path_old_without)
bst_new_with = bst_with        # already loaded in earlier notebook cells
bst_new_without = bst_without  # already loaded in earlier notebook cells

# --- Build common test set for line effect comparison ---
# Filter df_with to only the 4 shared lines (exclude 078 condo, 088 pup)
shared_lines = ["016", "032", "072", "090"]
df_shared = df_with[df_with["ply_line_cd"].isin(shared_lines)].copy()
print(f"Shared-lines test set: {len(df_shared):,} records "
      f"(dropped {len(df_with) - len(df_shared):,} PUP/Condo records from df_with)")
print(f"df_without test set:   {len(df_without):,} records")

# Sanity check: shared lines test set should be ~same size as df_without
assert abs(len(df_shared) - len(df_without)) / len(df_without) < 0.05, \\
    f"Size mismatch: df_shared={len(df_shared)}, df_without={len(df_without)} — check line filtering"
"""

# %% ── Cell 8: Score all combinations ──

# Uncomment once Cell 7 is done:
"""
print("\\nScoring all model x test set combinations...\\n")

# --- Window effect: same lines, old vs new model ---
# "With" lines: score df_with with old_with and new_with
df_with = score_with_model(df_with, bst_old_with, "pred_old_with")
# pred for new_with already exists as "model_prediction"

# "Without" lines: score df_without with old_without and new_without
df_without = score_with_model(df_without, bst_old_without, "pred_old_without")
# pred for new_without already exists as "model_prediction"

# --- Line effect: same population (shared lines), different model ---
# Score df_shared with new_with model
# (predictions already exist in df_shared["model_prediction"] since it's a subset of df_with)
df_shared["pred_new_with"] = df_shared["model_prediction"]

# Score df_shared with new_without model
df_shared = score_with_model(df_shared, bst_new_without, "pred_new_without")
if df_shared is None:
    print("Feature mismatch scoring df_shared with new_without model.")
    print("The 'without' model may use different encoded features.")
    print("Check: set(bst_new_without.feature_name()) - set(df_shared.columns)")

# Also score df_shared with both old models for the full picture
df_shared = score_with_model(df_shared, bst_old_with, "pred_old_with")
df_shared = score_with_model(df_shared, bst_old_without, "pred_old_without")

print("Scoring complete.")
"""

# %% ── Cell 9: Aggregate metrics — all 4 models ──

# Uncomment once Cell 8 is done:
"""
print("\\n" + "=" * 80)
print("AGGREGATE METRICS: 4-Model Comparison")
print("=" * 80)

results = []

# Window effect comparisons (each on its native test set)
results.append(compute_metrics_from_cols(
    df_with, "term_ind", "pred_old_with", "Old window + PUP/Condo"))
results.append(compute_metrics_from_cols(
    df_with, "term_ind", "model_prediction", "New window (5yr) + PUP/Condo"))
results.append(compute_metrics_from_cols(
    df_without, "term_ind", "pred_old_without", "Old window - PUP/Condo"))
results.append(compute_metrics_from_cols(
    df_without, "term_ind", "model_prediction", "New window (5yr) - PUP/Condo"))

print("\\n--- Each model scored on its native test set ---")
print(pd.DataFrame(results).to_string(index=False))
"""

# %% ── Cell 10: Isolate window effect ──

# Uncomment once Cell 8 is done:
"""
print("\\n" + "=" * 80)
print("ISOLATED WINDOW EFFECT (same lines, old vs new training window)")
print("  Note: old model evaluated on 5yr-window test data, which it wasn't")
print("  optimized for. This tests how well old patterns generalize to current data.")
print("=" * 80)

for line_label, df, old_col, new_col in [
    ("With PUP/Condo", df_with, "pred_old_with", "model_prediction"),
    ("Without PUP/Condo", df_without, "pred_old_without", "model_prediction"),
]:
    y_true = df["term_ind"]
    auc_old = roc_auc_score(y_true, df[old_col])
    auc_new = roc_auc_score(y_true, df[new_col])
    ap_old = average_precision_score(y_true, df[old_col])
    ap_new = average_precision_score(y_true, df[new_col])
    ll_old = log_loss(y_true, np.column_stack((1 - df[old_col], df[old_col])))
    ll_new = log_loss(y_true, np.column_stack((1 - df[new_col], df[new_col])))
    calib_old = y_true.mean() - df[old_col].mean()
    calib_new = y_true.mean() - df[new_col].mean()

    print(f"\\n  {line_label} ({len(df):,} records):")
    print(f"    {'':20s} {'Old':>10s} {'New (5yr)':>10s} {'Delta':>10s}")
    print(f"    {'AUC':20s} {auc_old:10.4f} {auc_new:10.4f} {auc_new - auc_old:+10.4f}")
    print(f"    {'Avg Precision':20s} {ap_old:10.4f} {ap_new:10.4f} {ap_new - ap_old:+10.4f}")
    print(f"    {'Log Loss':20s} {ll_old:10.4f} {ll_new:10.4f} {ll_new - ll_old:+10.4f}")
    print(f"    {'Calibration (A-P)':20s} {calib_old:+10.4f} {calib_new:+10.4f}")
"""

# %% ── Cell 11: Isolate line effect (SAME population, different model) ──

# Uncomment once Cell 8 is done:
"""
print("\\n" + "=" * 80)
print("ISOLATED LINE EFFECT (same 4-line population, 'with' vs 'without' model)")
print(f"  Test set: df_shared ({len(df_shared):,} records, lines {shared_lines})")
print("  Both models see identical rows — only difference is training data composition.")
print("=" * 80)

y_true = df_shared["term_ind"]

for window_label, with_col, without_col in [
    ("New window (5yr)", "pred_new_with", "pred_new_without"),
    ("Old window", "pred_old_with", "pred_old_without"),
]:
    if with_col not in df_shared.columns or without_col not in df_shared.columns:
        print(f"\\n  {window_label}: skipped (missing prediction columns)")
        continue

    auc_with = roc_auc_score(y_true, df_shared[with_col])
    auc_without = roc_auc_score(y_true, df_shared[without_col])
    ap_with = average_precision_score(y_true, df_shared[with_col])
    ap_without = average_precision_score(y_true, df_shared[without_col])
    ll_with = log_loss(y_true, np.column_stack((1 - df_shared[with_col], df_shared[with_col])))
    ll_without = log_loss(y_true, np.column_stack((1 - df_shared[without_col], df_shared[without_col])))
    calib_with = y_true.mean() - df_shared[with_col].mean()
    calib_without = y_true.mean() - df_shared[without_col].mean()

    print(f"\\n  {window_label}:")
    print(f"    {'':20s} {'With model':>12s} {'Without model':>14s} {'Delta':>10s}")
    print(f"    {'AUC':20s} {auc_with:12.4f} {auc_without:14.4f} {auc_without - auc_with:+10.4f}")
    print(f"    {'Avg Precision':20s} {ap_with:12.4f} {ap_without:14.4f} {ap_without - ap_with:+10.4f}")
    print(f"    {'Log Loss':20s} {ll_with:12.4f} {ll_without:14.4f} {ll_without - ll_with:+10.4f}")
    print(f"    {'Calibration (A-P)':20s} {calib_with:+12.4f} {calib_without:+14.4f}")
"""

# %% ── Cell 12: Window effect by nt6 bucket ──

# Uncomment once Cell 8 is done:
"""
print("\\n" + "=" * 80)
print("WINDOW EFFECT BY NT6 BUCKET")
print("  Does dropping pre-2020 data hurt more at high tenure?")
print("=" * 80)

for line_label, df, old_col, new_col in [
    ("With PUP/Condo", df_with, "pred_old_with", "model_prediction"),
    ("Without PUP/Condo", df_without, "pred_old_without", "model_prediction"),
]:
    print(f"\\n--- {line_label} ---")
    rows = []
    for nt6_val in sorted(df["nt6"].unique()):
        subset = df[df["nt6"] == nt6_val]
        y_true = subset["term_ind"]
        if y_true.nunique() <= 1:
            continue
        rows.append({
            "nt6": int(nt6_val),
            "records": len(subset),
            "auc_old": roc_auc_score(y_true, subset[old_col]),
            "auc_new": roc_auc_score(y_true, subset[new_col]),
            "auc_delta": roc_auc_score(y_true, subset[new_col]) - roc_auc_score(y_true, subset[old_col]),
            "calib_old": y_true.mean() - subset[old_col].mean(),
            "calib_new": y_true.mean() - subset[new_col].mean(),
        })
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
"""

# %% ── Cell 13: Line effect by nt6 bucket (on shared population) ──

# Uncomment once Cell 8 is done:
"""
print("\\n" + "=" * 80)
print("LINE EFFECT BY NT6 BUCKET (on shared 4-line population)")
print("  Does including PUP/Condo in training help/hurt at specific tenure levels?")
print("=" * 80)

for window_label, with_col, without_col in [
    ("New window (5yr)", "pred_new_with", "pred_new_without"),
]:
    if with_col not in df_shared.columns or without_col not in df_shared.columns:
        print(f"\\n  {window_label}: skipped")
        continue

    print(f"\\n--- {window_label} ---")
    rows = []
    for nt6_val in sorted(df_shared["nt6"].unique()):
        subset = df_shared[df_shared["nt6"] == nt6_val]
        y_true = subset["term_ind"]
        if y_true.nunique() <= 1:
            continue
        rows.append({
            "nt6": int(nt6_val),
            "records": len(subset),
            "auc_with_model": roc_auc_score(y_true, subset[with_col]),
            "auc_without_model": roc_auc_score(y_true, subset[without_col]),
            "auc_delta": roc_auc_score(y_true, subset[without_col]) - roc_auc_score(y_true, subset[with_col]),
            "calib_with": y_true.mean() - subset[with_col].mean(),
            "calib_without": y_true.mean() - subset[without_col].mean(),
        })
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
"""

print("\\n[Uncomment Cells 7-13 after filling in model paths]")
