"""
Additional analysis for PUP/Condo retention evaluation
1. Performance by nt6 bucket (especially nt6 >= 8)
2. Compare current model vs benchmark/old model (impact of 5-year rolling window)

Run after the existing evaluation notebook cells (assumes df_with, df_without,
training_df_with, training_df_without are already loaded).
"""

# %% ── Cell 1: Performance by nt6 bucket (with PUP/Condo) ──

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score


def metrics_by_nt6(df, label=""):
    """Compute metrics for each nt6 bucket."""
    results = []
    for nt6_val in sorted(df["nt6"].unique()):
        subset = df[df["nt6"] == nt6_val]
        y_true = subset["term_ind"]
        y_score = subset["model_prediction"]
        y_pred = np.column_stack((1 - y_score, y_score))

        row = {
            "nt6": int(nt6_val),
            "records": len(subset),
            "actual_term_rate": y_true.mean(),
            "predicted_term_rate": y_score.mean(),
            "diff": y_true.mean() - y_score.mean(),
        }

        # AUC and log loss need both classes present
        if y_true.nunique() > 1:
            row["roc_auc"] = roc_auc_score(y_true, y_score)
            row["log_loss"] = log_loss(y_true, y_pred)
            row["avg_precision"] = average_precision_score(y_true, y_score)
        else:
            row["roc_auc"] = np.nan
            row["log_loss"] = np.nan
            row["avg_precision"] = np.nan

        results.append(row)

    result_df = pd.DataFrame(results)
    result_df["label"] = label
    return result_df


nt6_with = metrics_by_nt6(df_with, "with PUP/Condo")
nt6_without = metrics_by_nt6(df_without, "without PUP/Condo")

print("=== With PUP/Condo: Performance by nt6 ===")
print(nt6_with.to_string(index=False))
print("\n=== Without PUP/Condo: Performance by nt6 ===")
print(nt6_without.to_string(index=False))

# %% ── Cell 2: Visualize metrics by nt6 ──

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for i, (metric, title) in enumerate([
    ("roc_auc", "ROC AUC by nt6"),
    ("avg_precision", "Avg Precision by nt6"),
    ("actual_term_rate", "Actual Term Rate by nt6"),
    ("log_loss", "Log Loss by nt6"),
]):
    ax = axes[i // 2, i % 2]
    ax.bar(
        nt6_with["nt6"] - 0.2, nt6_with[metric], width=0.4,
        label="With PUP/Condo", color="steelblue"
    )
    ax.bar(
        nt6_without["nt6"] + 0.2, nt6_without[metric], width=0.4,
        label="Without PUP/Condo", color="indianred"
    )
    ax.set_xlabel("nt6")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(range(10))
    ax.legend()

plt.tight_layout()
plt.show()

# %% ── Cell 3: Focus on nt6 >= 8 (Yinan's specific ask) ──

print("\n=== nt6 >= 8: With PUP/Condo ===")
high_tenure_with = df_with[df_with["nt6"] >= 8]
print(f"Records: {len(high_tenure_with):,}")
print(f"Actual term rate: {high_tenure_with['term_ind'].mean():.4f}")
print(f"Predicted term rate: {high_tenure_with['model_prediction'].mean():.4f}")
if high_tenure_with["term_ind"].nunique() > 1:
    print(f"ROC AUC: {roc_auc_score(high_tenure_with['term_ind'], high_tenure_with['model_prediction']):.4f}")
    print(f"Avg Precision: {average_precision_score(high_tenure_with['term_ind'], high_tenure_with['model_prediction']):.4f}")
else:
    print("Only one class present — cannot compute AUC")

print("\n=== nt6 >= 8: Without PUP/Condo ===")
high_tenure_without = df_without[df_without["nt6"] >= 8]
print(f"Records: {len(high_tenure_without):,}")
print(f"Actual term rate: {high_tenure_without['term_ind'].mean():.4f}")
print(f"Predicted term rate: {high_tenure_without['model_prediction'].mean():.4f}")
if high_tenure_without["term_ind"].nunique() > 1:
    print(f"ROC AUC: {roc_auc_score(high_tenure_without['term_ind'], high_tenure_without['model_prediction']):.4f}")
    print(f"Avg Precision: {average_precision_score(high_tenure_without['term_ind'], high_tenure_without['model_prediction']):.4f}")
else:
    print("Only one class present — cannot compute AUC")

# %% ── Cell 4: Calibration by nt6 (actual vs predicted scatter) ──

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, df_nt6, title in [
    (axes[0], nt6_with, "Calibration by nt6 — With PUP/Condo"),
    (axes[1], nt6_without, "Calibration by nt6 — Without PUP/Condo"),
]:
    ax.scatter(df_nt6["predicted_term_rate"], df_nt6["actual_term_rate"],
              s=df_nt6["records"] / 5000, alpha=0.7)
    for _, row in df_nt6.iterrows():
        ax.annotate(f'nt6={int(row["nt6"])}',
                    (row["predicted_term_rate"], row["actual_term_rate"]),
                    fontsize=8, ha="center", va="bottom")

    lims = [0, max(df_nt6["actual_term_rate"].max(), df_nt6["predicted_term_rate"].max()) * 1.1]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5)
    ax.set_xlabel("Predicted Term Rate")
    ax.set_ylabel("Actual Term Rate")
    ax.set_title(title)

plt.tight_layout()
plt.show()

# %% ── Cell 5: Compare vs benchmark model (impact of 5-year window) ──
# TODO: Update this path to the published V9.01 benchmark model scored output

benchmark_path = "tmx-smsiweb/classic-specialty-ltv/prod/main/retention/data/other_short_test_data_scored/"  # <-- UPDATE THIS

# Uncomment once you have the correct path:
# df_benchmark = nsh.read_parquet_s3_to_pandas(benchmark_path)
#
# # Subset benchmark to same 5-year window for apples-to-apples
# max_date = pd.to_datetime(training_df_with["release_day_pol"]).max()
# cutoff = max_date - pd.Timedelta(days=5 * 365)
# df_benchmark = df_benchmark[pd.to_datetime(df_benchmark["release_day_pol"]) >= cutoff]
#
# benchmark_metrics = compute_metrics(df_benchmark, "benchmark (V9.01)")
# with_metrics = compute_metrics(df_with, "with PUP/Condo (5yr)")
# without_metrics = compute_metrics(df_without, "without PUP/Condo (5yr)")
#
# comparison = pd.DataFrame([benchmark_metrics, with_metrics, without_metrics])
# print("=== Model Comparison: Benchmark vs New Models ===")
# print(comparison.to_string(index=False))
#
# # NT6 breakdown for benchmark
# nt6_benchmark = metrics_by_nt6(df_benchmark, "benchmark (V9.01)")
# print("\n=== Benchmark: Performance by nt6 ===")
# print(nt6_benchmark.to_string(index=False))
