from loguru import logger
import ltv_helpers.pipeline_helpers as ph
import pyspark.sql.functions as F

from classic_spl_ltv import paths as p
from classic_spl_ltv.jobs.expense.process_raw_expense import read_raw_expenses

# Validates that the expense-ratio lookups in the PUBLIC results file match the
# raw Finance file (State Hybrid Expense Ratio), recomputed independently.
#
# Source A = expense_ratio_{bucket}_{term} columns in public_results
# Source B = raw Finance dollars read live via read_raw_expenses(), aggregated
#            by THIS module's own bucket recipes (deliberately NOT reusing
#            add_components -- reusing the pipeline's aggregation would make the
#            check circular).
#
# public bucket -> Finance components summed into the numerator
BUCKET_COMPONENTS = {
    "acquisition": ["other_acq_x_mkt"],
    "overhead":    ["other_overhead"],
    "commission":  ["commission"],
    "service":     ["state_premium_tax", "write_offs", "assessments",
                    "pay_fees", "misc_income_exp", "other_lifetime"],  # = lifetime
    "marketing":   ["lower_funnel_advt", "other_mkt"],
}
# claims excluded: hardcoded constant, no Finance lookup (guarded separately)
CLAIMS_CONSTANT = 0.047832  # source: expenses_raw_2026, tab LTV_SPL, cell GG13

# public channel -> raw-file segment tokens (as read_raw_expenses names them)
CHANNEL_SEGMENTS = {"agt": ["EA", "IA"], "ccc": ["ccc"], "web": ["web"]}

TOL_REL = 1e-3   # 0.1% relative
RATIO_COLS = [f"expense_ratio_{b}_{t}"
              for b in list(BUCKET_COMPONENTS) + ["claims"]
              for t in ("new", "renew")]


def _fin(row, component, segment, term):
    col = f"{component}_{segment}_{term}"
    if col not in row.index:
        raise AssertionError(f"raw expense file is missing column {col}")
    return row[col]


def _expected_ratio(fin_row, components, channel, term):
    segs = CHANNEL_SEGMENTS[channel]
    num = sum(_fin(fin_row, c, s, term) for c in components for s in segs)
    den = sum(_fin(fin_row, "earned_prem", s, term) for s in segs)
    return num / den


def validate_expense_ratios(spark) -> None:
    """Check public-file expense ratios against the raw Finance file, one
    policy per channel present. Raises AssertionError on any mismatch."""
    fin_df = read_raw_expenses()
    assert len(fin_df) == 1, f"expected 1 CW row from read_raw_expenses, got {len(fin_df)}"
    fin_row = fin_df.iloc[0]

    df_pub = ph.read_parquet_s3(spark, p.public_results)

    failures = []
    for channel in CHANNEL_SEGMENTS:
        sample = (
            df_pub.filter(F.col("chnl_bnd") == channel)
                  .select(["adw_pol_id", "chnl_bnd"] + RATIO_COLS)
                  .limit(1)
                  .toPandas()
        )
        if len(sample) == 0:
            logger.warning(f"expense_ratio_validate: no {channel} policy in public file, skipping")
            continue
        row = sample.iloc[0]
        pol_id = row["adw_pol_id"]

        for bucket, components in BUCKET_COMPONENTS.items():
            for term in ("new", "renew"):
                col = f"expense_ratio_{bucket}_{term}"
                actual = row[col]
                expected = _expected_ratio(fin_row, components, channel, term)
                if expected == 0:
                    ok = abs(actual) < 1e-9
                else:
                    ok = abs(actual - expected) <= TOL_REL * abs(expected)
                if ok:
                    logger.info(f"expense_ratio_validate PASS {channel} {bucket}_{term}: {actual:.6f}")
                else:
                    failures.append(
                        f"{channel} (policy {pol_id}) {bucket}_{term}: "
                        f"public={actual:.6f} finance={expected:.6f}"
                    )

        for term in ("new", "renew"):
            actual = row[f"expense_ratio_claims_{term}"]
            if abs(actual - CLAIMS_CONSTANT) > 1e-6:
                failures.append(
                    f"{channel} (policy {pol_id}) claims_{term}: "
                    f"public={actual:.6f} expected constant {CLAIMS_CONSTANT}"
                )

    if failures:
        raise AssertionError(
            "expense ratio validation failed:\n  " + "\n  ".join(failures)
        )
    logger.info("expense_ratio_validate: all channels PASS")
