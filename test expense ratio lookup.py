"""Weekly validation: expense-ratio lookups in the PUBLIC LTV file match Finance.

Source A = expense_ratio_{bucket}_{term} column in public_results (the released file).
Source B = raw Finance dollars, read LIVE from the State Hybrid Expense Ratio file
           via the pipeline's own read_raw_expenses() (row 15 of LTV_SPL), then
           aggregated into buckets by THIS TEST'S OWN math.

Design (deliberate):
  - We reuse ONLY the raw Excel reader from process_raw_expense.py. We do NOT
    reuse add_components / pivot_df / format_cols -- if the test's "expected"
    came from the same aggregation code that produced the public file, the test
    would be circular and prove nothing. The bucket recipes below are the
    independent check.
  - Never uses the *_exp dollar columns (premium * ratio -> circular).

PUBLIC SCHEMA NOTES (differ from internal e006scl_* names):
  - prefix `e006scl_` dropped -> expense_ratio_acquisition_new, ...
  - internal `lifetime` bucket is named `service` in the public file
  - internal `commissions` is `commission` (singular) in the public file
  - key is `adw_pol_id`, channel is `chnl_bnd` (both repos, post-rename)
"""
import importlib.util
import sys

import pytest
import pyspark.sql.functions as F

TOL_REL = 1e-3   # relative: 0.1% -- honest across ratios spanning 0.001 to 0.67

# path to the pipeline's raw-expense reader (classic repo owns the Finance parse)
PROCESS_RAW_EXPENSE_PATH = (
    "/mnt/imported/code/classic-specialty-ltv/classic_spl_ltv/jobs/expense/"
    "process_raw_expense.py"
)

# --------------------------------------------------------------------------
# bucket recipes -- the test's INDEPENDENT math (public bucket -> components)
# --------------------------------------------------------------------------
PUBLIC_BUCKET_COMPONENTS = {
    "acquisition": ["other_acq_x_mkt"],
    "overhead":    ["other_overhead"],
    "commission":  ["commission"],                        # singular, per prefixes list
    "service":     ["state_premium_tax", "write_offs", "assessments",  # = lifetime
                    "pay_fees", "misc_income_exp", "other_lifetime"],
    "marketing":   ["lower_funnel_advt", "other_mkt"],
}
# claims excluded: hardcoded constant, no Finance lookup (guarded separately)

# raw-file channel tokens, exactly as read_raw_expenses builds them
CHANNEL_SEGMENTS = {"agt": ["EA", "IA"], "web": ["web"], "ccc": ["ccc"]}
CLAIMS_CONSTANT = 0.047832   # source: expenses_raw_2026, tab LTV_SPL, cell GG13

# (id, s3_path, policy_id, expected_channel)
CASES = [
    ("classic",
     "tmx-smsiweb/classic-specialty-ltv/prod/release/NB_Classic_SPL_v9.2.0/score/public_results/",
     "1844586908208", "agt"),
    ("renter",
     "tmx-smsiweb/specialty-ltv/prod/release/NB_SPL_v1.3.0/ltv_calc/public_results/",
     "5778686908048", "web"),
]


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def spark():
    from ltv_helpers.spark import create_spark
    from specialty_ltv import paths as p
    import ltv_helpers.non_spark_helpers as nsh
    nsh.initialize_config_path(p)
    return create_spark(buckets=p.buckets, app_name="expense_ratio_test")


@pytest.fixture(scope="session")
def finance_raw():
    """One-row DataFrame of raw Finance dollars, read live from the Excel file
    by the pipeline's own reader. Columns: {prefix}_{channel}_{type}."""
    spec = importlib.util.spec_from_file_location("pre", PROCESS_RAW_EXPENSE_PATH)
    pre = importlib.util.module_from_spec(spec)
    sys.modules["pre"] = pre
    spec.loader.exec_module(pre)          # runs top-level config init; main() is guarded
    df = pre.read_raw_expenses()
    assert len(df) == 1, f"expected 1 CW row from read_raw_expenses, got {len(df)}"
    return df.iloc[0]


def fin_value(row, component, segment, term):
    col = f"{component}_{segment}_{term}"
    assert col in row.index, f"raw expense file is missing column {col}"
    return row[col]


def expected_ratio(row, components, channel, term):
    segs = CHANNEL_SEGMENTS[channel]
    num = sum(fin_value(row, c, s, term) for c in components for s in segs)
    den = sum(fin_value(row, "earned_prem", s, term) for s in segs)
    return num / den


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def policy_row(request, spark):
    import ltv_helpers.pipeline_helpers as ph
    _, path, pol_id, channel = request.param
    df = ph.read_parquet_s3(spark, path)
    row = (
        df.filter((F.col("adw_pol_id") == pol_id) & (F.col("days_after_eff_date") == 28))
          .toPandas()
    )
    assert len(row) == 1, f"expected exactly 1 record for {pol_id}, got {len(row)}"
    assert row["chnl_bnd"].iloc[0] == channel, \
        f"channel drifted: expected {channel}, got {row['chnl_bnd'].iloc[0]}"
    return channel, row.iloc[0]


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
@pytest.mark.parametrize("term", ["new", "renew"])
@pytest.mark.parametrize("bucket,components", PUBLIC_BUCKET_COMPONENTS.items())
def test_expense_ratio_lookup(policy_row, finance_raw, bucket, components, term):
    channel, row = policy_row
    col = f"expense_ratio_{bucket}_{term}"
    assert col in row.index, f"public file is missing {col} (schema changed?)"
    actual = row[col]
    expected = expected_ratio(finance_raw, components, channel, term)
    if expected == 0:
        assert actual == pytest.approx(0, abs=1e-9), \
            f"{channel} {bucket}_{term}: finance says 0, public={actual}"
    else:
        assert actual == pytest.approx(expected, rel=TOL_REL), \
            f"{channel} {bucket}_{term}: public={actual} finance={expected}"


def test_renter_commission_is_zero(policy_row):
    channel, row = policy_row
    if channel == "web":                       # renters carry no commission lookup
        assert row["expense_ratio_commission_new"] == 0
        assert row["expense_ratio_commission_renew"] == 0


@pytest.mark.parametrize("term", ["new", "renew"])
def test_claims_is_constant(policy_row, term):
    # claims is a hardcoded constant, not a Finance lookup; guard against silent change.
    _, row = policy_row
    assert row[f"expense_ratio_claims_{term}"] == pytest.approx(CLAIMS_CONSTANT, abs=1e-6)
