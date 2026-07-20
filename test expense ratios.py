"""
Validation: expense-ratio lookups in the INTERNAL LTV file match Finance.

Finance numbers are hard-coded below (FINANCE_RATIOS), independent of
read_raw_expenses(), so a reader bug cannot mask a bad internal-file ratio.

VERSION COUPLING: FINANCE_RATIOS must be the SAME finance version as the file
under test. These are the v3 channel ratios, so the target file must be a
v3-scored release (NB_SPL_v1.4.1). Validating a pre-v3 file (e.g. v1.3.0) will
fail the *value* asserts even though the *schema* checks pass, because v3
reallocated channel expense.
"""

import pytest

from specialty_ltv import paths as p

TOL = 1e-4  # how far the file ratio may drift from Finance

EXPECTED_RELEASE = "NB_SPL_v1.4.1"  # fail-closed guard; bump when the target moves

# Internal file scale prefix. CONFIRMED e006scl on the v1.3.0 internal file
# (columns are expense_ratio_e006scl_*). Re-confirm on v1.4.1: if the projected
# read below KeyErrors on expense_ratio_<PREFIX>_acquisition_new, the prefix
# moved -- print df.columns and update this one line.
PREFIX = "e006scl"

# Finance ratios, sheet LTV_SPL row 15, keyed by channel. Bucket keys match the
# INTERNAL file columns (acquisition / lifetime / marketing / overhead). Internal
# 'lifetime' == Finance/public 'service' -- the service numbers live under the
# 'lifetime' key on purpose. Each value is (new, renew).
#
# !!! NUMBERS BELOW ARE TRANSCRIBED FROM YOUR SCREENSHOT -- re-paste from your
# working copy and confirm they are the v3 block. Only intended change vs your
# file is the "service" -> "lifetime" key rename.
FINANCE_RATIOS = {
    "agt": {
        "acquisition": (0.38474479410655, 0.00185425844556286),
        "commission":  (0.0738125138575392, 0.0729858619065783),
        "lifetime":    (0.0391282074606354, 0.0257909519791697),  # was "service"
        "marketing":   (0.0833125134697646, 0.0057195521456162),
        "overhead":    (0.0744555484982607, 0.0562542816065963),
    },
    "ccc": {
        "acquisition": (0.645680065885614, 0.0375672474345244),
        "commission":  (0.0, 0.0),
        "lifetime":    (0.064735283399428, 0.0441525757686819),   # was "service"
        "marketing":   (0.393693691758236, 0.0283294139930426),
        "overhead":    (0.127605512682649, 0.0679494901864043),
    },
    "web": {
        "acquisition": (0.077280121325849, 0.00746423025400954),
        "commission":  (0.0, 0.0),
        "lifetime":    (0.077415864038282, 0.0523055859869506),   # was "service"
        "marketing":   (0.807626119725766, 0.0841251697048437),
        "overhead":    (0.172368925331663, 0.12042436116669),
    },
}

BUCKETS = ["acquisition", "lifetime", "marketing", "overhead"]  # commission absent from lookup
TERMS = ["new", "renew"]

# (case_id, adw_pol_id, expected_channel) -- one policy that binds to each channel.
# Path is shared (single source of truth in the fixture), so it lives once.
CASE = [
    ("renter_ccc", "10000008833348", "ccc"),
    ("renter_agt", "6356186909914", "agt"),
]

# The only columns we need. Projecting these is what stops the OOM: ~14 cols
# instead of the full ~400-col internal file.
_ratio_cols = [f"expense_ratio_{PREFIX}_{b}_{t}" for b in BUCKETS for t in TERMS]
_claims_cols = [f"expense_ratio_{PREFIX}_claims_{t}" for t in TERMS]
_commission_cols = [f"expense_ratio_{PREFIX}_commission_{t}" for t in TERMS]  # singular
SEL_COLS = ["adw_pol_id", "drv_chnl_of_bnd", *_ratio_cols, *_claims_cols, *_commission_cols]


# ---- fixtures ---------------------------------------------------------------
@pytest.fixture(scope="session")
def claims_constant():
    return p.claims_expense_ratio  # golden ref from paths.toml, independent of the file


@pytest.fixture(scope="module", params=CASE, ids=[c[0] for c in CASE])
def policy_row(request):
    import ltv_helpers.non_spark_helpers as nsh

    nsh.initialize_config_path(p)

    path = p.score_internal_results
    assert EXPECTED_RELEASE in path, (
        f"expected {EXPECTED_RELEASE}, resolved to {path}. Fix the dynaconf "
        "checkout/version (or confirm the v1.4.1 file exists in S3) before "
        "trusting any result -- a green run on the wrong release is a false pass."
    )

    _, pol_id, channel = request.param

    # NOTE: assumes nsh.read_parquet_s3_to_pandas accepts columns=. If it does
    # not, this projection can't happen through nsh -- see the reply.
    df = nsh.read_parquet_s3_to_pandas(path, columns=SEL_COLS)

    mask = df["adw_pol_id"].astype(str) == str(pol_id)
    hits = df[mask]
    assert len(hits) == 1, f"expected exactly 1 record for {pol_id}, got {len(hits)}"

    row = hits.iloc[0]
    assert row["drv_chnl_of_bnd"] == channel, (
        f"channel drifted: expected {channel}, got {row['drv_chnl_of_bnd']}"
    )
    return channel, row


# ---- tests ------------------------------------------------------------------
@pytest.mark.parametrize("term", TERMS)
@pytest.mark.parametrize("bucket", BUCKETS)
def test_expense_ratio_lookup(policy_row, bucket, term):
    channel, row = policy_row
    col = f"expense_ratio_{PREFIX}_{bucket}_{term}"
    assert col in row.index, f"internal file is missing {col} (schema changed?)"
    actual = row[col]
    expected = FINANCE_RATIOS[channel][bucket][0 if term == "new" else 1]
    assert actual == pytest.approx(expected, abs=TOL), (
        f"{channel} {bucket} {term}: file={actual} finance={expected}"
    )


def test_web_commission_is_zero(policy_row):
    channel, row = policy_row
    if channel == "web":
        assert row[f"expense_ratio_{PREFIX}_commission_new"] == 0
        assert row[f"expense_ratio_{PREFIX}_commission_renew"] == 0


@pytest.mark.parametrize("term", TERMS)
def test_claims_is_constant(policy_row, term, claims_constant):
    _, row = policy_row
    col = f"expense_ratio_{PREFIX}_claims_{term}"
    assert row[col] == pytest.approx(claims_constant, abs=1e-6)
