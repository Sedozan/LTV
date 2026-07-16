import pytest

TOL = 1e-4  # tolerance for how far the public-file ratio may differ from Finance

# ---------------------------------------------------------------------------
# Golden reference: Finance expense ratios, copied directly from the Finance
# summary workbook ("2026_SPL_exp_summary" / the "v3 file" block). These are the
# already-aggregated per-channel ratios, so when Finance ships a new file you
# replace this block with the new numbers from that same tab -- no raw dollar
# figures, no re-aggregation.
#
# Finance labels the bucket "lifetime"; the public file calls it "service".
# Claims is NOT here: it is a flat constant validated separately against
# paths.claims_expense_ratio (see test_claims_is_constant).
#
# VERIFY before committing: these were transcribed from a screenshot. Re-copy
# each value from the source cells (full precision, not the rounded % block).
#
# channel -> bucket -> (new, renew)
# ---------------------------------------------------------------------------
FINANCE_RATIOS = {
    "agt": {
        "acquisition": (0.384744794, 0.001854258),
        "commission":  (0.073812514, 0.072985860),
        "service":     (0.039128300, 0.025791000),
        "marketing":   (0.083312510, 0.005719600),
        "overhead":    (0.074455548, 0.056254280),
    },
    "ccc": {
        "acquisition": (0.645680066, 0.037567247),
        "commission":  (0.000000000, 0.000000000),
        "service":     (0.064735300, 0.044152600),
        "marketing":   (0.393693690, 0.028329400),
        "overhead":    (0.127605513, 0.067949490),
    },
    "web": {
        "acquisition": (0.077280121, 0.007464230),
        "commission":  (0.000000000, 0.000000000),
        "service":     (0.077741600, 0.052305600),
        "marketing":   (0.807626120, 0.084125200),
        "overhead":    (0.172368925, 0.120424360),
    },
}

BUCKETS = ["acquisition", "commission", "service", "marketing", "overhead"]

# (case id, repo_label, s3_path, policy id, expected_channel)
CASE = [
    (
        "classic",
        "classic-specialty-ltv",
        "tmx-smsiweb/classic-specialty-ltv/prod/release/NB_Classic_SPL_v9.2.0/score/public_results/",
        "2598286908851",
        "agt",
    ),
]


# fixtures
@pytest.fixture(scope="session")
def claims_constant():
    from classic_spl_ltv import paths as p

    return p.claims_expense_ratio


@pytest.fixture(scope="module", params=CASE, ids=[c[0] for c in CASE])
def policy_row(request):
    import ltv_helpers.non_spark_helpers as nsh
    from classic_spl_ltv import paths as p

    nsh.initialize_config_path(p)

    _, _, path, pol_id, channel = request.param
    df = nsh.read_parquet_s3_to_pandas(path)

    mask = (df["adw_pol_id"].astype(str) == str(pol_id)) & (
        df["days_after_eff_date"] == 28
    )
    row = df[mask]
    assert len(row) == 1, f"expected exactly 1 record for {pol_id}, got {len(row)}"
    assert (
        row["chnl_bnd"].iloc[0] == channel
    ), f"channel drifted: expected {channel}, got {row['chnl_bnd'].iloc[0]}"
    return channel, row.iloc[0]


# tests
@pytest.mark.parametrize("term", ["new", "renew"])
@pytest.mark.parametrize("bucket", BUCKETS)
def test_expense_ratio_lookup(policy_row, bucket, term):
    channel, row = policy_row
    col = f"expense_ratio_{bucket}_{term}"
    assert col in row.index, f"public file is missing {col} (schema changed?)"
    actual = row[col]
    expected = FINANCE_RATIOS[channel][bucket][0 if term == "new" else 1]
    assert actual == pytest.approx(expected, abs=TOL), (
        f"{channel} {bucket} {term}: public={actual} finance={expected}"
    )


def test_web_commission_is_zero(policy_row):
    channel, row = policy_row
    if channel == "web":
        assert row["expense_ratio_commission_new"] == 0
        assert row["expense_ratio_commission_renew"] == 0


@pytest.mark.parametrize("term", ["new", "renew"])
def test_claims_is_constant(policy_row, term, claims_constant):
    _, row = policy_row
    assert row[f"expense_ratio_claims_{term}"] == pytest.approx(
        claims_constant, abs=1e-6
    )
