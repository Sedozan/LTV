"""
Validation: expense-ratio lookups in the PUBLIC LTV file match Finance.

Unit-test style (pytest), no Spark. Finance numbers are HARD-CODED below
(FIN), so a bug in read_raw_expenses() cannot mask a bad public-file ratio --
the golden reference is independent of the pipeline reader.
"""

import pytest

TOL = 1e-4  # tolerance of how far the ratios can differ

# Finance file, sheet LTV_SPL, row 15 (SPL v1.3.0 / Classic v9.2.0)
# each component -> {segment: (new, renew)}
# NOTE: release-varying. Must be refreshed from the Finance excel each release.
FIN = {
    "earned_prem": {
        "EA": (446836970, 2325310435),
        "IA": (8130333, 160350749),
        "WEB": (57401898, 41073401),
    },
    "other_acq_x_mkt": {
        "EA": (144407671, 3818976),
        "IA": (105409, 2788),
        "WEB": (3037449, 291655),
    },
    "other_overhead": {
        "EA": (34599354, 143942307),
        "IA": (600272, 2497287),
        "WEB": (8515334, 5943023),
    },
    "commissions": {
        "EA": (33163547, 172580936),
        "IA": (527984, 10413180),
        "WEB": (0, 0),
    },
    "state_premium_tax": {
        "EA": (9990303, 51988886),
        "IA": (159052, 3136903),
        "WEB": (1496442, 1070765),
    },
    "write_offs": {
        "EA": (1294960, 6738887),
        "IA": (20617, 406611),
        "WEB": (193971, 138794),
    },
    "assessments": {
        "EA": (279904, 1456603),
        "IA": (4456, 87888),
        "WEB": (41927, 30000),
    },
    "pay_fees": {
        "EA": (-7480277, -38926873),
        "IA": (-119091, -2348768),
        "WEB": (-1120467, -801740),
    },
    "misc_income_exp": {
        "EA": (-589889, -3069744),
        "IA": (-9391, -185222),
        "WEB": (-88359, -63225),
    },
    "other_lifetime": {
        "EA": (15114122, 50716166),
        "IA": (315518, 1058736),
        "WEB": (3420887, 2216578),
    },
    "lower_funnel_advt": {
        "EA": (22578399, 1963339),
        "IA": (684496, 59521),
        "WEB": (33791771, 2938415),
    },
    "other_mkt": {
        "EA": (17221608, 14885359),
        "IA": (415619, 359237),
        "WEB": (4764191, 1225317),
    },
}

# public bucket name -> Finance component keys (note: commission -> "commissions")
PUBLIC_BUCKET_COMPONENTS = {
    "acquisition": ["other_acq_x_mkt"],
    "overhead": ["other_overhead"],
    "commission": ["commissions"],
    "service": [
        "state_premium_tax",
        "write_offs",
        "assessments",
        "pay_fees",
        "misc_income_exp",
        "other_lifetime",
    ],
    "marketing": ["lower_funnel_advt", "other_mkt"],
}

CHANNEL_SEGMENTS = {"agt": ["EA", "IA"], "web": ["WEB"], "ccc": ["CCC"]}

# (id, repo_label, s3_path, policy_id, expected_channel)
# NOTE: policy_id must exist at days_after_eff_date==28 in that release's public
# file. Release-varying -- update with FIN each release.
CASES = [
    (
        "classic",
        "classic-specialty-ltv",
        "tmx-smsiweb/classic-specialty-ltv/prod/release/NB_Classic_SPL_v9.2.0/score/public_results/",
        "1844586908208",
        "agt",
    ),
    (
        "renter",
        "specialty-ltv",
        "tmx-smsiweb/specialty-ltv/prod/release/NB_SPL_v1.3.0/ltv_calc/public_results/",
        "5778686908048",
        "web",
    ),
]


def expected_ratio(components, channel, term):
    segs = CHANNEL_SEGMENTS[channel]
    i = 0 if term == "new" else 1
    num = sum(FIN[c][s][i] for c in components for s in segs)
    den = sum(FIN["earned_prem"][s][i] for s in segs)
    return num / den


# fixtures
@pytest.fixture(scope="session")
def claims_constant():
    # moved to config per PR review (release-varying). Confirm the attribute
    # name matches the config, and have process_raw_expense.py read the SAME
    # value so there is a single source of truth.
    from classic_spl_ltv import paths as p

    return p.claims_expense_ratio


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def policy_row(request):
    import ltv_helpers.non_spark_helpers as nsh
    from classic_spl_ltv import paths as p

    nsh.initialize_config_path(p)  # keep if the reader needs config initialized

    _, _, path, pol_id, channel = request.param
    df = nsh.read_parquet_s3_to_pandas(path)

    # cast both sides to str: public adw_pol_id may be int64 while CASES id is str
    mask = (df["adw_pol_id"].astype(str) == str(pol_id)) & (
        df["days_after_eff_date"] == 28
    )
    row = df[mask]
    assert len(row) == 1, f"expected exactly 1 record for {pol_id}, got {len(row)}"
    assert row["chnl_bnd"].iloc[0] == channel, (
        f"channel drifted: expected {channel}, got {row['chnl_bnd'].iloc[0]}"
    )
    return channel, row.iloc[0]


# tests
@pytest.mark.parametrize("term", ["new", "renew"])
@pytest.mark.parametrize("bucket,components", PUBLIC_BUCKET_COMPONENTS.items())
def test_expense_ratio_lookup(policy_row, bucket, components, term):
    channel, row = policy_row
    col = f"expense_ratio_{bucket}_{term}"
    assert col in row.index, f"public file is missing {col} (schema changed?)"
    actual = row[col]
    expected = expected_ratio(components, channel, term)
    assert actual == pytest.approx(
        expected, abs=TOL
    ), f"{channel} {bucket}_{term}: public={actual} finance={expected}"


def test_renter_commission_is_zero(policy_row):
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
