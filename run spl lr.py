"""
Driver for the SPL / renters loss ratio refit (Classic v9.4.0, Renters v1.5.0).

Run top to bottom in a notebook, or as a script. Every step that could fail
silently prints a check instead.

    Classic:  RELEASE = "classic"
    Renters:  RELEASE = "renters"

The fitting logic is identical for both. Only the line config differs, which
is what makes this "one notebook change applied to both".
"""

import pandas as pd

import spl_loss_ratio as lr

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

# ---------------------------------------------------------------------------
# 0. Config -- the only block that changes between releases
# ---------------------------------------------------------------------------

RELEASE = "classic"          # "classic" | "renters"
CAT_MODE = "ex_cat"          # headline run; reconciliation covers all three
BASE_DIR = "/mnt/data/eltv-policy/eltv-notebooks-d/ngraf/spec_lines_prop_lr/CDF_extr"

FILE_BASES = [
    "ssenp_20260729-162408",   # replace with the actual extract set
]

# ACTYR appears to be years since 1900 (123 = 2023). CONFIRM WITH NICK before
# relying on it. Set to None to keep every year in the extract.
ACTYR_KEEP = None            # e.g. [124, 125, 126] for 2024 -> present

LINE_SPEC = (
    lr.LINE_SPEC_CLASSIC if RELEASE == "classic" else lr.LINE_SPEC_RENTERS
)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

prem = lr.load_extract(BASE_DIR, FILE_BASES, "p")
clm = lr.load_extract(BASE_DIR, FILE_BASES, "c")

print(f"premium rows: {len(prem):,}   claims rows: {len(clm):,}")
print("lines in extract:", sorted(prem["ALINE"].unique()))
print("ACTYR in extract:", sorted(prem["ACTYR"].unique()))

missing_lines = set(LINE_SPEC) - set(prem["ALINE"].unique())
if missing_lines:
    print(f"\n!! extract is missing configured lines: {sorted(missing_lines)}")
    print("!! re-pull before releasing -- do not fit on a partial extract\n")

if ACTYR_KEEP is not None:
    prem = prem[prem["ACTYR"].isin(ACTYR_KEEP)]
    clm = clm[clm["ACTYR"].isin(ACTYR_KEEP)]
    print(f"filtered to ACTYR {ACTYR_KEEP}: {len(prem):,} / {len(clm):,} rows")

# ---------------------------------------------------------------------------
# 2. Cat indicator sanity checks -- run before trusting anything downstream
# ---------------------------------------------------------------------------

cat_check = lr.check_cat_split(clm)
for k, v in cat_check.items():
    print(f"  {k}: {v}")

assert cat_check["additive"], "all != ex_cat + cat_only -- cat filter is wrong"
if not cat_check["grain_changed"]:
    print("\n  note: CATCD did not split any cells. Either no cell mixes cat "
          "and non-cat, or the extract was pre-aggregated. Worth a look.\n")

# ---------------------------------------------------------------------------
# 3. Build modelling frame
# ---------------------------------------------------------------------------

claims_agg = lr.aggregate_claims(clm, mode=CAT_MODE)
cells, audit = lr.build_cells(prem, claims_agg)

print("\n--- join audit ---")
for k, v in audit.items():
    print(f"  {k}: {v}")

# Loss sitting on zero-premium cells is invisible to every fit, because the
# weight is premium. Surface it rather than discovering it later.
if audit["stranded_loss_share"] > 0.01:
    print(f"\n  !! {audit['stranded_loss_share']:.1%} of loss sits on "
          "zero-premium cells and is excluded from all fits. Raise with Yinan.")

# ---------------------------------------------------------------------------
# 4. Reconciliation -- what changed in losses (the Steven-facing number)
# ---------------------------------------------------------------------------

recon_line = lr.reconcile_cat(prem, clm, by=["ALINE"])
print("\n--- cat reconciliation by line ---")
print(recon_line.round(4).to_string(index=False))

recon_ntr = lr.reconcile_cat(prem, clm, by=["ALINE", "NTR"])

# ---------------------------------------------------------------------------
# 5. Fit
# ---------------------------------------------------------------------------

fits = lr.fit_all(cells, LINE_SPEC, cat_mode=CAT_MODE)

print("\n--- fits ---")
print(lr.fits_frame(fits).round(6).to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Compare to production
# ---------------------------------------------------------------------------

cmp_df = lr.compare_to_production(fits)
print("\n--- new vs production, by NTR ---")
print(cmp_df.round(4).to_string(index=False))

print("\n--- premium-weighted impact by line ---")
impact = (
    cmp_df.merge(
        recon_ntr[["ALINE", "NTR", "premium"]],
        left_on=["line", "NTR"], right_on=["ALINE", "NTR"], how="left",
    )
    .assign(dollar_delta=lambda d: d["delta"] * d["premium"])
    .groupby(["line", "label"], as_index=False)
    .agg(premium=("premium", "sum"), dollar_delta=("dollar_delta", "sum"))
    .assign(lr_pt_change=lambda d: d["dollar_delta"] / d["premium"])
)
print(impact.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Sensitivity: does the line 78 Florida carve-out still earn its keep?
# ---------------------------------------------------------------------------

if 78 in fits:
    with_fl = lr.fit_line(cells, 78, lr.LineSpec("all_points", (), "Condo"),
                          cat_mode=CAT_MODE)
    without_fl = fits[78]
    print("\n--- line 78 Florida sensitivity (post cat removal) ---")
    print(f"  FL included: slope={with_fl.slope:.5f} "
          f"intercept={with_fl.intercept:.5f} wavg={with_fl.wavg_lr:.4f}")
    print(f"  FL excluded: slope={without_fl.slope:.5f} "
          f"intercept={without_fl.intercept:.5f} wavg={without_fl.wavg_lr:.4f}")
    print("  If these have converged, the carve-out was cat-driven and can go.")

# ---------------------------------------------------------------------------
# 8. Emit for the pipeline
# ---------------------------------------------------------------------------

print("\n--- paste into pipeline ---")
print(lr.emit_spl_fits(fits))
print()
print(lr.emit_lr_tuples(fits))
