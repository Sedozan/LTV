"""
SPL / renters loss ratio vs NTR fitting, with catastrophe exclusion.

Reusable replacement for ngraf's spl_lr.ipynb. Both releases import this module
and call it with different config:

    Classic SPL v9.4.0 -> LINE_SPEC_CLASSIC (lines 16, 32, 71, 72, 78, 88, 90)
    Renters   v1.5.0   -> LINE_SPEC_RENTERS (line 71 only)

Pipeline order matters. CATCD adds grain to the claims extract (a cell with both
cat and non-cat losses is now two or more rows). Therefore:

    filter cat  ->  re-aggregate claims to cell grain  ->  merge to premium

Filtering after the merge, or leaving CATCD in the join keys, fans out premium
rows and silently inflates exposure.

Author: SB
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cell-level dimension keys shared by the premium and claims extracts.
#: CATCD is deliberately NOT a key -- it is collapsed out before the merge.
KEYS: list[str] = ["ACTMO", "ACTYR", "ALINE", "CLINE", "COMPNY", "GEOST", "NTR"]

#: Money columns on the claims side. CMLOSS = indemnity, CMEXP = claim
#: adjustment expense. TOTLOSS = CMEXP + CMLOSS (matches production).
CLAIM_COLS: list[str] = ["CMEXP", "CMLOSS"]

#: GEOST code for Florida (see state_table in the original notebook).
FLORIDA = 9

#: NTR is a censored bucket: the top value means "N or more", not literally N.
#: Recorded here so the assumption is visible rather than buried in the fit.
NTR_TOP_BUCKET = 9

CatMode = Literal["all", "ex_cat", "cat_only"]
FitMode = Literal["all_points", "ntr_gt0"]


# ---------------------------------------------------------------------------
# Per-line configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineSpec:
    """How one specialty line should be fit.

    fit_mode
        "all_points" -- single WLS line through every NTR bucket.
        "ntr_gt0"    -- WLS line fit on NTR > 0 only; NTR = 0 gets its own
                        standalone weighted-average value. Used where new
                        business behaves differently enough to drag the slope.
    exclude_states
        GEOST codes dropped before fitting. Line 78 historically excluded
        Florida; that carve-out is cat-motivated and must be revisited once
        cat losses are removed from the numerator.
    label
        Human-readable name for plots and the reconciliation table.
    """

    fit_mode: FitMode = "all_points"
    exclude_states: tuple[int, ...] = ()
    label: str = ""


#: Baseline reproducing ngraf's production behaviour. Change deliberately,
#: not by accident -- these choices are the judgment call Yinan wants to make.
LINE_SPEC_CLASSIC: dict[int, LineSpec] = {
    16: LineSpec("ntr_gt0", (), "Specialty Auto"),
    32: LineSpec("all_points", (), "Mobile Home"),
    71: LineSpec("all_points", (), "Renters"),
    72: LineSpec("all_points", (), "Landlord"),
    # REVISIT: FL exclusion predates cat filtering. If the FL gap was
    # cat-driven, keeping this double-counts the fix.
    78: LineSpec("all_points", (FLORIDA,), "Condo"),
    88: LineSpec("ntr_gt0", (), "PUP / Umbrella"),
    90: LineSpec("ntr_gt0", (), "Boat"),
}

LINE_SPEC_RENTERS: dict[int, LineSpec] = {
    71: LineSpec("all_points", (), "Renters"),
}

#: Production fits currently in the pipeline, for side-by-side comparison.
#: [slope, intercept] or [slope, intercept, ntr0_value].
PROD_FITS_V7: dict[int, list[float]] = {
    16: [-0.02294, 0.6347, 1.0057],
    32: [-0.01407, 0.6206],
    71: [-0.05045, 0.7259],
    72: [-0.003477, 0.6060],
    78: [-0.03717, 0.82678],
    88: [0.0, 0.7571, 1.2079],
    90: [-0.02329, 0.7527, 0.9611],
}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_extract(
    base_dir: str,
    file_bases: Sequence[str],
    side: Literal["p", "c"],
    dtype: dict | None = None,
) -> pd.DataFrame:
    """Read and concatenate one side of the CDF extract.

    ``side="p"`` reads ``{base}p.csv`` (premium: EEXP, EPREM).
    ``side="c"`` reads ``{base}c.csv`` (claims: CATCD, CMEXP, CMLOSS).

    Lists are rebuilt on every call. The original notebook appended auto onto
    the property lists because it reused ``df1_list`` across cells; that
    silently double-counted on re-run.
    """
    frames = []
    for base in file_bases:
        path = os.path.join(base_dir, f"{base}{side}.csv")
        df = pd.read_csv(path, dtype=dtype)
        df["_source_file"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        raise ValueError("no files loaded -- check base_dir and file_bases")
    out = pd.concat(frames, ignore_index=True)

    missing = [k for k in KEYS if k not in out.columns]
    if missing:
        raise KeyError(f"{side}-side extract missing key columns: {missing}")
    return out


# ---------------------------------------------------------------------------
# Catastrophe handling
# ---------------------------------------------------------------------------


def flag_cat(claims: pd.DataFrame, cat_col: str = "CATCD") -> pd.Series:
    """Boolean: True where the row carries a catastrophe code.

    CATCD is a single alpha character (A-Z, skipping I and O) or blank.
    Non-blank is treated as cat. Blank/NaN is non-cat.

    Caveat to confirm with Nick: this assumes *every* non-blank code is a true
    catastrophe. If some codes are non-cat weather or administrative, this
    over-removes.

    Because only 24 codes exist they must recycle across accident years, so a
    code alone cannot identify an event. Do not attempt event-level analysis
    without a companion cat-year/cat-date field.
    """
    if cat_col not in claims.columns:
        raise KeyError(
            f"{cat_col} not in claims extract -- cannot filter cat. "
            "Re-pull with the cat indicator included."
        )
    return claims[cat_col].fillna("").astype(str).str.strip().ne("")


def aggregate_claims(
    claims: pd.DataFrame,
    mode: CatMode = "ex_cat",
    cat_col: str = "CATCD",
) -> pd.DataFrame:
    """Filter by cat status, then collapse back to one row per cell.

    This is the step that makes the merge safe. Returns KEYS + CLAIM_COLS.
    """
    if mode not in ("all", "ex_cat", "cat_only"):
        raise ValueError(f"bad mode: {mode!r}")

    is_cat = flag_cat(claims, cat_col)
    if mode == "ex_cat":
        subset = claims.loc[~is_cat]
    elif mode == "cat_only":
        subset = claims.loc[is_cat]
    else:
        subset = claims

    subset = subset.copy()
    # CSV reads can hand back object dtype (thousands separators, stray blanks).
    # Coerce explicitly -- silent object-dtype sums are a real failure mode.
    for col in CLAIM_COLS:
        subset[col] = pd.to_numeric(subset[col], errors="coerce").fillna(0.0)

    return (
        subset.groupby(KEYS, as_index=False)[CLAIM_COLS]
        .sum()
        .reset_index(drop=True)
    )


def check_cat_split(claims: pd.DataFrame, cat_col: str = "CATCD") -> dict:
    """Sanity checks on the cat indicator. Run before trusting any fit.

    Returns a dict of diagnostics:
      grain_changed  -- did CATCD actually split cells into multiple rows?
      additive       -- does all == ex_cat + cat_only to the cent?
      cat_share_*    -- how much of the loss the filter removes.
    """
    is_cat = flag_cat(claims, cat_col)
    n_rows = len(claims)
    n_cells = claims[KEYS].drop_duplicates().shape[0]

    tot_all = aggregate_claims(claims, "all", cat_col)[CLAIM_COLS].sum()
    tot_ex = aggregate_claims(claims, "ex_cat", cat_col)[CLAIM_COLS].sum()
    tot_cat = aggregate_claims(claims, "cat_only", cat_col)[CLAIM_COLS].sum()

    additive = bool(
        np.allclose(
            tot_all.astype(float).values,
            (tot_ex + tot_cat).astype(float).values,
            atol=0.01,
        )
    )

    denom = float(tot_all["CMLOSS"]) or np.nan
    return {
        "n_rows": n_rows,
        "n_cells": n_cells,
        "grain_changed": n_rows > n_cells,
        "n_cat_rows": int(is_cat.sum()),
        "cat_codes": sorted(
            claims.loc[is_cat, cat_col].astype(str).str.strip().unique().tolist()
        ),
        "additive": additive,
        "cmloss_all": float(tot_all["CMLOSS"]),
        "cmloss_ex_cat": float(tot_ex["CMLOSS"]),
        "cmloss_cat_only": float(tot_cat["CMLOSS"]),
        "cat_share_cmloss": float(tot_cat["CMLOSS"]) / denom,
        "cmexp_all": float(tot_all["CMEXP"]),
        "cat_share_cmexp": float(tot_cat["CMEXP"]) / (float(tot_all["CMEXP"]) or np.nan),
    }


# ---------------------------------------------------------------------------
# Build the modelling frame
# ---------------------------------------------------------------------------


def build_cells(
    premium: pd.DataFrame,
    claims_agg: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Merge premium and cat-filtered claims, compute loss ratio and weights.

    Returns (cells, audit).

    Weight is earned premium. Zero-premium cells therefore contribute nothing
    to the fit, which means any claims sitting on a zero-premium cell are
    dropped from the analysis entirely. The audit dict quantifies that leak
    instead of letting it pass silently, which the original notebook did.
    """
    cells = premium.merge(
        claims_agg, on=KEYS, how="outer", indicator=True, validate="one_to_one"
    )

    audit = {
        "merge_counts": cells["_merge"].value_counts().to_dict(),
        # Claims with no matching premium row -> zero weight -> invisible.
        "orphan_claim_rows": int((cells["_merge"] == "right_only").sum()),
        "orphan_claim_cmloss": float(
            cells.loc[cells["_merge"] == "right_only", "CMLOSS"].fillna(0).sum()
        ),
    }

    for col in ["EEXP", "EPREM", "CMEXP", "CMLOSS"]:
        if col in cells.columns:
            cells[col] = cells[col].fillna(0.0)

    cells["TOTLOSS"] = cells["CMEXP"] + cells["CMLOSS"]

    # (EPREM > 0) guard reproduces production: zero-premium cells get LR 0.
    cells["loss_ratio"] = np.where(
        cells["EPREM"] > 0, cells["TOTLOSS"] / cells["EPREM"], 0.0
    )
    cells["w"] = cells["EPREM"]

    # Loss stranded on zero-premium cells, i.e. excluded from every fit.
    zero_prem = cells["EPREM"] <= 0
    audit["zero_premium_cells"] = int(zero_prem.sum())
    audit["zero_premium_cmloss"] = float(cells.loc[zero_prem, "TOTLOSS"].sum())
    audit["total_cmloss"] = float(cells["TOTLOSS"].sum())
    audit["stranded_loss_share"] = (
        audit["zero_premium_cmloss"] / audit["total_cmloss"]
        if audit["total_cmloss"]
        else np.nan
    )
    audit["ntr_max"] = int(cells["NTR"].max())
    audit["lines_present"] = sorted(cells["ALINE"].unique().tolist())
    audit["actyr_present"] = sorted(cells["ACTYR"].unique().tolist())

    return cells, audit


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _wavg(values, weights) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    tot = w.sum()
    return float((v * w).sum() / tot) if tot > 0 else np.nan


@dataclass
class FitResult:
    line: int
    label: str
    fit_mode: FitMode
    cat_mode: CatMode
    slope: float
    intercept: float
    ntr0_value: float | None
    p_value: float
    fell_back: bool
    fallback_reason: str
    n_cells: int
    premium: float
    wavg_lr: float
    excluded_states: tuple[int, ...]
    points: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def predict(self, ntr: Iterable[float]) -> np.ndarray:
        ntr = np.asarray(list(ntr), dtype=float)
        out = self.intercept + self.slope * ntr
        if self.ntr0_value is not None:
            out = np.where(ntr == 0, self.ntr0_value, out)
        return out

    def as_tuple(self) -> list[float]:
        """Match the production spl_fits format."""
        base = [self.slope, self.intercept]
        return base + [self.ntr0_value] if self.ntr0_value is not None else base


def fit_line(
    cells: pd.DataFrame,
    line: int,
    spec: LineSpec,
    cat_mode: CatMode = "ex_cat",
    p_threshold: float = 0.1,
    reject_positive_slope: bool = True,
) -> FitResult:
    """Weighted least squares of loss_ratio on NTR for one line.

    Fallback rule (reproduces production): if the NTR coefficient is not
    significant at ``p_threshold``, or -- when ``reject_positive_slope`` --
    the slope is positive, replace the fit with a flat weighted average.

    A positive slope means loss ratio rising with tenure, which contradicts
    the retention story, so production overrides it rather than shipping it.

    ``reject_positive_slope`` is exposed because the original notebook is
    inconsistent: the exploratory cells test ``p > 0.1 or slope > 0`` while
    the "Finalized Fits" cell tests only ``p > 0.1``. Confirm with Yinan which
    is intended before release.

    Note on the flat fallback: the weighted average is computed on the *same*
    subset used for the regression (NTR > 0 for ntr_gt0 lines). The original
    notebook averaged over all points even for ntr_gt0 lines, which mixes the
    new-business bucket back into a value meant to exclude it.
    """
    data = cells.loc[(cells["ALINE"] == line) & (cells["EPREM"] > 0)].copy()
    if spec.exclude_states:
        data = data.loc[~data["GEOST"].isin(spec.exclude_states)]
    if data.empty:
        raise ValueError(f"line {line}: no rows after filtering")

    fit_data = data.loc[data["NTR"] > 0] if spec.fit_mode == "ntr_gt0" else data
    if len(fit_data) < 3:
        raise ValueError(f"line {line}: only {len(fit_data)} rows to fit")

    lm = smf.wls("loss_ratio ~ NTR", data=fit_data, weights=fit_data["w"]).fit()
    p_value = float(lm.pvalues["NTR"])
    raw_slope = float(lm.params["NTR"])

    reasons = []
    if p_value > p_threshold:
        reasons.append(f"p={p_value:.3g} > {p_threshold}")
    if reject_positive_slope and raw_slope > 0:
        reasons.append(f"slope={raw_slope:.4g} > 0")

    if reasons:
        slope = 0.0
        intercept = _wavg(fit_data["loss_ratio"], fit_data["w"])
    else:
        slope = raw_slope
        intercept = float(lm.params["Intercept"])

    ntr0_value = None
    if spec.fit_mode == "ntr_gt0":
        ntr0 = data.loc[data["NTR"] == 0]
        if not ntr0.empty:
            ntr0_value = _wavg(ntr0["loss_ratio"], ntr0["w"])

    points = (
        data.groupby("NTR")
        .apply(
            lambda g: pd.Series(
                {
                    "loss_ratio_wavg": _wavg(g["loss_ratio"], g["w"]),
                    "premium": g["EPREM"].sum(),
                    "totloss": g["TOTLOSS"].sum(),
                    "n_cells": len(g),
                    "max_loss_ratio": g["loss_ratio"].max(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("NTR")
    )

    return FitResult(
        line=line,
        label=spec.label or str(line),
        fit_mode=spec.fit_mode,
        cat_mode=cat_mode,
        slope=slope,
        intercept=intercept,
        ntr0_value=ntr0_value,
        p_value=p_value,
        fell_back=bool(reasons),
        fallback_reason="; ".join(reasons),
        n_cells=len(data),
        premium=float(data["EPREM"].sum()),
        wavg_lr=_wavg(data["loss_ratio"], data["w"]),
        excluded_states=spec.exclude_states,
        points=points,
    )


def fit_all(
    cells: pd.DataFrame,
    line_spec: dict[int, LineSpec],
    cat_mode: CatMode = "ex_cat",
    **kwargs,
) -> dict[int, FitResult]:
    out = {}
    for line, spec in line_spec.items():
        if line not in cells["ALINE"].unique():
            print(f"  ! line {line} not present in extract -- skipped")
            continue
        out[line] = fit_line(cells, line, spec, cat_mode=cat_mode, **kwargs)
    return out


def fits_frame(fits: dict[int, FitResult]) -> pd.DataFrame:
    """One row per line: the fit summary you can paste into a status update."""
    return pd.DataFrame(
        [
            {
                "line": f.line,
                "label": f.label,
                "fit_mode": f.fit_mode,
                "cat_mode": f.cat_mode,
                "slope": f.slope,
                "intercept": f.intercept,
                "ntr0_value": f.ntr0_value,
                "p_value": f.p_value,
                "fell_back": f.fell_back,
                "fallback_reason": f.fallback_reason,
                "wavg_lr": f.wavg_lr,
                "premium": f.premium,
                "n_cells": f.n_cells,
                "excl_states": ",".join(map(str, f.excluded_states)) or "",
            }
            for f in fits.values()
        ]
    ).sort_values("line")


# ---------------------------------------------------------------------------
# Reconciliation -- the deliverable for the Yinan / Steven discussion
# ---------------------------------------------------------------------------


def reconcile_cat(
    premium: pd.DataFrame,
    claims: pd.DataFrame,
    by: Sequence[str] = ("ALINE",),
    cat_col: str = "CATCD",
) -> pd.DataFrame:
    """Premium and loss with cat in, cat out, and cat only, side by side.

    This answers "what is the change in losses", which is the reason Yinan
    wants the pipeline rerun. Group by ("ALINE",) for the headline and
    ("ALINE", "NTR") to see where in the tenure curve the change lands.
    """
    by = list(by)
    frames = {}
    for mode in ("all", "ex_cat", "cat_only"):
        cells, _ = build_cells(premium, aggregate_claims(claims, mode, cat_col))
        g = cells.groupby(by, dropna=False).agg(
            premium=("EPREM", "sum"), totloss=("TOTLOSS", "sum")
        )
        frames[mode] = g

    out = frames["all"][["premium"]].copy()
    for mode in ("all", "ex_cat", "cat_only"):
        out[f"loss_{mode}"] = frames[mode]["totloss"]
    for mode in ("all", "ex_cat", "cat_only"):
        out[f"lr_{mode}"] = out[f"loss_{mode}"] / out["premium"].replace(0, np.nan)

    out["lr_delta"] = out["lr_ex_cat"] - out["lr_all"]
    out["cat_load"] = out["loss_cat_only"] / out["loss_all"].replace(0, np.nan)
    return out.reset_index()


def compare_to_production(
    fits: dict[int, FitResult],
    prod: dict[int, list[float]] = PROD_FITS_V7,
    ntr_grid: Sequence[int] = tuple(range(0, 10)),
) -> pd.DataFrame:
    """New fit vs current production fit at each NTR. Feeds the waterfall."""
    rows = []
    for line, f in sorted(fits.items()):
        new = f.predict(ntr_grid)
        if line in prod:
            p = prod[line]
            old = np.array(
                [
                    p[2] if (len(p) > 2 and n == 0) else p[0] * n + p[1]
                    for n in ntr_grid
                ],
                dtype=float,
            )
        else:
            old = np.full(len(ntr_grid), np.nan)
        for n, o, nw in zip(ntr_grid, old, new):
            rows.append(
                {
                    "line": line,
                    "label": f.label,
                    "NTR": n,
                    "lr_prod": o,
                    "lr_new": nw,
                    "delta": nw - o,
                    "pct_change": (nw - o) / o if o else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------


def emit_spl_fits(fits: dict[int, FitResult], precision: int = 6) -> str:
    """spl_fits dict literal, for the hardcoded block in the pipeline."""
    lines = ["spl_fits = {"]
    for line, f in sorted(fits.items()):
        vals = ", ".join(f"{v:.{precision}f}" for v in f.as_tuple())
        lines.append(f"    {line}: [{vals}],  # {f.label}")
    lines.append("}")
    return "\n".join(lines)


def emit_lr_tuples(fits: dict[int, FitResult], precision: int = 10) -> str:
    """other_lr_tuple(...) form used by the classic jobs policy file."""
    return "\n".join(
        f'other_lr_tuple(line="{f.line}", slope_yr={f.slope:.{precision}f}, '
        f"intercept={f.intercept:.{precision}f}),"
        for f in sorted(fits.values(), key=lambda x: x.line)
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_fit(fit: FitResult, prod: dict[int, list[float]] | None = PROD_FITS_V7,
             out_path: str | None = None, ylim=(0.2, 1.25)):
    """Observed weighted averages, new fit, and current production fit."""
    import matplotlib.pyplot as plt

    pts = fit.points
    fig, ax = plt.subplots()
    ax.scatter(pts["NTR"], pts["loss_ratio_wavg"], label="observed (weighted)")
    ax.plot(pts["NTR"], fit.predict(pts["NTR"]), "-",
            label=f"new fit ({fit.cat_mode})")

    if prod and fit.line in prod:
        p = prod[fit.line]
        old = [p[2] if (len(p) > 2 and n == 0) else p[0] * n + p[1]
               for n in pts["NTR"]]
        ax.plot(pts["NTR"], old, ":", color="red", label="production")

    excl = f", excl GEOST {fit.excluded_states}" if fit.excluded_states else ""
    ax.set_title(f"Line {fit.line} {fit.label} -- {fit.fit_mode}{excl}")
    ax.set_xlabel("NTR")
    ax.set_ylabel("average loss ratio")
    ax.set_ylim(*ylim)
    ax.legend()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig, ax
