"""
Shared release-validation helpers: univariate compares + LTV/AAC waterfalls.

Ported from NB_SPL_v1.4.0_waterfall_and_univars.ipynb (renters, v1.4 vs v1.2).
Deliberate changes from the original:

  1. Component lookup is STRICT. The original `df.get(c, 0)` turns a typo into a
     silent zero bar. Suspect this already bit `cost_of_capital_20_pct_diff`
     (column is `cost_of_capital_20pct`, no underscore).
  2. Waterfall asserts closure: anchor + sum(signed diffs) == final, within tol.
     Without this the chart cannot tell you it is wrong.
  3. Merge asserts no fan-out and reports population drift, so classic's
     "same policies, different premium" problem shows up as a number rather
     than as component bars.
  4. Expense prefix is discovered from the frame, not hardcoded. The prefix
     changed WITHIN the renters repo (v1.2 = e005scl_, v1.4 = e006scl_), so
     "classic = e005scl, renters = e006scl" is not a safe assumption.

Usage in a notebook:

    import spl_waterfall as wf
    cfg = wf.CFG["renters"]
    ...
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

OLD = "_old"  # suffix on baseline-release columns (was "_v1_2")

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

CFG = {
    "renters": {
        "id_col": "ply_policy_id",
        # confirm at runtime; renters merged cleanly on id alone in v1.4
        "merge_keys": ["ply_policy_id"],
        "new_release": "release/NB_SPL_v1.5.0",
        "old_release": "release/NB_SPL_v1.4.0",
        "label_new": "SPL Renters v1.5",
        "label_old": "SPL Renters v1.4",
    },
    "classic": {
        "id_col": "adw_pol_id",
        # id alone fans out 2.25M -> 4.42M. release_day is required.
        "merge_keys": ["adw_pol_id", "release_day"],
        "new_release": "release/NB_SPL_v9.4.0",
        "old_release": "release/NB_SPL_v9.3.1",
        "label_new": "Classic SPL v9.4.0",
        "label_old": "Classic SPL v9.3.1",
    },
}

EXP_PREFIX_RE = re.compile(r"^e\d{3}scl_")

# Columns that participate in the waterfall, AFTER the expense prefix is stripped.
WF_COLS = [
    "lifetime_premium",
    "lifetime_loss",
    "balance_amt",
    "cat_loss_amt",
    "commission_exp_new",
    "commission_exp_renew",
    "investment_income",
    "tax",
    "ple",
    "ltv",
    "cost_of_capital",
    "cost_of_capital_20pct",
    "acquisition_exp",
    "claims_exp",
    "lifetime_exp",
    "cac_x_mkt",
    "aac_mkt",
]

# sign convention: diff = old - new, so a bar is the component's contribution
# to (old metric -> new metric).  -1 for terms that ADD to the metric.
LTV_SIGNS = {
    "lifetime_premium": -1,
    "lifetime_loss": 1,
    "balance_amt": 1,
    "cat_loss_amt": 1,
    "claims_exp": 1,
    "lifetime_exp": 1,
    "commission_exp_renew": 1,
    "cost_of_capital": 1,
    "tax": 1,
    "investment_income": -1,
}

AAC_SIGNS = {
    **{k: v for k, v in LTV_SIGNS.items() if k != "cost_of_capital"},
    "cost_of_capital_20pct": 1,
    "commission_exp_new": 1,
    "acquisition_exp": 1,
}

LABELS = {
    "lifetime_premium": "Lifetime Premium",
    "lifetime_loss": "Lifetime Loss",
    "balance_amt": "Balance",
    "cat_loss_amt": "Cat Loss",
    "claims_exp": "Claims Expense",
    "lifetime_exp": "Lifetime Expense",
    "commission_exp_renew": "Commission Renew",
    "commission_exp_new": "Commission New",
    "acquisition_exp": "Acquisition Expense",
    "cost_of_capital": "Cost of Capital",
    "cost_of_capital_20pct": "Cost of Capital (20%)",
    "tax": "Tax",
    "investment_income": "Investment Income",
}


# --------------------------------------------------------------------------
# load + merge
# --------------------------------------------------------------------------


def swap_release(path, new_release, old_release):
    """Point a config-resolved path at the baseline release. Fails loudly."""
    if new_release not in path:
        raise ValueError(f"{new_release!r} not found in {path!r}")
    return path.replace(new_release, old_release)


def strip_exp_prefix(df):
    """Drop e0NNscl_ from column names and report what was found."""
    found = sorted({m.group(0) for c in df.columns if (m := EXP_PREFIX_RE.match(c))})
    df = df.rename(columns=lambda c: EXP_PREFIX_RE.sub("", c))
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"prefix strip collided on: {dupes}")
    return df, found


def merge_releases(df_new, df_old, cfg, drift_col="drv_full_premium_amt"):
    """Inner-join the two releases. Raises on fan-out, reports drift."""
    keys = cfg["merge_keys"]
    n_new, n_old = len(df_new), len(df_old)

    for name, d in (("new", df_new), ("old", df_old)):
        if d.duplicated(keys).any():
            raise ValueError(f"{name} release is not unique on {keys}")

    old = df_old.rename(columns={c: c + OLD for c in df_old.columns if c not in keys})
    out = df_new.merge(old, how="inner", on=keys)

    if len(out) > max(n_new, n_old):
        raise ValueError(f"fan-out: {n_new}/{n_old} -> {len(out)} on {keys}")

    print(f"merged on {keys}: new={n_new:,} old={n_old:,} matched={len(out):,} "
          f"({len(out) / n_new:.1%} of new)")

    if drift_col in out and drift_col + OLD in out:
        d = (out[drift_col] - out[drift_col + OLD]).abs()
        print(f"{drift_col} drift: median|d|={d.median():,.2f} "
              f"max={d.max():,.2f} nulls={d.isna().sum():,} "
              f"pct_exact={np.isclose(d.fillna(1), 0).mean():.1%}")
    return out


def add_diffs(df, cols=WF_COLS):
    """diff = old - new, for every wf col present in both releases."""
    missing = [c for c in cols if c not in df or c + OLD not in df]
    if missing:
        print(f"note: no diff computed for {missing}")
    for c in cols:
        if c in df and c + OLD in df:
            df[f"{c}_diff"] = df[c + OLD] - df[c]
    return df


def stable_subset(df, cfg, drift_col="drv_full_premium_amt", atol=0.01):
    """
    Policies whose premium is unchanged between releases. Use this for classic:
    it isolates fit-driven movement from pipeline-state drift. Report coverage
    alongside any chart built on it.
    """
    same = np.isclose(df[drift_col], df[drift_col + OLD], atol=atol, equal_nan=True)
    print(f"stable subset: {same.sum():,} / {len(df):,} ({same.mean():.1%})")
    return df.loc[same].copy()


# --------------------------------------------------------------------------
# waterfall
# --------------------------------------------------------------------------


def build_components(metric, signs, label_old, label_new):
    """{col: (label, sign)} with the anchor first and the final bar last."""
    comps = {f"{metric}{OLD}": (f"{label_old} {metric.upper()}", 1)}
    for col, sign in signs.items():
        comps[f"{col}_diff"] = (f"{LABELS.get(col, col)} Diff", sign)
    comps[metric] = (f"{label_new} {metric.upper()}", 0)
    return comps


def make_waterfall(row, components, title, atol=0.5, ax=None):
    """
    row: a Series of means (one group). components: {col: (label, sign)}.
    Asserts the decomposition closes; that is the whole point of the chart.
    """
    order = list(components)
    missing = [c for c in order if c not in row.index]
    if missing:
        raise KeyError(f"components not in frame: {missing}")

    labels = [components[c][0] for c in order]
    signs = np.array([components[c][1] for c in order])
    values = np.array([float(row[c]) for c in order])

    heights = [values[0]] + list(values[1:-1] * signs[1:-1]) + [values[-1]]
    implied = heights[0] + sum(heights[1:-1])
    resid = implied - values[-1]
    if abs(resid) > atol:
        raise AssertionError(
            f"waterfall does not close for {title!r}: "
            f"anchor+diffs={implied:,.2f} vs final={values[-1]:,.2f} "
            f"(residual {resid:,.2f}). A component is missing, "
            f"double-counted, or has the wrong sign."
        )

    bottoms, cum = [0], values[0]
    for h in heights[1:-1]:
        bottoms.append(cum)
        cum += h
    bottoms.append(0)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, heights, bottom=bottoms,
           color=cm.tab20(np.arange(len(heights))))
    for i, (b, h) in enumerate(zip(bottoms, heights)):
        ax.text(i, b + h, f"{h:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(implied, color="red", linestyle="--", label="Final $")
    ax.set_ylabel("Value")
    ax.set_title(f"{title}   (residual {resid:+,.2f})")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    return resid


def group_means(df, components_cols, by=None):
    cols = [c for c in components_cols if c in df]
    if by is None:
        return df[cols].mean()
    return df.groupby(by)[cols].mean().reset_index()


# --------------------------------------------------------------------------
# univariate compare (unchanged behaviour, tightened signature)
# --------------------------------------------------------------------------


def plot_counts_with_two_lines(df, group_col, count_col, value_cols,
                               line_labels=None, group_order=None, figsize=None,
                               title=None, xlabel=None,
                               ylabel_left="Count of Policies", ylabel_right=None,
                               rotate_xticks=False):
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    assert len(value_cols) == 2, "value_cols must be two column names"

    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(count=(count_col, "count"),
             value1=(value_cols[0], "mean"),
             value2=(value_cols[1], "mean"))
        .reset_index()
    )
    if group_order:
        grouped[group_col] = pd.Categorical(grouped[group_col],
                                            categories=group_order, ordered=True)
        grouped = grouped.sort_values(group_col)

    if figsize is None:
        figsize = (max(8, 0.28 * len(grouped)), 5)
    x = range(len(grouped))

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(x, grouped["count"], width=0.6, label="Count")
    ax1.set_xlabel(xlabel or group_col)
    ax1.set_ylabel(ylabel_left)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(grouped[group_col], rotation=90 if rotate_xticks else 0)

    ax2 = ax1.twinx()
    for idx, col in enumerate(["value1", "value2"]):
        ax2.plot(x, grouped[col],
                 color=["blue", "orange"][idx], marker="o",
                 label=line_labels[idx] if line_labels else f"Average {value_cols[idx]}")
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel(ylabel_right or f"Average {', '.join(value_cols)}")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title(title or f"Count and Averages by {group_col}")
    plt.tight_layout()
    plt.show()
    return grouped
