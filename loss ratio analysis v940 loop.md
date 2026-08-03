# Loss Ratio Analysis — Cat Excluded

This is an update to the loss ratio analysis used in **V7 / v9.3.x**, for specialty lines.

- Dan Lithio's original work — *[link]*
- Soren's refresh for v7 — *[link]*
- Update for specialty lines (ngraf) — *[link]*
- **This update: Classic SPL v9.4.0 / Renters v1.5.0** — *[link to notebook]*

**What changed from the prior version:** catastrophe losses are now removed from the
loss ratio numerator before fitting. Earned premium in the denominator is unchanged.
Everything else — the cell grain, the WLS fit, the fallback rule — follows the prior
analysis.

Rule of thumb to determine the slope: if the slope had a p-value > 0.1 or if the slope
was positive (i.e., loss ratio does not improve with NTR), we just use the weighted
avg loss ratio for every NTR.

---

## Data pull

Five CDF extracts, each producing a premium file (`…p.csv`) and a claims file
(`…c.csv`). Joined at the cell grain:
`ACTMO × ACTYR × ALINE × CLINE × COMPNY × GEOST × NTR`.

| Extract | Lines | Product |
|---|---|---|
| `ssenp_20260730-142446` | 16 | Specialty Auto |
| `ssenp_20260731-125126` | 62, 64, 70, 71, 72, 78, 79 | Home, Renters, Landlord, Condo |
| `ssenp_20260731-125050` | 32 | Manufactured Home |
| `ssenp_20260731-125211` | 88 | PUP |
| `ssenp_20260731-130101` | 90 | Boat |

| Parameter | Value |
|---|---|
| Earnings Type | Calendar Year Policy Level |
| Evaluation Date (MM/YYYY) | *[confirm]* |
| Exposure Years | 2023 – 2025 (3) *[confirm — see Open Items]* |
| Accounting Start Date | 01/2023 |
| Accounting End Date | *[confirm]* |
| Lines fit | 16, 32, 71, 72, 78, 88, 90 |
| Lines excluded | 62, 64, 70, 79 (standard home/renters, not specialty) |

In most states, auto NTR reflects a 6-month policy term; property NTR is annual.
NTR = 9 is a "9 or more" bucket.

---

## Catastrophe treatment

The claims extract now carries **`CATCD`**. Non-blank = catastrophe; blank = non-cat.

Because a single cell can carry several cat events plus non-cat losses, `CATCD`
adds grain to the claims file — 273,016 claim rows against 91,299 unique cells.
Claims are therefore **filtered and re-aggregated back to cell level before the
merge to premium**. Merging first would fan out the premium rows and inflate exposure.

Data-quality checks on the pull:

| Check | Result |
|---|---|
| True duplicates on keys + CATCD | 0 |
| Additivity (all = ex-cat + cat-only) | passes to the cent |
| Merge outcome | both 82,913 / premium-only 51,832 / claims-only 4,963 |
| Loss on zero-premium cells (excluded from all fits) | 0.35% |

### Cat share of incurred loss, by line

| Line | Product | Total loss | Cat loss | Cat share |
|---|---|---:|---:|---:|
| 16 | Specialty Auto | 504.9M | 66.0M | 13.1% |
| 32 | Mfg Home | 170.3M | 75.7M | 44.5% |
| 71 | Renters | 135.9M | 22.6M | 16.7% |
| 72 | Landlord | 909.7M | 406.5M | 44.7% |
| 78 | Condo | 529.4M | 84.2M | 15.9% |
| 88 | PUP | 739.5M | 0.0 | 0.0% |
| 90 | Boat | 142.5M | 33.7M | 23.6% |

The ordering is coherent with what each line insures: manufactured home and landlord
own the structure and carry the roof exposure; condo and renters are largely interior
and contents; PUP is excess liability with no property cat exposure.

---

## Results by line

### Loss ratio, with and without cat

| Line | Product | Earned premium | LR with cat | LR ex-cat | Change |
|---|---|---:|---:|---:|---:|
| 16 | Specialty Auto | 827.6M | 0.6101 | 0.5303 | −0.0797 |
| 32 | Mfg Home | 325.6M | 0.5231 | 0.2905 | −0.2326 |
| 71 | Renters | 238.6M | 0.5696 | 0.4747 | −0.0949 |
| 72 | Landlord | 1,593.8M | 0.5708 | 0.3157 | −0.2550 |
| 78 | Condo | 1,021.4M | 0.5183 | 0.4358 | −0.0825 |
| 88 | PUP | 878.8M | 0.8414 | 0.8414 | 0.0000 |
| 90 | Boat | 229.7M | 0.6205 | 0.4740 | −0.1465 |

### Fits (ex-cat)

| Line | Product | Fit mode | Slope | Intercept | NTR=0 value | p-value | Raw slope | Fell back to flat |
|---|---|---|---:|---:|---:|---:|---:|---|
| 16 | Specialty Auto | NTR>0 + separate NTR=0 | −0.03416 | 0.61055 | 0.97010 | <0.001 | −0.03416 | No |
| 32 | Mfg Home | all points | −0.01334 | 0.36648 | — | <0.001 | −0.01334 | No |
| 71 | Renters | all points | −0.04357 | 0.62520 | — | <0.001 | −0.04357 | No |
| 72 | Landlord | all points | 0.00000 | 0.31543 | — | <0.001 | **+0.00733** | **Yes — positive slope** |
| 78 | Condo (excl FL) | all points | 0.00000 | 0.49302 | — | **0.193** | −0.02413 | **Yes — p > 0.1** |
| 88 | PUP | NTR>0 + separate NTR=0 | 0.00000 | 0.78882 | 1.18068 | **0.471** | −0.02069 | **Yes — p > 0.1** |
| 90 | Boat | NTR>0 + separate NTR=0 | −0.02375 | 0.56508 | 0.76990 | <0.001 | −0.02375 | No |

Three lines fall back to a flat weighted average. **Landlord (72) and Condo (78) are
new fallbacks** — both carried a slope in production. Removing cat flattens the
apparent tenure relationship on those lines, which suggests the prior slope was partly
cat-driven.

### Line 78 — Florida carve-out retained

The prior analysis excluded Florida from the condo fit. That carve-out was assumed to
be cat-driven and was expected to dissolve once cat losses were removed. **It did not.**

Loss ratio by NTR, ex-cat:

| NTR | Non-FL | FL |
|---:|---:|---:|
| 0 | 0.5913 | **4.3420** |
| 1 | 0.6535 | 0.4814 |
| 2 | 0.5977 | 0.2470 |
| 3 | 0.5041 | 0.1472 |
| 4 | 0.4975 | 0.0904 |
| 5 | 0.4905 | 0.0137 |
| 6 | 0.5049 | — |
| 7 | 0.5085 | — |
| 8 | 0.4424 | — |
| 9 | 0.4019 | 0.2229 |

Post-cat, FL condo at NTR=0 still runs 4.34 against 0.59 for the rest of the country,
and has no cells at all at NTR 6–8. The book is thin and erratic independent of cat,
so the exclusion is retained on volatility grounds. Including FL would move the
intercept from 0.49302 to 0.43179.

---

## Recommendations

| Line | Product | Recommendation |
|---|---|---|
| 16 | Specialty Auto | Weighted avg at NTR=0, fit for NTR>0 |
| 32 | Mfg Home | Fit all NTRs |
| 71 | Renters | Fit all NTRs |
| 72 | Landlord | Flat weighted avg for every NTR *(changed from fit)* |
| 78 | Condo | Flat weighted avg for every NTR, excluding FL *(changed from fit)* |
| 88 | PUP | Weighted avg at NTR=0, flat weighted avg for NTR>0 |
| 90 | Boat | Weighted avg at NTR=0, fit for NTR>0 |

```python
spl_fits = {
    16: [-0.034158, 0.610550, 0.970097],  # Specialty Auto
    32: [-0.013335, 0.366484],            # Mfg Home
    71: [-0.043571, 0.625204],            # Renters
    72: [ 0.000000, 0.315427],            # Landlord
    78: [ 0.000000, 0.493024],            # Condo
    88: [ 0.000000, 0.788817, 1.180678],  # PUP
    90: [-0.023748, 0.565079, 0.769902],  # Boat
}
```

```
other_lr_tuple(line="16", slope_yr=-0.0341583943, intercept=0.6105496031),
other_lr_tuple(line="32", slope_yr=-0.0133350880, intercept=0.3664844453),
other_lr_tuple(line="71", slope_yr=-0.0435713887, intercept=0.6252041479),
other_lr_tuple(line="72", slope_yr= 0.0000000000, intercept=0.3154266994),
other_lr_tuple(line="78", slope_yr= 0.0000000000, intercept=0.4930239249),
other_lr_tuple(line="88", slope_yr= 0.0000000000, intercept=0.7888171791),
other_lr_tuple(line="90", slope_yr=-0.0237481671, intercept=0.5650788252),
```

---

## Impact vs. current production fits

Premium-weighted across the NTR distribution:

| Line | Product | Production LR | New LR | Change | Loss dollar impact |
|---|---|---:|---:|---:|---:|
| 16 | Specialty Auto | 0.6027 | 0.5303 | −0.0724 | −59.9M |
| 32 | Mfg Home | 0.5404 | 0.2905 | −0.2499 | −81.4M |
| 71 | Renters | 0.5516 | 0.4747 | −0.0769 | −18.4M |
| 72 | Landlord | 0.5900 | 0.3154 | −0.2746 | −437.6M |
| 78 | Condo | 0.6553 | 0.4930 | −0.1623 | −165.7M |
| 88 | PUP | 0.7982 | 0.8245 | +0.0263 | +23.2M |
| 90 | Boat | 0.6643 | 0.4740 | −0.1902 | −43.7M |
| | **Total** | | | **−0.153** | **−783.5M** |

PUP is the only line to increase, and only slightly — it has no cat to remove, so the
change is entirely the data refresh.

Note this is the change to the **NTR-varying loss ratio component only**. Where the cat
load is reintroduced downstream is outstanding — see Open Items.

---

## Open items

1. **Exposure years.** The pull spans ACTYR 2023–2025; the prior analysis used 2023–2024.
   If 2025 losses are still developing, they bias loss ratios downward independent of cat.
   Partial reassurance: with-cat loss ratios track production closely on most lines
   (16: 0.610 vs 0.603; 32: 0.523 vs 0.540; 72: 0.571 vs 0.590), but Condo is 0.518 vs
   0.655 — a 14-point gap that is not cat. **Rerun restricted to 2023–2024 to separate
   the two effects before release.**
2. **Where does the cat load get added back?** Premium stays whole in the denominator,
   so the ex-cat loss ratio understates total expected loss by design.
3. **PUP shows exactly zero cat loss** across $739M. Consistent with excess liability
   having no property cat, but worth confirming CATCD populated correctly in that extract.
4. **Cat code `'9'`** appears alongside A–Z. A numeric value in an otherwise alphabetic
   field may be a sentinel rather than a catastrophe designation — confirm with Nick.
5. **NTR = 9 is censored.** It means "9 or more" but enters the regression as literally 9,
   and carries the heaviest premium weight on several lines. The prior notebook contained
   two abandoned attempts to correct for this (a `(10−NTR)(11−NTR)` weight, and truncating
   at NTR < 8); neither shipped. Options: drop the bucket, downweight it, or obtain its
   true mean tenure and use that as the x-value.
6. **Fallback rule wording.** Production tests `p > 0.1 or slope > 0`; the prior notebook's
   final cell tests only `p > 0.1`. This analysis uses the former. Confirm which is intended —
   it is what sends Landlord to flat.
7. **Production comparison figures** transcribed from the prior notebook; verify against
   pipeline source before publishing the impact table.
