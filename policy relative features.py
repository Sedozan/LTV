# Databricks notebook source
# MAGIC %md
# MAGIC # Policy-Relative Feature Engineering — Step 5
# MAGIC
# MAGIC **Purpose**: Convert raw utilization features into policy-anchored features.
# MAGIC "12 units/DOS" is a statistic; "4 units above the documented policy threshold" is a signal.
# MAGIC
# MAGIC **Input**: claims DataFrame + `policy_rules_validated` (clean/patched, machine_checkable only)
# MAGIC **Output**: one row per provider, joinable onto your existing feature matrix by provider_id.
# MAGIC
# MAGIC **Features produced (per provider)**:
# MAGIC - Per threshold rule & code group: `n_dos_above_thr__{code}`, `pct_dos_above_thr__{code}`,
# MAGIC   `excess_units__{code}` (sum of units beyond threshold), `max_units_over_thr__{code}`
# MAGIC - `discharge_day_billing_rate` (if discharge date available)
# MAGIC - `date_range_claim_rate` (post effective date)
# MAGIC - `wrong_pos_rate` (post effective date)
# MAGIC - `unbundled_dos_count`, `unbundled_dos_rate` (per-diem DOS with bundled codes billed)
# MAGIC - `freq_limit_member_overage__{code}` (member-years beyond frequency limits)
# MAGIC - `policy_violation_surface` (count of distinct rules with >=1 hit) — a compact scalar
# MAGIC   for the anomaly model.
# MAGIC
# MAGIC All computations are date-aware (claims before a rule's effective date are excluded)
# MAGIC and vectorized (groupbys, not per-provider loops).

# COMMAND ----------

import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

ROOT = dbutils.widgets.get("ROOT") if "ROOT" in [w.name for w in dbutils.widgets.getAll()] else os.environ.get("ROOT", "your_email@nttdata.com")
prefix = open(f"/Workspace/Users/{ROOT}/config/prefix.txt").read().strip() if os.path.exists(f"/Workspace/Users/{ROOT}/config/prefix.txt") else ""
CATALOG_DB = f"{prefix}policy_intelligence" if prefix else "policy_intelligence"
VALIDATED_TABLE = f"{CATALOG_DB}.policy_rules_validated"

# COMMAND ----------

def load_usable_rules(spark, population: str) -> pd.DataFrame:
    r = spark.table(VALIDATED_TABLE).toPandas()
    r = r[r["validation_status"].isin(["clean", "patched"])]
    r = r[r["machine_checkable"] == True]
    r["threshold_numeric"] = pd.to_numeric(r["threshold_numeric"], errors="coerce")
    r["effective_dt"] = pd.to_datetime(r["effective_date"], errors="coerce")
    pat = rf"(^|\|){re.escape(population)}($|\|)"
    r = r[r["population"].fillna("").str.contains(pat, regex=True)
          | (r["population"] == "cross_population")]
    return r.reset_index(drop=True)


def _codes(s) -> List[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    return [c.strip().upper() for c in str(s).split("|") if c.strip()]


def _code_tag(codes: List[str]) -> str:
    """Feature-name-safe tag for a code group."""
    return codes[0] if len(codes) == 1 else f"{codes[0]}_grp{len(codes)}"


def build_policy_relative_features(
    claims_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    provider_id_col: str = "provider_id",
    member_id_col: str = "member_id",
    proc_code_col: str = "proc_cd",
    units_col: str = "units",
    service_date_col: str = "srvc_bgn_dt",
    service_end_date_col: Optional[str] = "srvc_end_dt",
    discharge_date_col: Optional[str] = "discharge_dt",
    pos_col: str = "pos_cd",
    modifier_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Returns one row per provider with policy-relative features."""
    if modifier_cols is None:
        modifier_cols = ["mod_1", "mod_2", "mod_3", "mod_4"]

    df = claims_df.copy()
    df["_svc"] = pd.to_datetime(df[service_date_col], errors="coerce")
    df["_dos"] = df["_svc"].dt.date
    df["_code"] = df[proc_code_col].astype(str).str.upper()
    df["_units"] = pd.to_numeric(df[units_col], errors="coerce").fillna(0)

    providers = pd.DataFrame({provider_id_col: df[provider_id_col].unique()})
    feats = providers.set_index(provider_id_col)

    # Denominators
    total_claims = df.groupby(provider_id_col).size().rename("total_claims")
    feats = feats.join(total_claims)
    hit_rule_ids = {}   # provider -> set of rule_ids with hits

    def _record_hits(series_by_provider: pd.Series, rule_id: str):
        for pid, n in series_by_provider.items():
            if n > 0:
                hit_rule_ids.setdefault(pid, set()).add(rule_id)

    # ---------------- Threshold rules (unit_threshold + documentation_requirement) --------
    thr_rules = rules_df[rules_df["rule_type"].isin(["unit_threshold", "documentation_requirement"])
                         & rules_df["threshold_numeric"].notna()]
    for _, rule in thr_rules.iterrows():
        codes = _codes(rule["proc_codes"])
        if not codes:
            continue
        thr = float(rule["threshold_numeric"])
        sub = df[df["_code"].isin(codes)]
        if not pd.isna(rule["effective_dt"]):
            sub = sub[sub["_svc"] >= rule["effective_dt"]]
        if sub.empty:
            continue
        tag = _code_tag(codes)
        dos = (sub.groupby([provider_id_col, member_id_col, "_dos"])["_units"]
                  .sum().reset_index())
        dos["_excess"] = (dos["_units"] - thr).clip(lower=0)
        agg = dos.groupby(provider_id_col).agg(
            n_dos=("_units", "size"),
            n_above=("_excess", lambda s: int((s > 0).sum())),
            excess=("_excess", "sum"),
            max_over=("_excess", "max"),
        )
        feats[f"n_dos_above_thr__{tag}"] = agg["n_above"]
        feats[f"pct_dos_above_thr__{tag}"] = (agg["n_above"] / agg["n_dos"]).round(4)
        feats[f"excess_units__{tag}"] = agg["excess"]
        feats[f"max_units_over_thr__{tag}"] = agg["max_over"]
        _record_hits(agg["n_above"], rule["rule_id"])

    # ---------------- Discharge-day billing --------------------------------------------
    dis_rules = rules_df[(rules_df["rule_type"] == "billing_prohibition")
                         & (rules_df["rule_subtype"] == "discharge_day")]
    if discharge_date_col and discharge_date_col in df.columns and len(dis_rules):
        codes = sorted({c for _, r in dis_rules.iterrows() for c in _codes(r["proc_codes"])})
        sub = df[df["_code"].isin(codes)].copy()
        sub["_dis"] = pd.to_datetime(sub[discharge_date_col], errors="coerce").dt.date
        sub["_hit"] = sub["_dis"].notna() & (sub["_dos"] == sub["_dis"])
        agg = sub.groupby(provider_id_col)["_hit"].agg(["sum", "size"])
        feats["discharge_day_billing_count"] = agg["sum"]
        feats["discharge_day_billing_rate"] = (agg["sum"] / agg["size"]).round(4)
        _record_hits(agg["sum"], dis_rules.iloc[0]["rule_id"])

    # ---------------- Date-range billing ------------------------------------------------
    dr_rules = rules_df[(rules_df["rule_type"] == "billing_prohibition")
                        & (rules_df["rule_subtype"] == "date_range")]
    if service_end_date_col and service_end_date_col in df.columns and len(dr_rules):
        for _, rule in dr_rules.iterrows():
            codes = _codes(rule["proc_codes"])
            sub = df[df["_code"].isin(codes)].copy()
            if not pd.isna(rule["effective_dt"]):
                sub = sub[sub["_svc"] >= rule["effective_dt"]]
            if sub.empty:
                continue
            end = pd.to_datetime(sub[service_end_date_col], errors="coerce")
            sub["_hit"] = end.notna() & sub["_svc"].notna() & ((end - sub["_svc"]).dt.days > 0)
            agg = sub.groupby(provider_id_col)["_hit"].agg(["sum", "size"])
            feats["date_range_claim_count"] = agg["sum"]
            feats["date_range_claim_rate"] = (agg["sum"] / agg["size"]).round(4)
            _record_hits(agg["sum"], rule["rule_id"])

    # ---------------- Wrong POS ----------------------------------------------------------
    pos_rules = rules_df[(rules_df["rule_type"] == "pos_requirement")
                         & rules_df["threshold_numeric"].notna()]
    if pos_col in df.columns:
        for _, rule in pos_rules.iterrows():
            codes = _codes(rule["proc_codes"])
            required = str(int(rule["threshold_numeric"])).zfill(2)
            sub = df[df["_code"].isin(codes)].copy()
            if not pd.isna(rule["effective_dt"]):
                sub = sub[sub["_svc"] >= rule["effective_dt"]]
            if sub.empty:
                continue
            sub["_hit"] = sub[pos_col].astype(str).str.strip().str.zfill(2) != required
            agg = sub.groupby(provider_id_col)["_hit"].agg(["sum", "size"])
            tag = _code_tag(codes)
            feats[f"wrong_pos_count__{tag}"] = agg["sum"]
            feats[f"wrong_pos_rate__{tag}"] = (agg["sum"] / agg["size"]).round(4)
            _record_hits(agg["sum"], rule["rule_id"])

    # ---------------- Unbundling (inclusive rate) ---------------------------------------
    inc_rules = rules_df[(rules_df["rule_type"] == "inclusive_rate")
                         & rules_df["related_codes"].notna()]
    for _, rule in inc_rules.iterrows():
        inc_codes, bun_codes = _codes(rule["proc_codes"]), _codes(rule["related_codes"])
        if not inc_codes or not bun_codes:
            continue
        inc = df[df["_code"].isin(inc_codes)][[provider_id_col, member_id_col, "_dos"]].drop_duplicates()
        bun = df[df["_code"].isin(bun_codes)][[provider_id_col, member_id_col, "_dos"]].drop_duplicates()
        pairs = inc.merge(bun, on=[provider_id_col, member_id_col, "_dos"])
        n_pairs = pairs.groupby(provider_id_col).size()
        n_inc_dos = inc.groupby(provider_id_col).size()
        feats["unbundled_dos_count"] = n_pairs
        feats["unbundled_dos_rate"] = (n_pairs / n_inc_dos).round(4)
        _record_hits(n_pairs, rule["rule_id"])

    # ---------------- Frequency limits ---------------------------------------------------
    freq_rules = rules_df[(rules_df["rule_type"] == "frequency_limit")
                          & rules_df["threshold_numeric"].notna()]
    avail_mods = [c for c in modifier_cols if c in df.columns]
    for _, rule in freq_rules.iterrows():
        codes = _codes(rule["proc_codes"])
        sub = df[df["_code"].isin(codes)].copy()
        mods = _codes(rule["modifiers"])
        if mods and avail_mods:
            mask = pd.Series(False, index=sub.index)
            for mc in avail_mods:
                mask |= sub[mc].astype(str).str.upper().isin(mods)
            sub = sub[mask]
        if sub.empty:
            continue
        thr = float(rule["threshold_numeric"])
        sub["_yr"] = sub["_svc"].dt.year
        counts = (sub.groupby([provider_id_col, member_id_col, "_yr"])["_dos"]
                     .nunique().reset_index(name="n_dos"))
        counts["_over"] = (counts["n_dos"] - thr).clip(lower=0)
        agg = counts.groupby(provider_id_col)["_over"].agg(["sum", lambda s: int((s > 0).sum())])
        agg.columns = ["overage_dos", "member_years_over"]
        tag = _code_tag(codes) + ("_" + "_".join(mods) if mods else "")
        feats[f"freq_overage_dos__{tag}"] = agg["overage_dos"]
        feats[f"freq_member_years_over__{tag}"] = agg["member_years_over"]
        _record_hits(agg["member_years_over"], rule["rule_id"])

    # ---------------- Compact scalar ------------------------------------------------------
    feats["policy_violation_surface"] = pd.Series(
        {pid: len(rids) for pid, rids in hit_rule_ids.items()}, dtype="float")

    feats = feats.fillna(0).reset_index()
    return feats

# COMMAND ----------

# MAGIC %md
# MAGIC ## Usage

# COMMAND ----------

# rules = load_usable_rules(spark, "bh_residential")
# claims = spark.table("your_db.bh_residential_claims").toPandas()
#
# policy_feats = build_policy_relative_features(claims, rules)
# print(policy_feats.shape)
# print([c for c in policy_feats.columns][:20])
#
# # Join onto your existing feature matrix before scoring:
# # feature_matrix = feature_matrix.merge(policy_feats, on="provider_id", how="left").fillna(0)
#
# # Persist for the anomaly pipeline:
# # spark.createDataFrame(policy_feats).write.format("delta").mode("overwrite") \
# #      .saveAsTable(f"{CATALOG_DB}.policy_relative_features_bh_residential")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes & caveats (read before wiring into the model)
# MAGIC - **Sparsity is expected and meaningful.** Most providers should be 0 on most features.
# MAGIC   For tree/isolation models that's fine; for distance-based models consider log1p or
# MAGIC   rate-only variants.
# MAGIC - **Do not double-count**: `excess_units__X` and `n_dos_above_thr__X` are correlated by
# MAGIC   construction. Pick per model, or let regularization sort it out — but know they overlap.
# MAGIC - **discharge/date-range features require the corresponding columns**; if absent they're
# MAGIC   silently skipped (check output columns after first run).
# MAGIC - **Frequency features use calendar years** — same assumption as the rules engine, stated
# MAGIC   in investigator output. If AHCCCS means rolling 12 months, both must change together.
