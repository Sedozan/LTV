# Databricks notebook source
# MAGIC %md
# MAGIC # Policy KB Validation — Step 3 (Quality Gate)
# MAGIC
# MAGIC **Purpose**: Programmatic validation of `policy_rules`. Produces `policy_rules_validated` —
# MAGIC the ONLY table the rules engine (Step 4), features (Step 5), and LLM pipeline (Step 6) consume.
# MAGIC
# MAGIC **Every rule gets a `validation_status`**:
# MAGIC - `clean` — passed all checks
# MAGIC - `patched` — auto-fixed (fix recorded in `validation_notes`)
# MAGIC - `needs_review` — usable only after human sign-off (excluded from engine by default)
# MAGIC - `quarantined` — structurally unusable (no codes AND no checkable structure)
# MAGIC
# MAGIC Also assigns `rule_subtype` for billing prohibitions so the engine routes deterministically
# MAGIC instead of keyword-guessing (the v1 engine's "cannot" fallback was a false-positive generator).
# MAGIC
# MAGIC **Works with both schemas**: v2 parser output (no machine_checkable/origin columns) and
# MAGIC v3 parser output. Missing columns are derived.

# COMMAND ----------

import os
import re
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timezone

ROOT = dbutils.widgets.get("ROOT") if "ROOT" in [w.name for w in dbutils.widgets.getAll()] else os.environ.get("ROOT", "your_email@nttdata.com")
prefix = open(f"/Workspace/Users/{ROOT}/config/prefix.txt").read().strip() if os.path.exists(f"/Workspace/Users/{ROOT}/config/prefix.txt") else ""
CATALOG_DB = f"{prefix}policy_intelligence" if prefix else "policy_intelligence"

RULES_TABLE     = f"{CATALOG_DB}.policy_rules"
VALIDATED_TABLE = f"{CATALOG_DB}.policy_rules_validated"

# OPTIONAL: point to your BH Residential claims table to cross-reference proc codes.
# Leave as None to skip.
CLAIMS_TABLE = None          # e.g. "your_db.bh_residential_claims"
CLAIMS_PROC_COL = "proc_cd"

rules = spark.table(RULES_TABLE).toPandas()
if "superseded_at" in rules.columns:
    n_hist = rules["superseded_at"].notna().sum()
    rules = rules[rules["superseded_at"].isna()].reset_index(drop=True)
    print(f"Version filter: {n_hist} superseded (historical) rows excluded")
print(f"Loaded {len(rules)} current rules from {RULES_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schema normalization (handles v2 or v3 parser output)

# COMMAND ----------

POPULATION_VOCAB = {"bh_residential", "bh_outpatient", "dme", "nemt",
                    "nursing_home", "professional", "cross_population"}
POPULATION_SYNONYMS = {
    "bhrf": "bh_residential", "behavioral_health_residential": "bh_residential",
    "bh_residential_facility": "bh_residential", "bh_assessment": "bh_outpatient",
    "bh_partial_hospitalization": "bh_outpatient", "behavioral_health_outpatient": "bh_outpatient",
    "bh_crisis": "bh_outpatient", "durable_medical_equipment": "dme",
    "non_emergency_medical_transportation": "nemt", "snf": "nursing_home",
    "all": "cross_population", "general": "cross_population",
}
VALID_RULE_TYPES = {"billing_prohibition", "unit_threshold", "documentation_requirement",
                    "code_restriction", "provider_type_restriction", "modifier_requirement",
                    "pos_requirement", "frequency_limit", "prior_auth_requirement", "inclusive_rate"}
HCPCS_CPT_RE = re.compile(r'^([A-Z]\d{4}|\d{5})$')
UNCHECKABLE_HINTS = ("overnight", "medical record", "crisis system", "on behalf of",
                     "independent provider", "receives treatment elsewhere", "clinical",
                     "level of care", "authorization")

for col, default in [("machine_checkable", None), ("origin", "llm"), ("related_codes", None),
                     ("rule_label", None), ("population_raw", None), ("publication_date", None),
                     ("citation_page_exact", None)]:
    if col not in rules.columns:
        rules[col] = default

rules["validation_status"] = "clean"
rules["validation_notes"] = ""

def add_note(idx, note):
    rules.loc[idx, "validation_notes"] = (rules.loc[idx, "validation_notes"] + "; " + note).str.strip("; ")

def norm_str(s):
    return None if s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip().lower() in ("", "none", "null", "nan") else str(s).strip()

for c in ["proc_codes", "modifiers", "provider_types", "population", "rule_type",
          "condition", "effective_date", "related_codes", "threshold_unit"]:
    rules[c] = rules[c].apply(norm_str)

rules["threshold_numeric"] = pd.to_numeric(rules["threshold_value"], errors="coerce")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 1 — Structural audits (nulls, formats, types)

# COMMAND ----------

print("=" * 60)
print("STRUCTURAL AUDIT")
print("=" * 60)

# 1a. Invalid rule_type -> needs_review
bad_type = ~rules["rule_type"].isin(VALID_RULE_TYPES) | rules["rule_type"].isna()
rules.loc[bad_type, "validation_status"] = "needs_review"
add_note(rules.index[bad_type], "invalid_rule_type")
print(f"Invalid rule_type: {bad_type.sum()}")

# 1b. Malformed proc codes -> strip invalid tokens; if nothing survives, treat as null
def clean_codes(s):
    if s is None:
        return None, False
    toks = [t.strip().upper() for t in s.split("|") if t.strip()]
    valid = [t for t in toks if HCPCS_CPT_RE.match(t)]
    return ("|".join(valid) if valid else None), (len(valid) != len(toks))

cleaned = rules["proc_codes"].apply(clean_codes)
had_bad_tokens = cleaned.apply(lambda x: x[1])
rules["proc_codes"] = cleaned.apply(lambda x: x[0])
rules.loc[had_bad_tokens, "validation_status"] = rules.loc[had_bad_tokens, "validation_status"].replace("clean", "patched")
add_note(rules.index[had_bad_tokens], "dropped_malformed_code_tokens")
print(f"Rules with malformed code tokens stripped: {had_bad_tokens.sum()}")

# 1c. Null proc_codes: quarantine UNLESS the rule is inherently codeless
# (a codeless rule with no threshold and no provider_types cannot be applied to claims)
codeless = rules["proc_codes"].isna()
unusable = codeless & rules["threshold_numeric"].isna() & rules["provider_types"].isna()
rules.loc[unusable, "validation_status"] = "quarantined"
add_note(rules.index[unusable], "no_codes_no_threshold_no_ptype")
print(f"Null proc_codes: {codeless.sum()}  ->  quarantined (unusable): {unusable.sum()}")

# 1d. Threshold-type rules missing a numeric threshold -> needs_review
thr_types = rules["rule_type"].isin(["unit_threshold", "frequency_limit"])
thr_missing = thr_types & rules["threshold_numeric"].isna()
rules.loc[thr_missing, "validation_status"] = "needs_review"
add_note(rules.index[thr_missing], "threshold_rule_missing_numeric_threshold")
print(f"Threshold rules missing threshold: {thr_missing.sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 2 — Population normalization (fixes invented sub-populations)

# COMMAND ----------

def normalize_pop(p):
    if p is None:
        return "needs_review"
    v = p.lower().replace(" ", "_").replace("-", "_")
    if v in POPULATION_VOCAB:
        return v
    return POPULATION_SYNONYMS.get(v, "needs_review")

rules["population_raw"] = rules["population_raw"].fillna(rules["population"])
before = rules["population"].copy()
rules["population"] = rules["population"].apply(normalize_pop)
remapped = (before != rules["population"]) & (rules["population"] != "needs_review")
rules.loc[remapped, "validation_status"] = rules.loc[remapped, "validation_status"].replace("clean", "patched")
add_note(rules.index[remapped], "population_remapped")

pop_review = rules["population"] == "needs_review"
rules.loc[pop_review, "validation_status"] = rules.loc[pop_review, "validation_status"].replace(["clean", "patched"], "needs_review")
add_note(rules.index[pop_review], "population_unmapped")

print("Population distribution AFTER normalization:")
print(rules["population"].value_counts().to_string())
print(f"\nRemapped: {remapped.sum()} | Unmapped (needs_review): {pop_review.sum()}")
print("\nRaw values that needed mapping:")
print(rules.loc[remapped | pop_review, "population_raw"].value_counts().head(15).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 3 — Known-fact patches (H0018 -> B8, etc.)

# COMMAND ----------

# H0018 is exclusively a BHRF (provider type B8) code in AHCCCS FFS.
h0018_no_ptype = rules["proc_codes"].fillna("").str.contains("H0018") & rules["provider_types"].isna()
rules.loc[h0018_no_ptype, "provider_types"] = "B8"
rules.loc[h0018_no_ptype, "validation_status"] = rules.loc[h0018_no_ptype, "validation_status"].replace("clean", "patched")
add_note(rules.index[h0018_no_ptype], "backfilled_provider_type_B8")
print(f"H0018 rules backfilled with provider_types=B8: {h0018_no_ptype.sum()}")

# H0018 rules must be bh_residential
h0018_wrong_pop = rules["proc_codes"].fillna("").str.contains("H0018") & ~rules["population"].isin(["bh_residential"])
rules.loc[h0018_wrong_pop, "population"] = "bh_residential"
rules.loc[h0018_wrong_pop, "validation_status"] = rules.loc[h0018_wrong_pop, "validation_status"].replace(["clean", "needs_review"], "patched")
add_note(rules.index[h0018_wrong_pop], "population_forced_bh_residential_for_H0018")
print(f"H0018 rules repopulated to bh_residential: {h0018_wrong_pop.sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 4 — machine_checkable derivation + rule_subtype routing
# MAGIC This is what defuses the engine's old "cannot" false-positive cannon.

# COMMAND ----------

def derive_checkable(row):
    if isinstance(row["machine_checkable"], (bool, np.bool_)):
        base = bool(row["machine_checkable"])
    else:
        base = True
    cond = (row["condition"] or "").lower()
    if any(h in cond for h in UNCHECKABLE_HINTS):
        # documentation_requirement thresholds ARE checkable (the trigger is unit counts)
        if row["rule_type"] == "documentation_requirement" and not pd.isna(row["threshold_numeric"]):
            return base
        return False
    return base

rules["machine_checkable"] = rules.apply(derive_checkable, axis=1)

# dx/age-scoped rules: the engine v2 does not yet consume member DOB or diagnosis columns.
# Until it does, these rules CANNOT be evaluated correctly against all claims — an
# age-restricted rule applied to every age would over-flag. Route to context.
for scope_col in ("diagnosis_codes", "age_min", "age_max"):
    if scope_col not in rules.columns:
        rules[scope_col] = None
dx_age_scoped = rules["diagnosis_codes"].notna() | rules["age_min"].notna() | rules["age_max"].notna()
n_forced = (dx_age_scoped & rules["machine_checkable"]).sum()
rules.loc[dx_age_scoped, "machine_checkable"] = False
add_note(rules.index[dx_age_scoped], "dx_or_age_scoped_context_only_until_engine_supports")
if n_forced:
    print(f"dx/age-scoped rules routed to context (engine lacks DOB/dx columns): {n_forced}")

def derive_subtype(row):
    """Deterministic routing for billing_prohibition rules."""
    if row["rule_type"] != "billing_prohibition":
        return None
    cond = (row["condition"] or "").lower()
    if "discharge" in cond:
        return "discharge_day"
    if "single line item" in cond or "range of service dates" in cond or "range of dates" in cond:
        return "date_range"
    if not row["machine_checkable"]:
        return "conditional"          # engine: context only, NEVER a deterministic violation
    return "absolute"                 # engine: any matching claim = violation (e.g., H0030)

rules["rule_subtype"] = rules.apply(derive_subtype, axis=1)

print("machine_checkable:", rules["machine_checkable"].value_counts().to_dict())
print("\nbilling_prohibition subtypes:")
print(rules.loc[rules["rule_type"] == "billing_prohibition", "rule_subtype"].value_counts().to_string())
print("\nABSOLUTE prohibitions (engine will flag EVERY matching claim — review this list!):")
absolute = rules[(rules["rule_subtype"] == "absolute")]
for _, r in absolute.iterrows():
    print(f"  [{r['rule_id']}] {str(r['proc_codes'])[:30]} :: {str(r['condition'])[:90]}")
# Absolute prohibitions are high-stakes: force human review for LLM-extracted ones
llm_absolute = (rules["rule_subtype"] == "absolute") & (rules["origin"] == "llm")
rules.loc[llm_absolute, "validation_status"] = rules.loc[llm_absolute, "validation_status"].replace(["clean", "patched"], "needs_review")
add_note(rules.index[llm_absolute], "llm_absolute_prohibition_requires_signoff")
print(f"\nLLM-extracted absolute prohibitions sent to review: {llm_absolute.sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 5 — Near-duplicate detection (windows/multi-doc re-extraction)

# COMMAND ----------

def _tokens(t):
    return set(re.findall(r'[a-z0-9]+', str(t).lower()))

def jaccard(a, b):
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0

dupes_dropped = []
active = rules[rules["validation_status"] != "quarantined"].copy()
keep_mask = pd.Series(True, index=rules.index)

groups = active.groupby(["rule_type", "proc_codes"], dropna=False)
for _, grp in groups:
    if len(grp) < 2:
        continue
    idxs = list(grp.index)
    for i in range(len(idxs)):
        if not keep_mask[idxs[i]]:
            continue
        for j in range(i + 1, len(idxs)):
            if not keep_mask[idxs[j]]:
                continue
            a, b = rules.loc[idxs[i]], rules.loc[idxs[j]]
            if jaccard(a["condition"], b["condition"]) >= 0.7:
                # Keep manual over llm; else keep the earlier one
                drop = idxs[j] if (a["origin"] == "manual" or b["origin"] != "manual") else idxs[i]
                keep_mask[drop] = False
                dupes_dropped.append((rules.loc[drop, "rule_id"], rules.loc[drop, "condition"][:60]))

rules.loc[~keep_mask, "validation_status"] = "quarantined"
add_note(rules.index[~keep_mask], "near_duplicate")
print(f"Near-duplicates quarantined: {(~keep_mask).sum()}")
for rid, cond in dupes_dropped[:15]:
    print(f"  {rid}: {cond}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 6 — Safety-net reconciliation + 9-point policy checklist

# COMMAND ----------

active = rules[~rules["validation_status"].isin(["quarantined"])]

n_manual = (active["origin"] == "manual").sum()
print(f"Manual safety-net rules present and active: {n_manual} (expected 14)")
if n_manual < 14:
    print("  *** FAILURE: safety-net rules missing. Do not proceed. ***")

checks = [
    ("H0018 discharge day prohibition",
     lambda df: ((df.proc_codes.fillna("").str.contains("H0018")) & (df.condition.fillna("").str.lower().str.contains("discharge"))).any()),
    ("H0018 POS 56 requirement",
     lambda df: ((df.proc_codes.fillna("").str.contains("H0018")) & (df.rule_type == "pos_requirement") & (df.threshold_numeric == 56)).any()),
    ("H0018+U9 frequency limit (2/yr)",
     lambda df: ((df.modifiers.fillna("").str.contains("U9")) & (df.threshold_numeric == 2)).any()),
    ("H0018 inclusive rate w/ related_codes",
     lambda df: ((df.rule_type == "inclusive_rate") & df.related_codes.notna()).any()),
    ("H0004 documentation threshold (4)",
     lambda df: ((df.proc_codes.fillna("").str.contains("H0004")) & (df.threshold_numeric == 4)).any()),
    ("H2019 documentation threshold (8)",
     lambda df: ((df.proc_codes.fillna("").str.contains("H2019")) & (df.threshold_numeric == 8)).any()),
    ("H0030 absolute prohibition",
     lambda df: ((df.proc_codes.fillna("").str.contains("H0030")) & (df.rule_subtype == "absolute")).any()),
    ("Single line item per DOS (date_range subtype)",
     lambda df: (df.rule_subtype == "date_range").any()),
    ("Prior auth requirement present",
     lambda df: (df.rule_type == "prior_auth_requirement").any()),
]

print("\nPOLICY CHECKLIST (against active rules):")
all_pass = True
for name, fn in checks:
    ok = bool(fn(active))
    all_pass &= ok
    print(f"  [{'PASS' if ok else 'MISSING':7s}] {name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 7 — (Optional) cross-reference rule codes against real claims

# COMMAND ----------

if CLAIMS_TABLE:
    claim_codes = set(
        spark.table(CLAIMS_TABLE).select(CLAIMS_PROC_COL).distinct().toPandas()[CLAIMS_PROC_COL].astype(str).str.upper()
    )
    def code_hit_rate(codes_str):
        if not codes_str:
            return None
        codes = codes_str.split("|")
        return sum(c in claim_codes for c in codes) / len(codes)
    active_idx = rules["validation_status"] != "quarantined"
    rules.loc[active_idx, "claims_code_hit_rate"] = rules.loc[active_idx, "proc_codes"].apply(code_hit_rate)
    zero_hit = active_idx & (rules["claims_code_hit_rate"] == 0.0)
    print(f"Rules whose codes NEVER appear in claims: {zero_hit.sum()} "
          f"(not wrong per se — but they'll never fire; review for extraction noise)")
else:
    rules["claims_code_hit_rate"] = None
    print("CLAIMS_TABLE not set — skipping claims cross-reference. "
          "Set it before Step 4 testing; it also verifies your proc code column mapping.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write validated table + verdict

# COMMAND ----------

out_cols = ["rule_id", "rule_label", "population", "population_raw", "proc_codes", "modifiers",
            "diagnosis_codes", "age_min", "age_max",
            "provider_types", "rule_type", "rule_subtype", "condition", "threshold_value",
            "threshold_numeric", "threshold_unit", "effective_date", "related_codes",
            "machine_checkable", "citation", "citation_page_exact", "publication_date",
            "source_url", "source_filename", "origin", "doc_hash", "validation_status",
            "validation_notes", "claims_code_hit_rate"]
for c in out_cols:
    if c not in rules.columns:
        rules[c] = None

out = rules[out_cols].copy()
out["validated_at"] = datetime.now(timezone.utc).isoformat()

sdf = spark.createDataFrame(out.astype(object).where(out.notna(), None))
sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(VALIDATED_TABLE)

print("=" * 60)
print("VALIDATION VERDICT")
print("=" * 60)
print(out["validation_status"].value_counts().to_string())
usable = out["validation_status"].isin(["clean", "patched"]).sum()
print(f"\nUsable now (clean+patched): {usable}")
print(f"Blocked pending review: {(out['validation_status']=='needs_review').sum()}")
print(f"Quarantined: {(out['validation_status']=='quarantined').sum()}")
print(f"\nGATE {'OPEN — proceed to Step 4' if all_pass and n_manual >= 14 and usable > 0 else 'CLOSED — fix MISSING checklist items / safety net first'}")
print(f"\nWritten to {VALIDATED_TABLE}. The engine reads ONLY clean+patched rules from this table.")
print("Review queue: SELECT * FROM " + VALIDATED_TABLE + " WHERE validation_status = 'needs_review'")
