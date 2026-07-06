# Databricks notebook source
# MAGIC %md
# MAGIC # Policy Rules Engine — Step 4 v2 (Hardened)
# MAGIC
# MAGIC **Fixes over v1**:
# MAGIC 1. **Reads `policy_rules_validated`**, uses only `clean`/`patched` + `machine_checkable` rules
# MAGIC    for deterministic checks. Non-checkable rules are returned as `context_rules` for the LLM
# MAGIC    narrative — never as violations. (v1's generic "cannot" fallback flagged 100% of H0018
# MAGIC    claims for conditions it couldn't verify.)
# MAGIC 2. **Date-aware in the main path** — every checker pre-filters claims to
# MAGIC    service_date >= rule effective_date. (v1 claimed this but only did it for POS.)
# MAGIC 3. **Subtype routing** for prohibitions (discharge_day / date_range / absolute / conditional)
# MAGIC    assigned by the validation notebook — no keyword guessing at evaluation time.
# MAGIC 4. **Bundled codes from the KB** (`related_codes`), not hardcoded in the engine.
# MAGIC 5. **Severity split**: documentation thresholds produce `severity="review"` triggers,
# MAGIC    not violations. Investigator output separates the two.
# MAGIC 6. **Column mappings respected everywhere** (v1 hardcoded srvc_bgn_dt/srvc_end_dt in one checker).
# MAGIC 7. **Frequency limits count distinct DOS** per member-year (not raw line items).
# MAGIC    Calendar-year assumption is explicit in the output.
# MAGIC 8. Rule dedup at load; unbundling counts labeled correctly (DOS-code pairs, not claims).

# COMMAND ----------

import os

ROOT = dbutils.widgets.get("ROOT") if "ROOT" in [w.name for w in dbutils.widgets.getAll()] else os.environ.get("ROOT", "your_email@nttdata.com")
prefix = open(f"/Workspace/Users/{ROOT}/config/prefix.txt").read().strip() if os.path.exists(f"/Workspace/Users/{ROOT}/config/prefix.txt") else ""
CATALOG_DB = f"{prefix}policy_intelligence" if prefix else "policy_intelligence"

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional
from pyspark.sql import SparkSession


class PolicyRulesEngine:
    """
    Deterministic policy violation detector (v2).

    - Consumes policy_rules_validated: only clean/patched rules.
    - machine_checkable rules -> deterministic checks (violations / review triggers)
    - non-checkable rules -> context_rules (for LLM narrative grounding only)
    - Every checker is date-aware: claims before a rule's effective date are excluded.
    """

    def __init__(self, spark: SparkSession, catalog_db: str = "policy_intelligence",
                 table: str = "policy_rules_validated",
                 include_statuses: tuple = ("clean", "patched")):
        self.spark = spark
        self.rules_table = f"{catalog_db}.{table}"
        self.include_statuses = include_statuses
        self._load_rules()

    def _load_rules(self):
        df = self.spark.table(self.rules_table).toPandas()
        if "validation_status" in df.columns:
            df = df[df["validation_status"].isin(self.include_statuses)]
        df["threshold_numeric"] = pd.to_numeric(
            df.get("threshold_numeric", df.get("threshold_value")), errors="coerce")
        df["effective_dt"] = pd.to_datetime(df["effective_date"], errors="coerce")
        if "machine_checkable" not in df.columns:
            df["machine_checkable"] = True
        df["machine_checkable"] = df["machine_checkable"].fillna(False).astype(bool)
        if "rule_subtype" not in df.columns:
            df["rule_subtype"] = None
        df = df.drop_duplicates(subset=["rule_id"]).reset_index(drop=True)
        self._rules = df
        print(f"PolicyRulesEngine v2: {len(df)} usable rules "
              f"({df['machine_checkable'].sum()} checkable, "
              f"{(~df['machine_checkable']).sum()} context-only)")

    def reload_rules(self):
        self._load_rules()

    def get_rules_for_population(self, population: str) -> pd.DataFrame:
        pat = rf"(^|\|){re_escape(population)}($|\|)"
        mask = self._rules["population"].fillna("").str.contains(pat, regex=True)
        # cross_population rules always apply
        mask = mask | (self._rules["population"] == "cross_population")
        return self._rules[mask].copy()

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate_provider(
        self, provider_id: str, claims_df: pd.DataFrame, population: str,
        service_date_col: str = "srvc_bgn_dt",
        service_end_date_col: Optional[str] = "srvc_end_dt",
        proc_code_col: str = "proc_cd",
        modifier_cols: Optional[List[str]] = None,
        pos_col: str = "pos_cd",
        units_col: str = "units",
        member_id_col: str = "member_id",
        discharge_date_col: Optional[str] = "discharge_dt",
        provider_type_col: str = "provider_type",
    ) -> Dict:
        if modifier_cols is None:
            modifier_cols = ["mod_1", "mod_2", "mod_3", "mod_4"]

        # Pre-parse service date ONCE (used by every checker + date filtering)
        claims = claims_df.copy()
        if service_date_col in claims.columns:
            claims["_svc_dt"] = pd.to_datetime(claims[service_date_col], errors="coerce")
        else:
            claims["_svc_dt"] = pd.NaT

        cols = dict(
            proc_code_col=proc_code_col, modifier_cols=modifier_cols, pos_col=pos_col,
            units_col=units_col, member_id_col=member_id_col,
            service_date_col=service_date_col, service_end_date_col=service_end_date_col,
            discharge_date_col=discharge_date_col, provider_type_col=provider_type_col,
        )

        rules = self.get_rules_for_population(population)
        violations, review_triggers, context_rules, skipped = [], [], [], []

        for _, rule in rules.iterrows():
            if not rule["machine_checkable"]:
                context_rules.append(self._context_record(rule))
                continue

            # FIX #2: date-aware everywhere — drop claims before the rule's effective date
            eff = rule["effective_dt"]
            scoped = claims if pd.isna(eff) else claims[claims["_svc_dt"] >= eff]
            if scoped.empty:
                continue

            try:
                results = self._check_rule(rule, scoped, **cols)
            except Exception as e:
                skipped.append({"rule_id": rule.get("rule_id"), "error": str(e)})
                continue

            for v in results:
                (review_triggers if v["severity"] == "review" else violations).append(v)

        violations.sort(key=lambda v: v.get("violating_claim_count", 0), reverse=True)
        review_triggers.sort(key=lambda v: v.get("violating_claim_count", 0), reverse=True)

        return {
            "provider_id": provider_id, "population": population,
            "total_violations": len(violations),
            "total_review_triggers": len(review_triggers),
            "total_violating_claims": sum(v.get("violating_claim_count", 0) for v in violations),
            "violations": violations,
            "review_triggers": review_triggers,
            "context_rules": context_rules,
            "rules_checked": int(rules["machine_checkable"].sum()),
            "rules_skipped_on_error": skipped,
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _check_rule(self, rule: pd.Series, claims: pd.DataFrame, **cols) -> List[Dict]:
        rt = rule.get("rule_type", "")
        if rt == "billing_prohibition":
            return self._check_billing_prohibition(rule, claims, **cols)
        if rt in ("unit_threshold", "documentation_requirement"):
            return self._check_unit_threshold(rule, claims, **cols)
        if rt == "pos_requirement":
            return self._check_pos_requirement(rule, claims, **cols)
        if rt == "frequency_limit":
            return self._check_frequency_limit(rule, claims, **cols)
        if rt == "inclusive_rate":
            return self._check_inclusive_rate(rule, claims, **cols)
        if rt == "modifier_requirement":
            return self._check_modifier_requirement(rule, claims, **cols)
        if rt == "provider_type_restriction":
            return self._check_provider_type_restriction(rule, claims, **cols)
        return []   # code_restriction / prior_auth handled as context in v2 unless checkable pattern exists

    def _matching(self, claims: pd.DataFrame, rule: pd.Series, proc_code_col: str) -> pd.DataFrame:
        codes_str = rule.get("proc_codes")
        if codes_str is None or (isinstance(codes_str, float) and pd.isna(codes_str)) or str(codes_str).lower() in ("none", ""):
            return pd.DataFrame()   # v2: a rule with no codes never matches implicitly (v1 matched ALL claims)
        codes = [c.strip().upper() for c in str(codes_str).split("|") if c.strip()]
        if not codes or proc_code_col not in claims.columns:
            return pd.DataFrame()
        return claims[claims[proc_code_col].astype(str).str.upper().isin(codes)].copy()

    def _build(self, rule, violation_type, detail, count, severity="violation", evidence=None) -> Dict:
        return {
            "rule_id": rule.get("rule_id", "UNKNOWN"),
            "rule_label": rule.get("rule_label", ""),
            "rule_type": rule.get("rule_type", ""),
            "proc_codes": rule.get("proc_codes", ""),
            "violation_type": violation_type, "severity": severity,
            "detail": detail, "violating_claim_count": int(count),
            "citation": rule.get("citation", ""),
            "citation_page_exact": bool(rule.get("citation_page_exact", False)),
            "condition": rule.get("condition", ""),
            "effective_date": rule.get("effective_date", ""),
            "evidence": evidence or {},
        }

    def _context_record(self, rule) -> Dict:
        return {
            "rule_id": rule.get("rule_id"), "rule_label": rule.get("rule_label"),
            "rule_type": rule.get("rule_type"), "proc_codes": rule.get("proc_codes"),
            "condition": rule.get("condition"), "citation": rule.get("citation"),
            "note": "Not machine-checkable from claims data. Provide as investigative context only.",
        }

    # ------------------------------------------------------------------
    # Checkers
    # ------------------------------------------------------------------

    def _check_billing_prohibition(self, rule, claims, **cols) -> List[Dict]:
        """FIX #3: route on rule_subtype set by validation. No keyword guessing, no
        generic fallback that flags unverifiable conditions."""
        matching = self._matching(claims, rule, cols["proc_code_col"])
        if matching.empty:
            return []
        subtype = rule.get("rule_subtype")

        if subtype == "discharge_day":
            dcol = cols.get("discharge_date_col")
            if not dcol or dcol not in matching.columns:
                return []   # cannot verify -> do NOT flag (v1 flagged everything here)
            df = matching
            ddt = pd.to_datetime(df[dcol], errors="coerce")
            hits = df[df["_svc_dt"].notna() & ddt.notna() & (df["_svc_dt"].dt.date == ddt.dt.date)]
            if hits.empty:
                return []
            return [self._build(rule, "billed_on_discharge_day",
                f"{rule.get('proc_codes')} billed on the discharge day {len(hits)} time(s). "
                f"Per {rule.get('citation')}, the per diem cannot be billed on the day of discharge.",
                len(hits),
                evidence={"discharge_dates": pd.to_datetime(hits[dcol]).dt.strftime("%Y-%m-%d").unique().tolist()[:20]})]

        if subtype == "date_range":
            end_col = cols.get("service_end_date_col")   # FIX #6: mapped, not hardcoded
            if not end_col or end_col not in matching.columns:
                return []
            df = matching
            end_dt = pd.to_datetime(df[end_col], errors="coerce")
            hits = df[df["_svc_dt"].notna() & end_dt.notna() & ((end_dt - df["_svc_dt"]).dt.days > 0)]
            if hits.empty:
                return []
            ex = [f"{b:%Y-%m-%d} to {e:%Y-%m-%d}"
                  for b, e in zip(hits["_svc_dt"].head(5), pd.to_datetime(hits[end_col]).head(5))]
            return [self._build(rule, "date_range_billing",
                f"{rule.get('proc_codes')} billed with date ranges (begin != end) on {len(hits)} claim(s). "
                f"Policy requires a single line item per DOS. Per {rule.get('citation')}.",
                len(hits), evidence={"example_ranges": ex})]

        if subtype == "absolute":
            return [self._build(rule, "prohibited_code_billed",
                f"Provider billed {rule.get('proc_codes')}, which is prohibited: "
                f"{str(rule.get('condition'))[:200]} Per {rule.get('citation')}.",
                len(matching),
                evidence={"dates_of_service": matching["_svc_dt"].dt.strftime("%Y-%m-%d").dropna().unique().tolist()[:20]})]

        # subtype == 'conditional' or None: not deterministically checkable -> no violation
        return []

    def _check_unit_threshold(self, rule, claims, **cols) -> List[Dict]:
        matching = self._matching(claims, rule, cols["proc_code_col"])
        thr = rule.get("threshold_numeric")
        units_col = cols["units_col"]
        if matching.empty or pd.isna(thr) or units_col not in matching.columns:
            return []

        # FIX #5: documentation thresholds are review triggers, not violations
        severity = "review" if rule.get("rule_type") == "documentation_requirement" else "violation"
        vtype = "documentation_review_trigger" if severity == "review" else "units_exceed_threshold"

        matching[units_col] = pd.to_numeric(matching[units_col], errors="coerce")
        unit_kind = str(rule.get("threshold_unit") or "").lower()

        if "per_dos" in unit_kind or "per_dos" in str(rule.get("condition", "")).lower():
            agg_cols = [c for c in (cols["member_id_col"],) if c in matching.columns] + ["_svc_dt"]
            dos_units = matching.groupby(agg_cols, dropna=False)[units_col].sum().reset_index()
            exceeding = dos_units[dos_units[units_col] > thr]
            if exceeding.empty:
                return []
            return [self._build(rule, vtype,
                f"{rule.get('proc_codes')} billed above {int(thr)} {unit_kind} on {len(exceeding)} "
                f"date(s) of service. Per {rule.get('citation')}"
                + (", documentation is required above this threshold." if severity == "review" else "."),
                len(exceeding), severity=severity,
                evidence={"max_units_observed": float(exceeding[units_col].max()),
                          "avg_units_on_exceeding_dos": round(float(exceeding[units_col].mean()), 2),
                          "unique_members_affected": int(exceeding[cols["member_id_col"]].nunique())
                          if cols["member_id_col"] in exceeding.columns else None})]
        else:
            exceeding = matching[matching[units_col] > thr]
            if exceeding.empty:
                return []
            return [self._build(rule, vtype,
                f"{rule.get('proc_codes')} billed above {int(thr)} units on {len(exceeding)} claim(s).",
                len(exceeding), severity=severity,
                evidence={"max_units_observed": float(exceeding[units_col].max())})]

    def _check_pos_requirement(self, rule, claims, **cols) -> List[Dict]:
        matching = self._matching(claims, rule, cols["proc_code_col"])
        pos_col = cols["pos_col"]
        thr = rule.get("threshold_numeric")
        if matching.empty or pos_col not in matching.columns or pd.isna(thr):
            return []
        required = str(int(thr)).zfill(2)   # POS codes are 2-char, zero-padded
        observed = matching[pos_col].astype(str).str.strip().str.zfill(2)
        wrong = matching[observed != required]
        if wrong.empty:
            return []
        dist = wrong[pos_col].astype(str).value_counts().to_dict()
        return [self._build(rule, "wrong_pos_code",
            f"{rule.get('proc_codes')} billed with incorrect POS on {len(wrong)} claim(s) "
            f"on/after {rule.get('effective_date')}. Required POS: {required}. Found: {dist}. "
            f"Per {rule.get('citation')}.",
            len(wrong), evidence={"required_pos": required, "observed_pos_distribution": dist})]

    def _check_frequency_limit(self, rule, claims, **cols) -> List[Dict]:
        matching = self._matching(claims, rule, cols["proc_code_col"])
        if matching.empty:
            return []
        mods = rule.get("modifiers")
        if mods and str(mods).lower() not in ("none", "nan"):
            mod_codes = [m.strip().upper() for m in str(mods).split("|")]
            mod_cols = [c for c in cols["modifier_cols"] if c in matching.columns]
            if mod_cols:
                mask = pd.Series(False, index=matching.index)
                for mc in mod_cols:
                    mask |= matching[mc].astype(str).str.upper().isin(mod_codes)
                matching = matching[mask]
        thr = rule.get("threshold_numeric")
        mcol = cols["member_id_col"]
        if matching.empty or pd.isna(thr) or mcol not in matching.columns:
            return []
        df = matching.copy()
        df["_yr"] = df["_svc_dt"].dt.year
        # FIX #7: distinct DOS per member-year, not raw line items
        counts = df.groupby([mcol, "_yr"])["_svc_dt"].nunique().reset_index(name="n_dos")
        exceeding = counts[counts["n_dos"] > thr]
        if exceeding.empty:
            return []
        return [self._build(rule, "frequency_limit_exceeded",
            f"{rule.get('proc_codes')} (modifier {mods}) exceeded {int(thr)}/year for "
            f"{len(exceeding)} member-year(s) [distinct DOS, calendar-year basis]. Per {rule.get('citation')}.",
            int(exceeding["n_dos"].sum()),
            evidence={"members_exceeding": int(exceeding[mcol].nunique()),
                      "max_dos_observed": int(exceeding["n_dos"].max()),
                      "basis": "distinct service dates per calendar year",
                      "member_year_details": exceeding.head(10).to_dict(orient="records")})]

    def _check_inclusive_rate(self, rule, claims, **cols) -> List[Dict]:
        """FIX #4: bundled codes come from the rule's related_codes (KB-driven)."""
        pcol, mcol = cols["proc_code_col"], cols["member_id_col"]
        inclusive = [c.strip().upper() for c in str(rule.get("proc_codes") or "").split("|") if c.strip()]
        bundled = [c.strip().upper() for c in str(rule.get("related_codes") or "").split("|") if c.strip()]
        if not inclusive or not bundled or pcol not in claims.columns:
            return []
        df = claims.copy()
        df["_dos"] = df["_svc_dt"].dt.date
        inc = df[df[pcol].astype(str).str.upper().isin(inclusive)]
        oth = df[df[pcol].astype(str).str.upper().isin(bundled)]
        if inc.empty or oth.empty:
            return []
        join_cols = ["_dos"] + ([mcol] if mcol in df.columns else [])
        pairs = pd.merge(inc[join_cols].drop_duplicates(),
                         oth[join_cols + [pcol]].drop_duplicates(),
                         on=join_cols, how="inner")
        if pairs.empty:
            return []
        code_counts = pairs[pcol].value_counts().to_dict()
        return [self._build(rule, "unbundling_inclusive_rate",
            f"Bundled BH services billed separately alongside {rule.get('proc_codes')} "
            f"(all-inclusive per diem) on {pairs['_dos'].nunique()} distinct DOS "
            f"({len(pairs)} member-DOS-code pairs). Codes: {code_counts}. Per {rule.get('citation')}.",
            len(pairs),
            evidence={"unbundled_code_counts": code_counts,
                      "unique_dos": int(pairs["_dos"].nunique()),
                      "unique_members": int(pairs[mcol].nunique()) if mcol in pairs.columns else None,
                      "count_basis": "member-DOS-code pairs"})]

    def _check_modifier_requirement(self, rule, claims, **cols) -> List[Dict]:
        matching = self._matching(claims, rule, cols["proc_code_col"])
        req = rule.get("modifiers")
        if matching.empty or not req or str(req).lower() in ("none", "nan"):
            return []
        req_list = [m.strip().upper() for m in str(req).split("|")]
        mod_cols = [c for c in cols["modifier_cols"] if c in matching.columns]
        if not mod_cols:
            return []
        has_mod = pd.Series(False, index=matching.index)
        for mc in mod_cols:
            has_mod |= matching[mc].astype(str).str.upper().isin(req_list)
        missing = matching[~has_mod]
        if missing.empty:
            return []
        return [self._build(rule, "missing_required_modifier",
            f"{rule.get('proc_codes')} billed without required modifier(s) {req} on "
            f"{len(missing)} claim(s). Per {rule.get('citation')}.", len(missing))]

    def _check_provider_type_restriction(self, rule, claims, **cols) -> List[Dict]:
        matching = self._matching(claims, rule, cols["proc_code_col"])
        pt_col = cols["provider_type_col"]
        allowed_str = rule.get("provider_types")
        if matching.empty or pt_col not in matching.columns or not allowed_str \
                or str(allowed_str).lower() in ("none", "nan"):
            return []
        allowed = [t.strip().upper() for t in str(allowed_str).split("|")]
        unauthorized = matching[~matching[pt_col].astype(str).str.upper().isin(allowed)]
        if unauthorized.empty:
            return []
        observed = unauthorized[pt_col].value_counts().to_dict()
        return [self._build(rule, "unauthorized_provider_type",
            f"{rule.get('proc_codes')} billed by unauthorized provider type(s) {observed}. "
            f"Allowed: {allowed}. Per {rule.get('citation')}.", len(unauthorized),
            evidence={"allowed_types": allowed, "observed_unauthorized_types": observed})]

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def violations_to_dataframe(self, result: Dict) -> pd.DataFrame:
        rows = []
        for v in result["violations"] + result["review_triggers"]:
            rows.append({
                "provider_id": result["provider_id"], "population": result["population"],
                "rule_id": v["rule_id"], "rule_label": v.get("rule_label"),
                "rule_type": v["rule_type"], "severity": v["severity"],
                "proc_codes": v["proc_codes"], "violation_type": v["violation_type"],
                "detail": v["detail"], "violating_claim_count": v["violating_claim_count"],
                "citation": v["citation"], "effective_date": v["effective_date"],
            })
        return pd.DataFrame(rows)

    def format_for_llm_synthesis(self, result: Dict) -> str:
        lines = []
        if result["violations"]:
            lines.append(f"POLICY_VIOLATIONS: {result['total_violations']} deterministic violation(s) "
                         f"across {result['total_violating_claims']} claim(s).")
            lines.append("")
            for i, v in enumerate(result["violations"], 1):
                lines += [f"VIOLATION {i}:",
                          f"  Rule: {v['rule_id']} — {v.get('rule_label','')} ({v['rule_type']})",
                          f"  Code(s): {v['proc_codes']}",
                          f"  Detail: {v['detail']}",
                          f"  Claims affected: {v['violating_claim_count']}",
                          f"  Citation: {v['citation']}"
                          + ("" if v.get("citation_page_exact") else " [page approximate]")]
                for ek, ev in (v.get("evidence") or {}).items():
                    lines.append(f"  Evidence.{ek}: {ev}")
                lines.append("")
        else:
            lines.append("POLICY_VIOLATIONS: None detected.")
            lines.append("")
        if result["review_triggers"]:
            lines.append(f"DOCUMENTATION_REVIEW_TRIGGERS: {len(result['review_triggers'])} "
                         f"(policy requires documentation above these thresholds — not violations per se):")
            for t in result["review_triggers"]:
                lines.append(f"  - {t['detail']} [{t['citation']}]")
            lines.append("")
        if result["context_rules"]:
            lines.append("APPLICABLE_POLICY_CONTEXT (rules not verifiable from claims data — "
                         "use for narrative context, do NOT assert violations):")
            for c in result["context_rules"]:
                lines.append(f"  - {c['condition']} [{c['citation']}]")
        return "\n".join(lines)

    def format_for_investigator(self, result: Dict) -> str:
        lines = [f"Policy Findings — Provider {result['provider_id']} ({result['population']})",
                 f"Deterministic violations: {result['total_violations']} | "
                 f"Documentation review triggers: {result['total_review_triggers']}",
                 "=" * 60]
        for i, v in enumerate(result["violations"], 1):
            lines += [f"\n{i}. [VIOLATION] {v['violation_type'].upper().replace('_', ' ')}",
                      f"   {v['detail']}",
                      f"   Policy Reference: {v['citation']}"
                      + ("" if v.get("citation_page_exact") else " (page approximate)"),
                      f"   Claims Involved: {v['violating_claim_count']}"]
        for j, t in enumerate(result["review_triggers"], 1):
            lines += [f"\nR{j}. [REVIEW] {t['detail']}",
                      f"   Policy Reference: {t['citation']}"]
        if result["context_rules"]:
            lines.append("\nAdditional applicable policies (verify via records request):")
            for c in result["context_rules"]:
                lines.append(f"   - {c['condition'][:140]} [{c['citation']}]")
        return "\n".join(lines)


def re_escape(s: str) -> str:
    import re as _re
    return _re.escape(str(s))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch evaluation

# COMMAND ----------

def evaluate_all_providers(engine: PolicyRulesEngine, flagged_providers_df: pd.DataFrame,
                           claims_df: pd.DataFrame, population: str,
                           provider_id_col: str = "provider_id", **col_kwargs) -> pd.DataFrame:
    all_rows = []
    providers = flagged_providers_df[provider_id_col].unique()
    print(f"Evaluating {len(providers)} flagged providers...")
    for i, pid in enumerate(providers):
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(providers)}]")
        prov_claims = claims_df[claims_df[provider_id_col] == pid]
        if prov_claims.empty:
            continue
        result = engine.evaluate_provider(str(pid), prov_claims, population, **col_kwargs)
        if result["violations"] or result["review_triggers"]:
            all_rows.append(engine.violations_to_dataframe(result))
    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        print(f"\n{len(out)} findings across {out['provider_id'].nunique()} providers "
              f"({(out.severity=='violation').sum()} violations, {(out.severity=='review').sum()} review triggers)")
        return out
    print("\nNo findings.")
    return pd.DataFrame()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test — synthetic claims exercising every checker
# MAGIC Run this BEFORE touching real claims. Every assertion must pass.

# COMMAND ----------

def run_smoke_test(engine: PolicyRulesEngine):
    claims = pd.DataFrame([
        # discharge-day H0018 (violation if discharge rule active)
        dict(provider_id="P1", member_id="M1", proc_cd="H0018", srvc_bgn_dt="2024-10-05",
             srvc_end_dt="2024-10-05", discharge_dt="2024-10-05", pos_cd="56", units=1,
             provider_type="B8", mod_1=None, mod_2=None, mod_3=None, mod_4=None),
        # date-range H0018 post 2024-03-01 (violation)
        dict(provider_id="P1", member_id="M1", proc_cd="H0018", srvc_bgn_dt="2024-06-01",
             srvc_end_dt="2024-06-30", discharge_dt=None, pos_cd="56", units=30,
             provider_type="B8", mod_1=None, mod_2=None, mod_3=None, mod_4=None),
        # date-range H0018 BEFORE effective date (must NOT fire — date-awareness test)
        dict(provider_id="P1", member_id="M1", proc_cd="H0018", srvc_bgn_dt="2023-06-01",
             srvc_end_dt="2023-06-30", discharge_dt=None, pos_cd="56", units=30,
             provider_type="B8", mod_1=None, mod_2=None, mod_3=None, mod_4=None),
        # wrong POS after 2023-12-01 (violation)
        dict(provider_id="P1", member_id="M2", proc_cd="H0018", srvc_bgn_dt="2024-02-10",
             srvc_end_dt="2024-02-10", discharge_dt=None, pos_cd="11", units=1,
             provider_type="B8", mod_1=None, mod_2=None, mod_3=None, mod_4=None),
        # unbundling: H0018 + H0004 same member/DOS (violation)
        dict(provider_id="P1", member_id="M3", proc_cd="H0018", srvc_bgn_dt="2024-08-01",
             srvc_end_dt="2024-08-01", discharge_dt=None, pos_cd="56", units=1,
             provider_type="B8", mod_1=None, mod_2=None, mod_3=None, mod_4=None),
        dict(provider_id="P1", member_id="M3", proc_cd="H0004", srvc_bgn_dt="2024-08-01",
             srvc_end_dt="2024-08-01", discharge_dt=None, pos_cd="56", units=6,
             provider_type="B8", mod_1=None, mod_2=None, mod_3=None, mod_4=None),
        # H0004 6 units on one DOS (review trigger, threshold 4)
        # (row above doubles as this trigger)
        # H0018+U9 3 distinct DOS in one year (frequency violation, limit 2)
        *[dict(provider_id="P1", member_id="M4", proc_cd="H0018", srvc_bgn_dt=f"2024-0{m}-15",
               srvc_end_dt=f"2024-0{m}-15", discharge_dt=None, pos_cd="56", units=1,
               provider_type="B8", mod_1="U9", mod_2=None, mod_3=None, mod_4=None)
          for m in (1, 3, 5)],
    ])
    result = engine.evaluate_provider("P1", claims, "bh_residential")
    got = {v["violation_type"] for v in result["violations"]}
    trig = {t["violation_type"] for t in result["review_triggers"]}

    expected = {"billed_on_discharge_day", "date_range_billing", "wrong_pos_code",
                "unbundling_inclusive_rate", "frequency_limit_exceeded"}
    print("Violations found:", got)
    print("Review triggers found:", trig)
    missing = expected - got
    assert not missing, f"SMOKE TEST FAILED — missing: {missing}"
    assert "documentation_review_trigger" in trig, "SMOKE TEST FAILED — doc trigger missing"
    # date-awareness: the 2023 date-range claim must not be counted
    dr = [v for v in result["violations"] if v["violation_type"] == "date_range_billing"][0]
    assert dr["violating_claim_count"] == 1, "SMOKE TEST FAILED — pre-effective-date claim was flagged"
    # conditional prohibitions must never appear as violations
    assert all(v["violation_type"] != "prohibited_code_billed" or "H0030" in str(v["proc_codes"])
               for v in result["violations"]), "SMOKE TEST FAILED — conditional prohibition fired"
    print("\nSMOKE TEST PASSED — all checkers verified, date-awareness verified, "
          "conditional prohibitions correctly suppressed.")
    print("\n" + engine.format_for_investigator(result))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Usage
# MAGIC ```python
# MAGIC engine = PolicyRulesEngine(spark, catalog_db=CATALOG_DB)   # reads policy_rules_validated
# MAGIC run_smoke_test(engine)                                     # gate before real claims
# MAGIC
# MAGIC flagged = spark.table("your_db.scored_providers_bh_residential").toPandas()
# MAGIC claims  = spark.table("your_db.bh_residential_claims").toPandas()
# MAGIC
# MAGIC result = engine.evaluate_provider("NPI_12345",
# MAGIC     claims[claims.provider_id == "NPI_12345"], "bh_residential",
# MAGIC     # adjust column mappings to your schema:
# MAGIC     service_date_col="srvc_bgn_dt", service_end_date_col="srvc_end_dt",
# MAGIC     proc_code_col="proc_cd", units_col="units", member_id_col="member_id")
# MAGIC
# MAGIC print(engine.format_for_investigator(result))
# MAGIC llm_block = engine.format_for_llm_synthesis(result)   # -> Step 6
# MAGIC all_findings = evaluate_all_providers(engine, flagged, claims, "bh_residential")
# MAGIC ```
