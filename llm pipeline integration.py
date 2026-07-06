# Databricks notebook source
# MAGIC %md
# MAGIC # LLM Pipeline Integration — Step 6
# MAGIC
# MAGIC **Purpose**: Wire policy intelligence into the existing two-pass narrative pipeline
# MAGIC (`llm_summarization_v4_notebook.py`).
# MAGIC
# MAGIC **Pass 1 (deterministic synthesis)**: inject the rules-engine block
# MAGIC (violations + review triggers + context rules) into the structured inputs.
# MAGIC **Pass 2 (LLM narrative)**: inject retrieved `policy_chunks` passages, scoped by
# MAGIC population + the provider's violated codes, under a strict citation contract.
# MAGIC
# MAGIC **Retrieval**: Databricks Vector Search if available; TF-IDF fallback otherwise
# MAGIC (workspace VS availability was unconfirmed — the fallback keeps Step 6 unblocked).
# MAGIC
# MAGIC **Anti-hallucination contract**: the LLM may cite ONLY citations present in the
# MAGIC injected block. A post-generation check verifies this.

# COMMAND ----------

import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

ROOT = dbutils.widgets.get("ROOT") if "ROOT" in [w.name for w in dbutils.widgets.getAll()] else os.environ.get("ROOT", "your_email@nttdata.com")
prefix = open(f"/Workspace/Users/{ROOT}/config/prefix.txt").read().strip() if os.path.exists(f"/Workspace/Users/{ROOT}/config/prefix.txt") else ""
CATALOG_DB = f"{prefix}policy_intelligence" if prefix else "policy_intelligence"
CHUNKS_TABLE = f"{CATALOG_DB}.policy_chunks"

VS_ENDPOINT = None        # e.g. "policy_vs_endpoint" — set if Vector Search is enabled
VS_INDEX    = None        # e.g. f"{CATALOG_DB}.policy_chunks_index"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieval layer — Vector Search with TF-IDF fallback

# COMMAND ----------

class PolicyChunkRetriever:
    """
    Retrieves policy_chunks passages for narrative grounding.
    Prefers Databricks Vector Search; falls back to TF-IDF over the chunks table.
    """

    def __init__(self, spark, chunks_table: str = CHUNKS_TABLE,
                 vs_endpoint: Optional[str] = VS_ENDPOINT,
                 vs_index: Optional[str] = VS_INDEX):
        self.spark = spark
        self.chunks_table = chunks_table
        self.vs_endpoint, self.vs_index = vs_endpoint, vs_index
        self.mode = None
        self._init_backend()

    def _init_backend(self):
        if self.vs_endpoint and self.vs_index:
            try:
                from databricks.vector_search.client import VectorSearchClient
                self._vs = VectorSearchClient().get_index(
                    endpoint_name=self.vs_endpoint, index_name=self.vs_index)
                self.mode = "vector_search"
                print("Retriever: Databricks Vector Search")
                return
            except Exception as e:
                print(f"Vector Search unavailable ({e}); falling back to TF-IDF.")
        # TF-IDF fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._chunks = self.spark.table(self.chunks_table).toPandas()
        if "superseded_at" in self._chunks.columns:
            self._chunks = self._chunks[self._chunks["superseded_at"].isna()].reset_index(drop=True)
        self._chunks["chunk_text"] = self._chunks["chunk_text"].fillna("")
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=50000,
                                           ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(self._chunks["chunk_text"])
        self.mode = "tfidf"
        print(f"Retriever: TF-IDF fallback over {len(self._chunks)} chunks")

    def retrieve(self, query: str, k: int = 5,
                 source_urls: Optional[List[str]] = None) -> pd.DataFrame:
        """Returns top-k chunks: columns [chunk_id, source_title, section_title,
        chunk_text, page_start, page_end, page_exact, score]."""
        if self.mode == "vector_search":
            res = self._vs.similarity_search(
                query_text=query, num_results=k * 3,
                columns=["chunk_id", "source_title", "section_title", "chunk_text",
                         "page_start", "page_end", "page_exact", "source_url"])
            rows = res.get("result", {}).get("data_array", [])
            cols = ["chunk_id", "source_title", "section_title", "chunk_text",
                    "page_start", "page_end", "page_exact", "source_url", "score"]
            out = pd.DataFrame(rows, columns=cols[:len(rows[0])] if rows else cols)
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            qv = self._vectorizer.transform([query])
            sims = cosine_similarity(qv, self._matrix).ravel()
            out = self._chunks.copy()
            out["score"] = sims
            out = out.sort_values("score", ascending=False).head(k * 3)

        if source_urls:
            pref = out[out["source_url"].isin(source_urls)]
            rest = out[~out["source_url"].isin(source_urls)]
            out = pd.concat([pref, rest])
        return out.head(k).reset_index(drop=True)


def build_retrieval_query(evaluation_result: Dict, anomaly_signals: Optional[List[str]] = None) -> str:
    """Query = violated codes + violation types + top anomaly signals."""
    terms = []
    for v in evaluation_result.get("violations", [])[:5]:
        terms += str(v.get("proc_codes") or "").split("|")
        terms.append(str(v.get("violation_type") or "").replace("_", " "))
    for t in evaluation_result.get("review_triggers", [])[:3]:
        terms += str(t.get("proc_codes") or "").split("|")
    if anomaly_signals:
        terms += anomaly_signals[:5]
    terms = [t for t in dict.fromkeys(terms) if t]   # dedupe, preserve order
    return " ".join(terms) or evaluation_result.get("population", "")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Policy context assembly (Pass 1 + Pass 2 inputs)

# COMMAND ----------

def build_policy_context(engine, retriever, provider_id: str, claims_df: pd.DataFrame,
                         population: str, anomaly_signals: Optional[List[str]] = None,
                         k_chunks: int = 4, **engine_col_kwargs) -> Dict:
    """
    One call per flagged provider. Returns everything the two-pass pipeline needs:
    - 'pass1_block': deterministic engine output for structured synthesis
    - 'pass2_chunks_block': retrieved narrative policy passages
    - 'allowed_citations': the ONLY citations the LLM may use
    - 'evaluation_result': raw engine output (for features / storage)
    """
    result = engine.evaluate_provider(provider_id, claims_df, population, **engine_col_kwargs)
    pass1_block = engine.format_for_llm_synthesis(result)

    query = build_retrieval_query(result, anomaly_signals)
    chunks = retriever.retrieve(query, k=k_chunks)

    chunk_lines = ["RETRIEVED_POLICY_PASSAGES (verbatim source material — cite by [P#]):"]
    allowed_citations = set()
    for i, row in chunks.iterrows():
        page_note = "" if bool(row.get("page_exact", False)) else " (page approximate)"
        cite = f"{row['source_title']}, p.{row['page_start']}-{row['page_end']}{page_note}"
        allowed_citations.add(cite)
        chunk_lines.append(f"\n[P{i+1}] {cite}\n{str(row['chunk_text'])[:1800]}")
    pass2_chunks_block = "\n".join(chunk_lines)

    for v in result["violations"] + result["review_triggers"]:
        allowed_citations.add(str(v.get("citation", "")))
    for c in result["context_rules"]:
        allowed_citations.add(str(c.get("citation", "")))
    allowed_citations.discard("")

    return {
        "provider_id": provider_id,
        "evaluation_result": result,
        "pass1_block": pass1_block,
        "pass2_chunks_block": pass2_chunks_block,
        "allowed_citations": sorted(allowed_citations),
        "retrieval_query": query,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt contract for Pass 2 (append to your existing narrative system prompt)

# COMMAND ----------

POLICY_GROUNDING_INSTRUCTIONS = """
POLICY GROUNDING RULES (mandatory):
1. You are given POLICY_VIOLATIONS (deterministic findings), DOCUMENTATION_REVIEW_TRIGGERS,
   APPLICABLE_POLICY_CONTEXT, and RETRIEVED_POLICY_PASSAGES.
2. State deterministic violations as findings, with their citation, exactly as provided.
3. State review triggers as "documentation is required and should be requested" — they are
   NOT violations. Do not describe them as violations.
4. APPLICABLE_POLICY_CONTEXT rules could not be verified from claims data. You may say the
   billing pattern "warrants verification against" these policies. NEVER assert they were violated.
5. Cite ONLY citations that appear verbatim in the provided blocks. NEVER construct, infer,
   or embellish a citation. If a claim has no provided citation, state it without one.
6. If a citation is marked "(page approximate)", carry that marker into the narrative.
7. Statistical anomalies (z-scores, peer comparisons) and policy findings are DIFFERENT
   evidence types. Present anomalies as "unusual relative to peers" and policy findings as
   "inconsistent with policy," and never convert one into the other.
"""

def build_pass2_user_prompt(provider_summary: str, anomaly_block: str, ctx: Dict) -> str:
    """Compose the Pass 2 user message. provider_summary/anomaly_block come from your
    existing pipeline (Pass 1 outputs); ctx from build_policy_context()."""
    return f"""PROVIDER SUMMARY:
{provider_summary}

STATISTICAL ANOMALY FINDINGS:
{anomaly_block}

{ctx['pass1_block']}

{ctx['pass2_chunks_block']}

TASK: Write the investigator-facing narrative for this provider. Follow the POLICY
GROUNDING RULES exactly. Structure: (1) summary of concern, (2) deterministic policy
violations with citations, (3) statistical anomalies, (4) documentation to request,
(5) recommended next steps."""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Post-generation citation audit (hallucination guard)

# COMMAND ----------

def audit_citations(narrative: str, allowed_citations: List[str]) -> Dict:
    """
    Verify every citation-like string in the narrative exists in the allowed set.
    Heuristic: sentences referencing 'slide', 'p.', 'page', 'AMPM', 'Manual', 'Policy'
    near a source-title fragment.
    """
    problems = []
    allowed_norm = [re.sub(r"\s+", " ", c).lower() for c in allowed_citations]
    # Extract candidate citation phrases
    candidates = re.findall(
        r'([A-Z][A-Za-z0-9 ,&/\-\.]{8,90}(?:slide|slides|p\.|page|pages)\s*[\d\-\., ]+)',
        narrative)
    for cand in candidates:
        cn = re.sub(r"\s+", " ", cand).strip().lower()
        if not any(cn[:40] in a or a[:40] in cn for a in allowed_norm):
            problems.append(cand.strip())
    return {"passed": not problems, "unverified_citations": problems,
            "n_candidates": len(candidates)}

# COMMAND ----------

# MAGIC %md
# MAGIC ## End-to-end wiring example

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC # --- once per session ---
# MAGIC # %run ./policy_rules_engine_v2          # defines PolicyRulesEngine
# MAGIC engine = PolicyRulesEngine(spark, catalog_db=CATALOG_DB)
# MAGIC retriever = PolicyChunkRetriever(spark)
# MAGIC
# MAGIC flagged = spark.table("your_db.scored_providers_bh_residential").toPandas()
# MAGIC claims  = spark.table("your_db.bh_residential_claims").toPandas()
# MAGIC
# MAGIC # --- per flagged provider ---
# MAGIC pid = flagged.provider_id.iloc[0]
# MAGIC ctx = build_policy_context(
# MAGIC     engine, retriever, pid,
# MAGIC     claims[claims.provider_id == pid], "bh_residential",
# MAGIC     anomaly_signals=["H0018 utilization 3.2 sigma", "same-day H0004"])
# MAGIC
# MAGIC # Pass 1: append ctx['pass1_block'] to your deterministic synthesis inputs.
# MAGIC # It also enriches needs_llm_scheme_validation(): a provider with
# MAGIC # ctx['evaluation_result']['total_violations'] > 0 short-circuits to "validate".
# MAGIC
# MAGIC # Pass 2: your existing generate_with_harmony call, with the grounding contract:
# MAGIC messages = [
# MAGIC     {"role": "system", "content": YOUR_NARRATIVE_SYSTEM_PROMPT + POLICY_GROUNDING_INSTRUCTIONS},
# MAGIC     {"role": "user", "content": build_pass2_user_prompt(provider_summary, anomaly_block, ctx)},
# MAGIC ]
# MAGIC narrative, _ = generate_with_harmony(messages, max_new_tokens=1500)
# MAGIC
# MAGIC # Guard: reject/regenerate narratives with invented citations
# MAGIC audit = audit_citations(narrative, ctx["allowed_citations"])
# MAGIC if not audit["passed"]:
# MAGIC     print("CITATION AUDIT FAILED:", audit["unverified_citations"])
# MAGIC     # regenerate once with the failures appended as a correction instruction;
# MAGIC     # if it fails again, fall back to Pass 1 deterministic text only.
# MAGIC ```
# MAGIC
# MAGIC ### Vector Search index (when confirmed available)
# MAGIC ```python
# MAGIC # One-time setup — requires a VS endpoint in the workspace:
# MAGIC # from databricks.vector_search.client import VectorSearchClient
# MAGIC # vsc = VectorSearchClient()
# MAGIC # vsc.create_delta_sync_index(
# MAGIC #     endpoint_name="policy_vs_endpoint",
# MAGIC #     index_name=f"{CATALOG_DB}.policy_chunks_index",
# MAGIC #     source_table_name=f"{CATALOG_DB}.policy_chunks",
# MAGIC #     pipeline_type="TRIGGERED",
# MAGIC #     primary_key="chunk_id",
# MAGIC #     embedding_source_column="chunk_text",
# MAGIC #     embedding_model_endpoint_name="databricks-gte-large-en",
# MAGIC # )
# MAGIC # Then set VS_ENDPOINT / VS_INDEX at the top of this notebook.
# MAGIC ```
