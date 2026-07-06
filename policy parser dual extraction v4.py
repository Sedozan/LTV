# Databricks notebook source
# MAGIC %md
# MAGIC # Policy Parser — Step 2 v4: Dual Extraction (Hardened + Versioned)
# MAGIC
# MAGIC **New in v4**:
# MAGIC - **Rule versioning** — re-extraction of an updated document SUPERSEDES prior rows
# MAGIC   (`superseded_at` timestamp) instead of deleting them. AHCCCS overwrites PDFs in place;
# MAGIC   deletion destroyed the history needed to answer "was this rule in effect when the
# MAGIC   claim was billed?" All readers must filter `superseded_at IS NULL` for current rules.
# MAGIC - **Content-hash change detection** — `doc_hash` stored per document; unchanged docs
# MAGIC   are skipped even when REPROCESS_UPDATED=True.
# MAGIC - **HTML ingestion route** — AMPM chapters and other AHCCCS pages published as HTML are
# MAGIC   segmented from real heading tags (BeautifulSoup). Citations use section titles
# MAGIC   (HTML has no page numbers).
# MAGIC - **Extraction schema extended** — `diagnosis_codes` (ICD-10), `age_min`, `age_max`.
# MAGIC   Note: the engine can only enforce these if the claims extract carries dx codes and
# MAGIC   member DOB; otherwise they surface as context rules.
# MAGIC
# MAGIC **Fixes over v2** (each maps to an identified defect):
# MAGIC 1. **Document-driven extraction** — extracts once per document (tracked in `policy_processed_docs`),
# MAGIC    not per population. Population is assigned per-RULE against a controlled vocabulary.
# MAGIC    Population is now a query-time filter, not an extraction boundary.
# MAGIC 2. **Safety-net dedup fixed** — v2's `proc_codes|rule_type` key silently dropped 3 of 4 H0018
# MAGIC    billing prohibitions. Manual rules are now authoritative: ALL are kept; LLM rules that
# MAGIC    duplicate a manual rule (token-Jaccard on condition) are dropped instead.
# MAGIC 3. **No silent truncation** — long sections are windowed (4,500 chars, 500 overlap) so the
# MAGIC    LLM sees all text. v2 silently discarded everything past 5,000 chars.
# MAGIC 4. **Error persistence** — every failed section lands in `policy_extraction_errors` with its
# MAGIC    full text, enabling targeted re-extraction (e.g., on GPU with 120B). No more unknown error rate.
# MAGIC 5. **Checkpointing** — results append to Delta after EVERY document. A crash at hour 11
# MAGIC    loses one document, not the run. Re-runs skip completed docs (idempotent delete-then-append).
# MAGIC 6. **Deterministic, globally unique rule_id** — content hash. LLM's descriptive name kept as `rule_label`.
# MAGIC 7. **effective_date honesty** — only populated when stated in the text. Publication date is a
# MAGIC    separate column. v2 conflated them, corrupting temporal filtering downstream.
# MAGIC 8. **Enforceability tagging** — rules are marked machine-checkable vs contextual, so the rules
# MAGIC    engine never fires deterministic violations from conditions it cannot verify in claims data.
# MAGIC 9. **Robust JSON parsing** — balanced-bracket extraction with string awareness, plus one
# MAGIC    self-repair retry, instead of fragile non-greedy regex.
# MAGIC 10. **Schema validation** — rule_type whitelist, HCPCS/CPT format checks, numeric coercion.
# MAGIC     Invalid rules are quarantined to the error table, never silently written.
# MAGIC 11. **Typed Delta writes** — proper schema instead of astype(str).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os

ROOT = dbutils.widgets.get("ROOT") if "ROOT" in [w.name for w in dbutils.widgets.getAll()] else os.environ.get("ROOT", "your_email@nttdata.com")

project_name = open(f"/Workspace/Users/{ROOT}/project.txt").read().strip()
prefix = open(f"/Workspace/Users/{ROOT}/config/prefix.txt").read().strip() if os.path.exists(f"/Workspace/Users/{ROOT}/config/prefix.txt") else ""

CATALOG_DB = f"{prefix}policy_intelligence" if prefix else "policy_intelligence"
REGISTRY_TABLE   = f"{CATALOG_DB}.policy_registry"
RULES_TABLE      = f"{CATALOG_DB}.policy_rules"
CHUNKS_TABLE     = f"{CATALOG_DB}.policy_chunks"
PROCESSED_TABLE  = f"{CATALOG_DB}.policy_processed_docs"
ERRORS_TABLE     = f"{CATALOG_DB}.policy_extraction_errors"

# Extraction scope: documents, ordered by priority. NOT a population boundary.
# Set MAX_DOCS to limit a run; re-runs resume where the processed log left off.
MIN_INGESTION_PRIORITY = 8      # this run: bh_residential(10) + bh_outpatient(9) docs first
MAX_DOCS_THIS_RUN      = 50     # safety valve for long runs
REPROCESS_UPDATED      = False  # True: re-download processed docs, re-extract if doc_hash changed
                                # (prior rows are superseded, not deleted — history preserved)

MODEL_ID = "openai/gpt-oss-120b"   # use 20b only if GPU unavailable; expect lower field fidelity

# Controlled population vocabulary — the ONLY values allowed in policy_rules.population
POPULATION_VOCAB = {
    "bh_residential", "bh_outpatient", "dme", "nemt",
    "nursing_home", "professional", "cross_population",
}
# Synonym mapping for LLM-invented sub-populations. Unmapped values -> 'needs_review'
POPULATION_SYNONYMS = {
    "bhrf": "bh_residential",
    "behavioral_health_residential": "bh_residential",
    "bh_residential_facility": "bh_residential",
    "bh_assessment": "bh_outpatient",
    "bh_partial_hospitalization": "bh_outpatient",
    "behavioral_health_outpatient": "bh_outpatient",
    "bh_crisis": "bh_outpatient",
    "durable_medical_equipment": "dme",
    "non_emergency_medical_transportation": "nemt",
    "snf": "nursing_home",
    "all": "cross_population",
    "general": "cross_population",
}

VALID_RULE_TYPES = {
    "billing_prohibition", "unit_threshold", "documentation_requirement",
    "code_restriction", "provider_type_restriction", "modifier_requirement",
    "pos_requirement", "frequency_limit", "prior_auth_requirement", "inclusive_rate",
}

WINDOW_CHARS   = 4500
WINDOW_OVERLAP = 500

print(f"DB: {CATALOG_DB} | model: {MODEL_ID}")
print(f"Doc scope: priority >= {MIN_INGESTION_PRIORITY}, max {MAX_DOCS_THIS_RUN} docs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model

# COMMAND ----------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from openai_harmony import load_harmony_encoding, HarmonyEncodingName

with open(f'/Workspace/Users/{ROOT}/Projects/all_projects_predictions/data/hf_token.txt', 'r') as f:
    login(f.read().strip())

print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto", trust_remote_code=True
)
harmony_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
print("Model + harmony encoder loaded.")

# COMMAND ----------

def generate_with_harmony(messages, max_new_tokens=2000):
    """Generate with GPT-OSS; parse harmony channels. Returns (final_text, analysis_text)."""
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
    )
    input_ids = inputs.to(model.device) if isinstance(inputs, torch.Tensor) else inputs["input_ids"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None,
        )
    new_token_ids = output_ids[0][input_ids.shape[1]:].tolist()

    try:
        parsed = harmony_enc.parse_messages_from_completion_tokens(new_token_ids, role="assistant")
        final_text, analysis_text = "", ""
        for msg in parsed:
            channel = str(getattr(msg, 'channel', None) or getattr(msg, 'type', ''))
            content = str(getattr(msg, 'content', '') or getattr(msg, 'text', '') or msg)
            if channel == 'final':
                final_text = content
            elif channel == 'analysis':
                analysis_text += content + " "
        if final_text:
            return final_text.strip(), analysis_text.strip()
    except Exception:
        pass

    raw = tokenizer.decode(new_token_ids, skip_special_tokens=False)
    if '<|channel|>final<|message|>' in raw:
        part = raw.split('<|channel|>final<|message|>')[-1]
        for stop in ['<|return|>', '<|end|>', '<|start|>']:
            part = part.split(stop)[0]
        return part.strip(), ""

    clean = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    if clean.startswith('analysis'):
        for marker in ['finalProvider', 'final\n', 'assistantfinal']:
            if marker in clean:
                return clean.split(marker, 1)[1].strip(), ""
        return "", clean
    return clean.strip(), ""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Document Selection — priority-ordered, resume-aware
# MAGIC Population no longer gates extraction. We take the highest-priority unprocessed documents.

# COMMAND ----------

from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.getOrCreate()

registry_df = spark.table(REGISTRY_TABLE).toPandas()

# Resume awareness: skip documents already fully processed.
# With REPROCESS_UPDATED=True, processed docs stay in scope; they are re-downloaded and
# re-extracted ONLY if their content hash changed (prior rows superseded, never deleted).
try:
    processed = spark.table(PROCESSED_TABLE).toPandas()
    complete = processed[processed["status"] == "complete"].sort_values("processed_at")
    done_urls = set(complete["source_url"])
    known_hashes = (complete.dropna(subset=["doc_hash"]).groupby("source_url")["doc_hash"].last().to_dict()
                    if "doc_hash" in complete.columns else {})
except Exception:
    done_urls, known_hashes = set(), {}

scope = pd.to_numeric(registry_df["ingestion_priority"], errors="coerce").fillna(0) >= MIN_INGESTION_PRIORITY
if not REPROCESS_UPDATED:
    scope &= ~registry_df["url"].isin(done_urls)

candidates = registry_df[scope].sort_values(
    "ingestion_priority", ascending=False).head(MAX_DOCS_THIS_RUN).reset_index(drop=True)

print(f"Registry: {len(registry_df)} docs | already complete: {len(done_urls)} | this run: {len(candidates)}"
      + (" (incl. hash-check of processed docs)" if REPROCESS_UPDATED else ""))
for _, r in candidates.iterrows():
    print(f"  [p{r['ingestion_priority']}] [{str(r['doc_type'])[:22]:22s}] {str(r['title'])[:65]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download & Segmentation — PDF and HTML routes

# COMMAND ----------

import requests
import pdfplumber
import io
import time
import re
import json
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Optional

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("WARNING: beautifulsoup4 not installed — HTML docs will be skipped. "
          "Run: %pip install beautifulsoup4")

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _fetch(url: str) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": "AHCCCS-PolicyParser/1.0"})
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"  DOWNLOAD ERROR {url}: {e}")
        return None

def is_pdf_response(url: str, resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "").lower()
    return url.lower().endswith(".pdf") or "application/pdf" in ctype \
        or resp.content[:5] == b"%PDF-"

def download_document(url: str) -> Optional[Dict]:
    """Fetch a policy document (PDF or HTML). Returns unified dict with:
    source_format, doc_hash (md5 of raw bytes — drives change detection),
    and either page_texts (PDF) or html_soup (HTML)."""
    resp = _fetch(url)
    if resp is None:
        return None
    doc_hash = hashlib.md5(resp.content).hexdigest()

    if is_pdf_response(url, resp):
        try:
            page_texts = []
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        page_texts.append({"page_num": i + 1, "text": text.strip(),
                                           "char_count": len(text.strip())})
            return {"url": url, "source_format": "pdf", "doc_hash": doc_hash,
                    "total_pages": len(page_texts), "page_texts": page_texts,
                    "total_chars": sum(p["char_count"] for p in page_texts)}
        except Exception as e:
            print(f"  PDF PARSE ERROR {url}: {e}")
            return None

    # HTML route
    if not HAS_BS4:
        print(f"  SKIP (HTML, bs4 missing): {url}")
        return None
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        return {"url": url, "source_format": "html", "doc_hash": doc_hash,
                "total_pages": 0, "html_soup": soup,
                "total_chars": len(soup.get_text())}
    except Exception as e:
        print(f"  HTML PARSE ERROR {url}: {e}")
        return None


def segment_html(doc: Dict) -> List[Dict]:
    """Segment an HTML document on real heading tags (h1-h4). No fake page numbers —
    citations use section titles; page_start/page_end are null, page_exact=False."""
    soup = doc["html_soup"]
    headings = soup.find_all(["h1", "h2", "h3", "h4"])
    sections = []

    def _mk(title, text):
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) < 50:
            return None
        return {"section_title": title[:200], "section_text": f"{title}\n{text}",
                "body_text": text, "start_page": None, "end_page": None,
                "page_exact": False, "html_section": True}

    if not headings:
        s = _mk(doc["title"], soup.get_text(separator="\n"))
        return [s] if s else []

    # Preamble before the first heading
    full = soup.get_text(separator="\n")
    heading_titles = [h.get_text(strip=True) for h in headings]
    positions = []
    cursor = 0
    for t in heading_titles:
        p = full.find(t, cursor)
        positions.append(p)
        if p != -1:
            cursor = p + len(t)

    first_pos = next((p for p in positions if p != -1), 0)
    s = _mk("Introduction / Preamble", full[:first_pos])
    if s:
        sections.append(s)

    for i, (title, pos) in enumerate(zip(heading_titles, positions)):
        if pos == -1 or not title:
            continue
        end_pos = next((p for p in positions[i + 1:] if p != -1), len(full))
        s = _mk(title, full[pos + len(title):end_pos])
        if s:
            sections.append(s)
    return sections


def segment_training_deck(doc: Dict) -> List[Dict]:
    """Page-level segmentation with '(cont.)' merging. Page numbers are EXACT for decks."""
    sections, current = [], None
    for page in doc["page_texts"]:
        text = page["text"]
        title_line = ""
        for line in text.split("\n"):
            c = line.strip()
            if c and len(c) > 3 and not c.isdigit():
                title_line = c
                break
        section = {"section_title": title_line[:200], "section_text": text,
                   "body_text": text, "start_page": page["page_num"],
                   "end_page": page["page_num"], "page_exact": True}
        if current and any(x in title_line.lower() for x in ["(cont", "cont."]):
            current["section_text"] += "\n\n" + text
            current["body_text"] += "\n\n" + text
            current["end_page"] = page["page_num"]
        else:
            if current:
                sections.append(current)
            current = section
    if current:
        sections.append(current)
    return sections


def segment_manual_chapter(doc: Dict) -> List[Dict]:
    """Heading-based segmentation. Page numbers are ESTIMATED (flagged page_exact=False)."""
    full_text = "\n\n".join(p["text"] for p in doc["page_texts"])
    heading_pattern = re.compile(
        r'\n([A-Z]\.\s+[A-Z][A-Z\s/&,()-]+)\n'
        r'|\n((?:SECTION|CHAPTER|POLICY)\s+\d+[A-Z]?\s*[-–]?\s*[A-Z][A-Z\s/&,()-]+)\n'
        r'|\n(\d+\.\s+[A-Z][A-Za-z\s/&,()-]+)\n',
        re.MULTILINE)
    matches = list(heading_pattern.finditer(full_text))
    if not matches:
        return [{"section_title": doc["title"], "section_text": full_text,
                 "body_text": full_text, "start_page": 1,
                 "end_page": doc["total_pages"], "page_exact": False}]
    sections = []
    if matches[0].start() > 100:
        head = full_text[:matches[0].start()].strip()
        sections.append({"section_title": "Introduction / Preamble", "section_text": head,
                         "body_text": head, "start_page": 1, "end_page": 1, "page_exact": False})
    avg_cpp = doc["total_chars"] / max(doc["total_pages"], 1)
    for i, m in enumerate(matches):
        heading = (m.group(1) or m.group(2) or m.group(3)).strip()
        start, end = m.end(), matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        est_page = max(1, int(m.start() / avg_cpp) + 1)
        sections.append({"section_title": heading[:200],
                         "section_text": f"{heading}\n{body}", "body_text": body,
                         "start_page": est_page,
                         "end_page": min(est_page + 1, doc["total_pages"]),
                         "page_exact": False})
    return sections


def window_section(section: Dict) -> List[Dict]:
    """FIX #3: split long sections into overlapping windows so no text is silently dropped."""
    body = section["body_text"]
    if len(body) <= WINDOW_CHARS:
        return [dict(section, window_idx=0, n_windows=1)]
    windows, start, idx = [], 0, 0
    while start < len(body):
        chunk = body[start:start + WINDOW_CHARS]
        w = dict(section)
        w["body_text"] = chunk
        w["window_idx"] = idx
        w["n_windows"] = -1  # filled below
        windows.append(w)
        if start + WINDOW_CHARS >= len(body):
            break
        start += WINDOW_CHARS - WINDOW_OVERLAP
        idx += 1
    for w in windows:
        w["n_windows"] = len(windows)
    return windows


def segment_document(doc: Dict) -> List[Dict]:
    if doc.get("source_format") == "html":
        sections = segment_html(doc)
    elif doc["doc_type"] in ("training_presentation", "training_schedule"):
        sections = segment_training_deck(doc)
    elif doc["doc_type"] in ("manual_chapter", "ampm_policy", "service_guide"):
        sections = segment_manual_chapter(doc)
    else:
        sections = segment_training_deck(doc)
    out = []
    for s in sections:
        for w in window_section(s):
            w.update({
                "source_url": doc["url"], "source_title": doc["title"],
                "source_key": doc["source_key"], "doc_type": doc["doc_type"],
                "publication_date": doc.get("publication_date"),
                "filename": doc["filename"],
            })
            out.append(w)
    return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extraction Prompt — rule-level population, stated-date honesty, checkability

# COMMAND ----------

CLASSIFICATION_SYSTEM_PROMPT = """You are a Medicaid policy analyst extracting structured rules from AHCCCS policy documents.

Given a section of an AHCCCS policy document, do TWO things:

1. CLASSIFY the section:
   - "rule" — contains specific, deterministic billing rules, code restrictions, unit thresholds, modifier requirements, documentation requirements, provider type restrictions, or prohibitions.
   - "narrative" — medical necessity criteria, admission/discharge criteria, clinical guidelines, care coordination — content requiring interpretation.
   - "skip" — boilerplate (table of contents, contact info, portal instructions, thank-you slides, schedules).

2. If "rule", EXTRACT each distinct rule as a JSON object:
   - rule_label: short descriptive name (e.g., "H0018 discharge day prohibition")
   - population: which claim population the rule governs. MUST be one of exactly:
     bh_residential | bh_outpatient | dme | nemt | nursing_home | professional | cross_population
     Choose based on the SERVICE the rule governs, not the document title. If it applies broadly, use cross_population.
   - proc_codes: pipe-separated procedure codes (e.g., "H0018|H0004") or null
   - diagnosis_codes: pipe-separated ICD-10 codes ONLY if the rule restricts to specific diagnoses (e.g., "F33.1|F20.9"). Diagnosis CATEGORIES stated without codes (e.g., "SMI determination") go in the condition text, not here. Otherwise null.
   - age_min: minimum member age in years if the rule is age-restricted (e.g., 18), else null
   - age_max: maximum member age in years if the rule is age-restricted (e.g., 20 for under-21 EPSDT rules), else null
   - modifiers: pipe-separated modifiers (e.g., "U9|TF") or null
   - provider_types: pipe-separated AHCCCS provider type codes (e.g., "B8") or null. Only if the text states them.
   - rule_type: one of [billing_prohibition, unit_threshold, documentation_requirement, code_restriction, provider_type_restriction, modifier_requirement, pos_requirement, frequency_limit, prior_auth_requirement, inclusive_rate]
   - condition: plain-English statement of when the rule applies
   - threshold_value: numeric threshold or null
   - threshold_unit: "units_per_dos", "units_per_year", "pos_code", "hours", "days", or null
   - effective_date: ONLY if an effective date is EXPLICITLY stated in this text, ISO format. Otherwise null. Do NOT use the document date.
   - related_codes: for inclusive_rate rules only — pipe-separated codes that are bundled INTO the inclusive rate and must not be billed separately. Otherwise null.
   - machine_checkable: true if a claims database (codes, units, dates, modifiers, POS, provider type) is sufficient to detect a violation. false if verification requires facts outside claims data (e.g., whether the member was present overnight, whether a provider belongs to the crisis system, medical records content).
   - citation: "Document Title, page/slide X"

RESPOND IN THIS EXACT FORMAT and nothing else:

CLASSIFICATION: rule|narrative|skip

RULES_JSON (only if classification is "rule"):
[
  {"rule_label": "...", "population": "...", ...}
]

NARRATIVE_SUMMARY (only if classification is "narrative"):
A 1-2 sentence summary of what this section covers."""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Robust JSON extraction + validation

# COMMAND ----------

def extract_json_array(text: str) -> Optional[str]:
    """FIX #9: balanced-bracket scan, string-aware. Replaces fragile non-greedy regex."""
    start = text.find('[')
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        start = text.find('[', start + 1)
    return None


HCPCS_CPT_RE = re.compile(r'^([A-Z]\d{4}|\d{5})$')      # HCPCS Level II or CPT
MODIFIER_RE  = re.compile(r'^[A-Z0-9]{2}$')
ICD10_RE     = re.compile(r'^[A-Z]\d{2}(\.\d{1,4})?$')   # ICD-10-CM, e.g. F33.1

def normalize_population(raw) -> str:
    if raw is None:
        return "needs_review"
    v = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if v in POPULATION_VOCAB:
        return v
    if v in POPULATION_SYNONYMS:
        return POPULATION_SYNONYMS[v]
    return "needs_review"   # FIX: never silently default to a target population

def clean_pipe_codes(raw, validator) -> Optional[str]:
    """Normalize pipe-separated code lists; drop tokens failing format validation."""
    if raw is None or str(raw).strip().lower() in ("", "none", "null", "n/a"):
        return None
    tokens = [t.strip().upper() for t in str(raw).split("|") if t.strip()]
    valid = [t for t in tokens if validator.match(t)]
    return "|".join(valid) if valid else None

def deterministic_rule_id(source_url: str, condition: str, proc_codes) -> str:
    """FIX #6: stable, globally unique, re-run-safe rule IDs."""
    key = f"{source_url}|{proc_codes}|{re.sub(r'\\s+', ' ', str(condition)).strip().lower()}"
    return "R-" + hashlib.md5(key.encode()).hexdigest()[:12]

def validate_rule(rule: Dict, section: Dict) -> (Optional[Dict], Optional[str]):
    """FIX #10: schema validation. Returns (clean_rule, None) or (None, reason)."""
    condition = str(rule.get("condition", "")).strip()
    if len(condition) < 15:
        return None, "condition_too_short"

    rule_type = str(rule.get("rule_type", "")).strip().lower()
    if rule_type not in VALID_RULE_TYPES:
        return None, f"invalid_rule_type:{rule_type}"

    proc_codes = clean_pipe_codes(rule.get("proc_codes"), HCPCS_CPT_RE)
    modifiers  = clean_pipe_codes(rule.get("modifiers"), MODIFIER_RE)
    related    = clean_pipe_codes(rule.get("related_codes"), HCPCS_CPT_RE)
    dx_codes   = clean_pipe_codes(rule.get("diagnosis_codes"), ICD10_RE)

    def _int_or_none(v):
        try:
            return int(float(v)) if v is not None and str(v).strip().lower() not in ("", "none", "null") else None
        except (ValueError, TypeError):
            return None
    age_min, age_max = _int_or_none(rule.get("age_min")), _int_or_none(rule.get("age_max"))

    thr = rule.get("threshold_value")
    try:
        threshold_value = float(thr) if thr is not None and str(thr).strip().lower() not in ("", "none", "null") else None
    except (ValueError, TypeError):
        return None, f"non_numeric_threshold:{thr}"

    eff = rule.get("effective_date")
    effective_date = None
    if eff and str(eff).strip().lower() not in ("none", "null", ""):
        try:
            effective_date = pd.to_datetime(str(eff)).date().isoformat()
        except Exception:
            effective_date = None   # bad date -> null, never a guess

    mc = rule.get("machine_checkable")
    machine_checkable = bool(mc) if isinstance(mc, bool) else str(mc).strip().lower() == "true"
    # Heuristic backstop: conditions referencing unverifiable facts are never machine-checkable
    UNCHECKABLE_HINTS = ("overnight", "medical record", "documentation must include",
                         "crisis system", "on behalf of", "independent provider",
                         "receives treatment elsewhere", "clinical")
    if any(h in condition.lower() for h in UNCHECKABLE_HINTS):
        machine_checkable = False

    default_cite = (f"{section['source_title']}, section '{section['section_title']}'"
                    if section.get("html_section")
                    else f"{section['source_title']}, page {section['start_page']}")
    clean = {
        "rule_id": deterministic_rule_id(section["source_url"], condition, proc_codes),
        "rule_label": str(rule.get("rule_label") or rule.get("rule_id") or "")[:120],
        "population": normalize_population(rule.get("population")),
        "population_raw": str(rule.get("population") or ""),
        "proc_codes": proc_codes, "modifiers": modifiers,
        "diagnosis_codes": dx_codes, "age_min": age_min, "age_max": age_max,
        "provider_types": clean_pipe_codes(rule.get("provider_types"), MODIFIER_RE),
        "rule_type": rule_type, "condition": condition,
        "threshold_value": threshold_value,
        "threshold_unit": (str(rule.get("threshold_unit")).strip().lower()
                           if rule.get("threshold_unit") not in (None, "", "null", "None") else None),
        "effective_date": effective_date,
        "related_codes": related,
        "machine_checkable": machine_checkable,
        "citation": str(rule.get("citation") or default_cite)[:300],
        "citation_page_exact": bool(section.get("page_exact", False)),
        "publication_date": section.get("publication_date"),
        "source_url": section["source_url"],
        "source_filename": section["filename"],
        "extraction_timestamp": now_iso(),
        "extraction_model": MODEL_ID,
        "origin": "llm",
    }
    return clean, None

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Classification & Extraction (with self-repair retry)

# COMMAND ----------

def classify_and_extract(section: Dict) -> Dict:
    loc = (f"SECTION: {section['section_title']}" if section.get("html_section")
           else f"PAGE(S): {section['start_page']}-{section['end_page']}")
    user_content = f"""SOURCE DOCUMENT: {section['source_title']}
SECTION TITLE: {section['section_title']}
{loc}
{f"(window {section['window_idx']+1}/{section['n_windows']} of a long section)" if section.get('n_windows', 1) > 1 else ""}

SECTION TEXT:
{section['body_text']}"""

    messages = [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    result = {"classification": "narrative", "rules": [], "invalid_rules": [],
              "narrative_summary": "", "error": None, "raw_output": ""}
    try:
        final_text, analysis_text = generate_with_harmony(messages, max_new_tokens=2000)
        if not final_text:
            final_text = analysis_text
        result["raw_output"] = final_text[:4000]
    except Exception as e:
        result["classification"] = "error"
        result["error"] = f"llm_exception: {e}"
        return result

    m = re.search(r'CLASSIFICATION:\s*(rule|narrative|skip)', final_text, re.IGNORECASE)
    if m:
        result["classification"] = m.group(1).lower()

    if result["classification"] == "rule":
        json_text = extract_json_array(final_text)
        parsed = None
        if json_text:
            try:
                parsed = json.loads(json_text)
            except json.JSONDecodeError:
                parsed = None
        if parsed is None:
            # FIX #9: one self-repair attempt before giving up
            try:
                repair_msgs = messages + [
                    {"role": "assistant", "content": final_text},
                    {"role": "user", "content": "Your RULES_JSON was not valid JSON. Respond with ONLY the corrected JSON array, nothing else."},
                ]
                repair_text, _ = generate_with_harmony(repair_msgs, max_new_tokens=1500)
                json_text = extract_json_array(repair_text)
                parsed = json.loads(json_text) if json_text else None
            except Exception:
                parsed = None
        if parsed is None:
            result["classification"] = "error"
            result["error"] = "json_parse_failure"
            return result
        if isinstance(parsed, dict):
            parsed = [parsed]
        for raw_rule in parsed:
            clean, reason = validate_rule(raw_rule, section)
            if clean:
                result["rules"].append(clean)
            else:
                result["invalid_rules"].append({"reason": reason, "raw": json.dumps(raw_rule)[:1000]})

    if result["classification"] == "narrative":
        sm = re.search(r'NARRATIVE_SUMMARY.*?\n(.+?)(?:\n\n|$)', final_text, re.DOTALL)
        if sm:
            result["narrative_summary"] = sm.group(1).strip()

    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manual Safety-Net Rules (authoritative — ALWAYS kept)

# COMMAND ----------

def manual_rule(label, population, proc, mods, ptypes, rtype, condition,
                thr, thr_unit, eff, related, checkable, citation, url, fname):
    return {
        "rule_id": deterministic_rule_id(url, condition, proc),
        "rule_label": label, "population": population, "population_raw": population,
        "proc_codes": proc, "modifiers": mods, "provider_types": ptypes,
        "rule_type": rtype, "condition": condition,
        "threshold_value": thr, "threshold_unit": thr_unit,
        "effective_date": eff, "related_codes": related,
        "diagnosis_codes": None, "age_min": None, "age_max": None,
        "machine_checkable": checkable, "citation": citation,
        "citation_page_exact": True, "publication_date": None,
        "source_url": url, "source_filename": fname,
        "extraction_timestamp": now_iso(), "extraction_model": "manual",
        "origin": "manual",
    }

BHRF_URL = "https://azahcccs.gov/Resources/Downloads/DFSMTraining/2024/Behavioral_HealthResidentialFacilityBHP_ResponsibilitiesSeptember2024.pdf"
BHRF_F   = "Behavioral_HealthResidentialFacilityBHP_ResponsibilitiesSeptember2024.pdf"
OUTBH_URL = "https://azahcccs.gov/Resources/Downloads/DFSMTraining/2023/7-14-2023Update-OutpatientBH-BillingCodes.pdf"
OUTBH_F   = "7-14-2023Update-OutpatientBH-BillingCodes.pdf"

MANUAL_RULES = [
    manual_rule("H0018 discharge day prohibition", "bh_residential", "H0018", None, "B8",
        "billing_prohibition",
        "H0018 may NOT be billed on the day of discharge. The per diem rate cannot be billed for the day of discharge as the member is not there for the full day.",
        None, None, None, None, True,
        "BHRF BHP Responsibilities Sept 2024, slide 49", BHRF_URL, BHRF_F),
    manual_rule("H0018 POS 56 requirement", "bh_residential", "H0018", None, "B8",
        "pos_requirement",
        "Effective for dates of service beginning December 1, 2023, BHRF claims must be submitted using Place of Service (POS) code 56 'Psychiatric Residential Treatment'.",
        56, "pos_code", "2023-12-01", None, True,
        "BHRF BHP Responsibilities Sept 2024, slide 50", BHRF_URL, BHRF_F),
    manual_rule("H0018+U9 frequency limit 2/yr", "bh_residential", "H0018", "U9", "B8",
        "frequency_limit",
        "H0018 may be paired with the U9 modifier at intake and during discharge planning. Limit 2 per year per member when using ASAM to determine appropriate level of care.",
        2, "units_per_year", None, None, True,
        "BHRF BHP Responsibilities Sept 2024, slide 51", BHRF_URL, BHRF_F),
    manual_rule("H0018 inclusive per diem (no unbundling)", "bh_residential", "H0018", None, "B8",
        "inclusive_rate",
        "H0018 is a comprehensive service code inclusive of ALL screening, assessment, counseling, case management, rehabilitation, and supportive services. Counseling (H0004) and other BH HCPCS codes cannot be billed separately alongside H0018.",
        None, None, None,
        "H0004|H0031|H0038|H2014|H2016|H2017|H2019|H2025|H2027|H0006|H0036|T1016|90832|90834|90837|90847|90853",
        True,
        "BHRF BHP Responsibilities Sept 2024, slide 49", BHRF_URL, BHRF_F),
    manual_rule("H0018 single line item per DOS", "bh_residential", "H0018", None, "B8",
        "billing_prohibition",
        "Effective March 1, 2024, all BHRF claims must be billed on the CMS 1500 claim form using H0018 on a single line item per date of service. Claims that list a range of service dates or months on a single line item will be denied.",
        None, None, "2024-03-01", None, True,
        "BHRF BHP Responsibilities Sept 2024, slide 50", BHRF_URL, BHRF_F),
    manual_rule("BHRF external services not claimable", "bh_residential", "H0018", None, "B8",
        "billing_prohibition",
        "BHRF cannot submit claims to AHCCCS or bill for services on behalf of an independent provider. External services (equine therapy, sweat lodge, yoga, etc.) are the BHRF's responsibility to pay and cannot be claimed to AHCCCS.",
        None, None, None, None, False,   # requires facts outside claims data
        "BHRF BHP Responsibilities Sept 2024, slide 52", BHRF_URL, BHRF_F),
    manual_rule("H0018 overnight presence required", "bh_residential", "H0018", None, "B8",
        "billing_prohibition",
        "H0018 may only be billed on days when the member is present overnight. If the member only comes to the BHRF to eat and sleep and receives treatment elsewhere (e.g., day hospital or IOP), the BHRF cannot submit the claim.",
        None, None, None, None, False,   # requires facts outside claims data
        "BHRF BHP Responsibilities Sept 2024, slides 49 & 61", BHRF_URL, BHRF_F),
    manual_rule("H0018 TF modifier for intermediate LOC", "bh_residential", "H0018", "TF", "B8",
        "modifier_requirement",
        "H0018 with TF modifier is required for intermediate level of care when BHRF provides personal care services. BHRF must be licensed by ADHS to provide personal care services.",
        None, None, None, None, False,   # LOC not observable in claims alone
        "BHRF BHP Responsibilities Sept 2024, slide 64", BHRF_URL, BHRF_F),
    manual_rule("BHRF prior authorization required", "bh_residential", "H0018", None, "B8",
        "prior_auth_requirement",
        "BHRF care requires prior and continued authorization. IHS/638 tribal facilities do NOT require prior authorization for members enrolled under Title XIX.",
        None, None, None, None, False,   # auth status not in claims extract
        "BHRF BHP Responsibilities Sept 2024, slides 4 & 6", BHRF_URL, BHRF_F),
    manual_rule("Hourly BH codes documentation threshold", "bh_outpatient", 
        "H0006|H0036|H2010|H2012|T1002|T1003", None, None,
        "documentation_requirement",
        "Documentation required when billing more than 2 hourly units OR 4 fifteen-minute units on a single date of service.",
        4, "fifteen_min_units_per_dos", "2023-07-17", None, True,
        "Outpatient BH Billing Codes July 2023, slides 9-10", OUTBH_URL, OUTBH_F),
    manual_rule("H2019/H2025 documentation threshold >8", "bh_outpatient", "H2019|H2025", None, None,
        "documentation_requirement",
        "Documentation required when billing more than 8 units on a single date of service.",
        8, "units_per_dos", "2023-07-17", None, True,
        "Outpatient BH Billing Codes July 2023, slide 11", OUTBH_URL, OUTBH_F),
    manual_rule("15-min BH codes documentation threshold >4", "bh_outpatient",
        "H0004|H0038|H2011|H2014|H2015|H2017|H2025|H2027|H5150|T1016|T1019|H0034", None, None,
        "documentation_requirement",
        "Documentation required when billing more than 4 units on a single date of service.",
        4, "units_per_dos", "2023-07-17", None, True,
        "Outpatient BH Billing Codes July 2023, slides 12-14", OUTBH_URL, OUTBH_F),
    manual_rule("H0030 crisis-system only", "bh_outpatient", "H0030", None, None,
        "billing_prohibition",
        "H0030 Behavioral Health Hotline Services can only be utilized by a provider that is part of the state crisis system. Claims cannot be submitted to DFSM.",
        None, None, "2023-07-17", None, True,   # any H0030 claim in FFS data is checkable
        "Outpatient BH Billing Codes July 2023, slide 18", OUTBH_URL, OUTBH_F),
    manual_rule("Per-claim documentation codes", "bh_outpatient",
        "T1503|S5131|T2020|T2026|S5130|S9484", None, None,
        "documentation_requirement",
        "Documentation required for claims submitted in ANY unit quantity (no threshold - all claims require documentation).",
        1, "units_per_dos", "2023-07-17", None, True,
        "Outpatient BH Billing Codes July 2023, slide 8", OUTBH_URL, OUTBH_F),
]

print(f"Manual safety-net rules: {len(MANUAL_RULES)} (all will be written — FIX #2)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dedup: manual rules authoritative; LLM near-duplicates dropped

# COMMAND ----------

def _norm_tokens(text: str) -> set:
    return set(re.findall(r'[a-z0-9]+', str(text).lower()))

def jaccard(a: str, b: str) -> float:
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def dedup_llm_against_manual(llm_rules: List[Dict], manual_rules: List[Dict],
                             threshold: float = 0.55) -> (List[Dict], int):
    """Drop LLM rules that duplicate a manual rule (same code overlap + similar condition)."""
    kept, dropped = [], 0
    for lr in llm_rules:
        lr_codes = set(str(lr.get("proc_codes") or "").split("|"))
        dup = False
        for mr in manual_rules:
            mr_codes = set(str(mr.get("proc_codes") or "").split("|"))
            if lr.get("rule_type") == mr.get("rule_type") and (lr_codes & mr_codes):
                if jaccard(lr.get("condition", ""), mr.get("condition", "")) >= threshold:
                    dup = True
                    break
        if dup:
            dropped += 1
        else:
            kept.append(lr)
    return kept, dropped

def dedup_within(rules: List[Dict]) -> List[Dict]:
    """Exact dedup on deterministic rule_id (windows/overlaps can re-extract the same rule)."""
    seen, out = set(), []
    for r in rules:
        if r["rule_id"] not in seen:
            seen.add(r["rule_id"])
            out.append(r)
    return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## Typed Delta writers (FIX #11) + per-document checkpointing (FIX #5)

# COMMAND ----------

from pyspark.sql.types import (StructType, StructField, StringType, DoubleType,
                               BooleanType, IntegerType)

RULES_SCHEMA = StructType([
    StructField("rule_id", StringType()), StructField("rule_label", StringType()),
    StructField("population", StringType()), StructField("population_raw", StringType()),
    StructField("proc_codes", StringType()), StructField("modifiers", StringType()),
    StructField("diagnosis_codes", StringType()),
    StructField("age_min", IntegerType()), StructField("age_max", IntegerType()),
    StructField("provider_types", StringType()), StructField("rule_type", StringType()),
    StructField("condition", StringType()), StructField("threshold_value", DoubleType()),
    StructField("threshold_unit", StringType()), StructField("effective_date", StringType()),
    StructField("related_codes", StringType()), StructField("machine_checkable", BooleanType()),
    StructField("citation", StringType()), StructField("citation_page_exact", BooleanType()),
    StructField("publication_date", StringType()), StructField("source_url", StringType()),
    StructField("source_filename", StringType()), StructField("extraction_timestamp", StringType()),
    StructField("extraction_model", StringType()), StructField("origin", StringType()),
    StructField("doc_hash", StringType()), StructField("superseded_at", StringType()),
])

CHUNKS_SCHEMA = StructType([
    StructField("chunk_id", StringType()), StructField("source_url", StringType()),
    StructField("source_title", StringType()), StructField("source_key", StringType()),
    StructField("doc_type", StringType()), StructField("section_title", StringType()),
    StructField("chunk_text", StringType()), StructField("char_count", IntegerType()),
    StructField("page_start", IntegerType()), StructField("page_end", IntegerType()),
    StructField("page_exact", BooleanType()), StructField("publication_date", StringType()),
    StructField("narrative_summary", StringType()), StructField("extraction_timestamp", StringType()),
    StructField("doc_hash", StringType()), StructField("superseded_at", StringType()),
])

ERRORS_SCHEMA = StructType([
    StructField("error_id", StringType()), StructField("source_url", StringType()),
    StructField("source_title", StringType()), StructField("section_title", StringType()),
    StructField("window_idx", IntegerType()), StructField("page_start", IntegerType()),
    StructField("page_end", IntegerType()), StructField("error_type", StringType()),
    StructField("detail", StringType()), StructField("section_text", StringType()),
    StructField("raw_llm_output", StringType()), StructField("timestamp", StringType()),
])

PROCESSED_SCHEMA = StructType([
    StructField("source_url", StringType()), StructField("source_title", StringType()),
    StructField("status", StringType()), StructField("n_sections", IntegerType()),
    StructField("n_rules", IntegerType()), StructField("n_chunks", IntegerType()),
    StructField("n_errors", IntegerType()), StructField("processed_at", StringType()),
    StructField("extraction_model", StringType()), StructField("doc_hash", StringType()),
])

def _supersede_doc_rows(table: str, url: str):
    """VERSIONING: mark prior rows for this document as superseded instead of deleting.
    History is preserved — 'was this rule in effect when the claim was billed?' stays
    answerable across AHCCCS's in-place PDF updates. Readers filter superseded_at IS NULL."""
    try:
        spark.sql(f"UPDATE {table} SET superseded_at = '{now_iso()}' "
                  f"WHERE source_url = '{url}' AND superseded_at IS NULL")
    except Exception:
        pass  # table may not exist yet, or may lack the column on first v4 run

def _delete_doc_rows(table: str, url: str):
    """Errors table only — failed-section records are operational, not history."""
    try:
        spark.sql(f"DELETE FROM {table} WHERE source_url = '{url}'")
    except Exception:
        pass

def append_typed(table: str, rows: List[Dict], schema: StructType):
    if not rows:
        return
    df = pd.DataFrame(rows)
    for f in schema.fields:
        if f.name not in df.columns:
            df[f.name] = None
    df = df[[f.name for f in schema.fields]]
    # Coerce types
    for f in schema.fields:
        if isinstance(f.dataType, DoubleType):
            df[f.name] = pd.to_numeric(df[f.name], errors="coerce")
        elif isinstance(f.dataType, IntegerType):
            df[f.name] = pd.to_numeric(df[f.name], errors="coerce").astype("Int64")
        elif isinstance(f.dataType, BooleanType):
            df[f.name] = df[f.name].astype("boolean")
        else:
            df[f.name] = df[f.name].where(df[f.name].notna(), None)
            df[f.name] = df[f.name].apply(lambda x: None if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
    sdf = spark.createDataFrame(df, schema=schema)
    sdf.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table)

# One-time migration: pre-v4 tables lack the versioning/dx columns. UPDATE ... superseded_at
# fails silently without them, which would break history-keeping on the first v4 run.
def migrate_pre_v4_tables():
    migrations = {
        RULES_TABLE: ["diagnosis_codes STRING", "age_min INT", "age_max INT",
                      "doc_hash STRING", "superseded_at STRING"],
        CHUNKS_TABLE: ["doc_hash STRING", "superseded_at STRING"],
        PROCESSED_TABLE: ["doc_hash STRING"],
    }
    for table, cols in migrations.items():
        try:
            existing = {f.name for f in spark.table(table).schema.fields}
        except Exception:
            continue   # table doesn't exist yet — created with full v4 schema on first append
        to_add = [c for c in cols if c.split()[0] not in existing]
        if to_add:
            spark.sql(f"ALTER TABLE {table} ADD COLUMNS ({', '.join(to_add)})")
            print(f"Migrated {table}: added {to_add}")

migrate_pre_v4_tables()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Extraction — per-document loop with checkpointing

# COMMAND ----------

# Write manual rules (versioned: prior copies superseded, never deleted)
for src_url in {BHRF_URL, OUTBH_URL}:
    _supersede_doc_rows(RULES_TABLE, src_url)
for mr in MANUAL_RULES:
    mr["doc_hash"] = None
    mr["superseded_at"] = None
append_typed(RULES_TABLE, MANUAL_RULES, RULES_SCHEMA)
print(f"Wrote {len(MANUAL_RULES)} manual rules (authoritative).")

run_stats = {"docs": 0, "skipped_unchanged": 0, "rules": 0, "chunks": 0,
             "errors": 0, "llm_dupes_dropped": 0}

for d_idx, drow in candidates.iterrows():
    url, title = drow["url"], drow["title"]
    print(f"\n{'='*70}\nDOC [{d_idx+1}/{len(candidates)}] {title[:60]}")

    doc = download_document(url)
    if not doc:
        append_typed(ERRORS_TABLE, [{
            "error_id": hashlib.md5(f"{url}|download".encode()).hexdigest()[:12],
            "source_url": url, "source_title": title, "section_title": None,
            "window_idx": None, "page_start": None, "page_end": None,
            "error_type": "download_failure", "detail": "download or parse failed",
            "section_text": None, "raw_llm_output": None, "timestamp": now_iso(),
        }], ERRORS_SCHEMA)
        append_typed(PROCESSED_TABLE, [{
            "source_url": url, "source_title": title, "status": "download_failed",
            "n_sections": 0, "n_rules": 0, "n_chunks": 0, "n_errors": 1,
            "processed_at": now_iso(), "extraction_model": MODEL_ID, "doc_hash": None,
        }], PROCESSED_SCHEMA)
        continue

    # Change detection: identical content -> skip (no re-extraction, no superseding)
    if url in done_urls and known_hashes.get(url) == doc["doc_hash"]:
        print("  UNCHANGED (hash match) — skipping.")
        run_stats["skipped_unchanged"] += 1
        continue
    if url in done_urls:
        print("  CONTENT CHANGED — re-extracting; prior rules will be superseded, not deleted.")

    doc.update({"title": title, "source_key": drow["source_key"], "doc_type": drow["doc_type"],
                "publication_date": drow["publication_date"], "filename": drow["filename"]})

    sections = [s for s in segment_document(doc) if len(s.get("body_text", "")) > 50]
    fmt = doc.get("source_format", "pdf")
    print(f"  [{fmt}] {doc['total_pages'] or '-'} pages -> {len(sections)} sections/windows")

    doc_rules, doc_chunks, doc_errors = [], [], []

    for i, section in enumerate(sections):
        result = classify_and_extract(section)
        cls = result["classification"]

        if cls == "rule" and result["rules"]:
            doc_rules.extend(result["rules"])
        elif cls == "narrative":
            chunk_id = hashlib.md5(
                f"{section['source_url']}_{section['start_page']}_{section['section_title']}_{section.get('window_idx',0)}".encode()
            ).hexdigest()[:12]
            doc_chunks.append({
                "chunk_id": chunk_id, "source_url": section["source_url"],
                "source_title": section["source_title"], "source_key": section["source_key"],
                "doc_type": section["doc_type"], "section_title": section["section_title"],
                "chunk_text": section["section_text"] if section.get("n_windows", 1) == 1 else section["body_text"],
                "char_count": len(section["body_text"]),
                "page_start": section["start_page"], "page_end": section["end_page"],
                "page_exact": bool(section.get("page_exact", False)),
                "publication_date": section.get("publication_date"),
                "narrative_summary": result.get("narrative_summary", ""),
                "extraction_timestamp": now_iso(),
                "doc_hash": doc["doc_hash"], "superseded_at": None,
            })
        elif cls == "error":
            doc_errors.append({
                "error_id": hashlib.md5(f"{url}|{section['section_title']}|{section.get('window_idx',0)}".encode()).hexdigest()[:12],
                "source_url": url, "source_title": title,
                "section_title": section["section_title"],
                "window_idx": section.get("window_idx", 0),
                "page_start": section["start_page"], "page_end": section["end_page"],
                "error_type": result.get("error", "unknown"),
                "detail": result.get("error", ""),
                "section_text": section["body_text"],           # full text -> re-extractable
                "raw_llm_output": result.get("raw_output", ""),
                "timestamp": now_iso(),
            })
        # invalid rules quarantined too
        for inv in result.get("invalid_rules", []):
            doc_errors.append({
                "error_id": hashlib.md5(f"{url}|{section['section_title']}|{inv['raw'][:60]}".encode()).hexdigest()[:12],
                "source_url": url, "source_title": title,
                "section_title": section["section_title"],
                "window_idx": section.get("window_idx", 0),
                "page_start": section["start_page"], "page_end": section["end_page"],
                "error_type": "rule_validation_failure",
                "detail": inv["reason"], "section_text": section["body_text"],
                "raw_llm_output": inv["raw"], "timestamp": now_iso(),
            })

        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
            print(f"    [{i+1}/{len(sections)}] rules={len(doc_rules)} chunks={len(doc_chunks)} errors={len(doc_errors)}")

    # Dedup within doc, then against manual safety net
    doc_rules = dedup_within(doc_rules)
    doc_rules, n_dupes = dedup_llm_against_manual(doc_rules, MANUAL_RULES)
    for r in doc_rules:
        r["doc_hash"] = doc["doc_hash"]
        r["superseded_at"] = None

    # CHECKPOINT: write this document's results now (FIX #5).
    # v4: rules/chunks are SUPERSEDED (history kept); errors are deleted (operational only).
    _supersede_doc_rows(RULES_TABLE, url)
    _supersede_doc_rows(CHUNKS_TABLE, url)
    _delete_doc_rows(ERRORS_TABLE, url)
    append_typed(RULES_TABLE, doc_rules, RULES_SCHEMA)
    append_typed(CHUNKS_TABLE, doc_chunks, CHUNKS_SCHEMA)
    append_typed(ERRORS_TABLE, doc_errors, ERRORS_SCHEMA)
    append_typed(PROCESSED_TABLE, [{
        "source_url": url, "source_title": title, "status": "complete",
        "n_sections": len(sections), "n_rules": len(doc_rules),
        "n_chunks": len(doc_chunks), "n_errors": len(doc_errors),
        "processed_at": now_iso(), "extraction_model": MODEL_ID,
        "doc_hash": doc["doc_hash"],
    }], PROCESSED_SCHEMA)

    run_stats["docs"] += 1
    run_stats["rules"] += len(doc_rules)
    run_stats["chunks"] += len(doc_chunks)
    run_stats["errors"] += len(doc_errors)
    run_stats["llm_dupes_dropped"] += n_dupes
    print(f"  CHECKPOINTED: {len(doc_rules)} rules, {len(doc_chunks)} chunks, "
          f"{len(doc_errors)} errors, {n_dupes} manual-dupes dropped")

print(f"\n{'='*70}\nRUN COMPLETE: {run_stats}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Summary Report

# COMMAND ----------

rules_all = spark.table(RULES_TABLE).toPandas()
rules_now = rules_all[rules_all["superseded_at"].isna()] if "superseded_at" in rules_all.columns else rules_all
errs_now  = spark.table(ERRORS_TABLE).toPandas() if run_stats["errors"] or True else pd.DataFrame()

print("=" * 60)
print("EXTRACTION SUMMARY")
print("=" * 60)
print(f"\nCurrent rules: {len(rules_now)} "
      f"(manual: {(rules_now.origin=='manual').sum()}, llm: {(rules_now.origin=='llm').sum()}) "
      f"| superseded (history): {len(rules_all) - len(rules_now)}")
print(f"\nBy population:\n{rules_now['population'].value_counts().to_string()}")
print(f"\nBy rule_type:\n{rules_now['rule_type'].value_counts().to_string()}")
print(f"\nMachine-checkable: {rules_now['machine_checkable'].sum()} / {len(rules_now)}")
print(f"Rules needing population review: {(rules_now['population']=='needs_review').sum()}")
if "diagnosis_codes" in rules_now.columns:
    print(f"Rules with ICD-10 restrictions: {rules_now['diagnosis_codes'].notna().sum()}")
    print(f"Rules with age restrictions: {(rules_now['age_min'].notna() | rules_now['age_max'].notna()).sum()}")

try:
    print(f"\nERROR TABLE: {len(errs_now)} rows")
    if len(errs_now):
        print(errs_now["error_type"].value_counts().to_string())
        print("\nRe-extraction candidates are fully preserved in "
              f"{ERRORS_TABLE} (section_text column). Re-run failed sections on GPU/120B.")
except Exception:
    pass

print("\nNext: run Step 3 validation notebook (policy_kb_validation.py) — it is the gate for Step 4.")

# COMMAND ----------

del model
del tokenizer
torch.cuda.empty_cache()
print("GPU memory released.")
