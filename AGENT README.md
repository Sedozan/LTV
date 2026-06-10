# README — Copilot Agent Implementation Guide

## 1. Purpose

Update the `specialty-ltv` (renters) pipeline to read expense ratios
from the new 2026 CJ Keim expense Excel file.

- **Problem:** Renters pipeline still points to the old expense file.
  The new 2026 Excel file has a different structure and sheet layout.
- **Expected outcome:** `expense_object.py` updated with correct params
  for the new file. Pipeline reads and processes 2026 expense ratios
  without errors.

-----

## 2. Project Scope

- **Target repo:** `specialty-ltv`, branch `b-2859088`
- **Reference repo:** `home-ltv` (read-only — Miriam’s repo, same architecture)
- **Language:** Python 3
- **Environment:** Domino ML platform, Scality/S3 storage
- **Out of scope:**
  - Do not touch retention model files
  - Do not modify commission logic (`create_commission_data.py`)
  - Do not touch `process_finance_helpers.py` — it is config-driven and needs no changes
  - Do not modify any other repo

-----

## 3. How This Pipeline Works

The expense pipeline in `specialty-ltv` is **config-driven**. All Excel
reading params (sheet name, row offsets, column ranges) come from a
single config dict returned by `set_expense_object()` in `expense_object.py`.

`process_finance_helpers.py` reads that config and does all the actual
Excel parsing — **it does not need to change**.

The only file to modify is:

```
specialty_ltv/jobs/expense/process_expense_data/expense_object.py
```

-----

## 4. Task — Step by Step

### Step 1: Inspect the new Excel file

The new 2026 expense Excel file is located at:

```
[INSERT PATH TO EXCEL FILE IN VS CODE WORKSPACE]
```

Open it and inspect the `LTV_SPL` tab. Find and report:

- Exact sheet name (tab label)
- Which row number contains the column headers (`Line`, `N/R`, `Channel`, `IS`)
- Which row number the data starts on
- How many rows of data there are
- Which columns contain the expense ratio data

Also scroll right on the tab bar and check if a `Worksheet` tab exists
(used for claims). Report what you find.

**Do not make any changes yet. Report your findings first.**

### Step 2: Propose updated expense_object.py values

Based on your findings, propose the updated values for these keys in
`set_expense_object()`:

```python
"expense_sheet_name_at_scale"  # new sheet name
"expenses_headers"             # new column range e.g. "G:J"
"expense_row_skip"             # new skiprows value
"expense_size"                 # new nrows value
"expense_row_cnt"              # number of actual data rows
```

If the claims tab (`Worksheet`) has changed, also propose updates to:

```python
"claims_expense_sheet_name_at_scale"
"claim_header_row_skip"
```

**Show me the proposed changes. Do not edit any file yet.**

### Step 3: Wait for approval

Do not modify any file until I explicitly approve the proposed changes.

### Step 4: Make the change

Only after approval — update `expense_object.py` with the new values.
Change nothing else.

-----

## 5. Repository Structure

### Target: `specialty-ltv` (branch `b-2859088`)

```
specialty_ltv/jobs/expense/process_expense_data/
├── expense_object.py           ← ONLY file to modify
├── process_finance_helpers.py  ← DO NOT TOUCH
├── create_expense_data.py      ← DO NOT TOUCH
└── create_commission_data.py   ← DO NOT TOUCH
```

### Reference: `home-ltv` (read-only)

```
home_ltv/jobs/expense/process_expense_data/
├── process_raw_expense.py      ← different architecture, reference only
└── process_finance_helpers.py  ← reference only
```

Note: `home-ltv` uses a different architecture (params hardcoded in
`process_raw_expense.py`). Do not copy it directly. Use it only to
understand the 2026 Excel structure.

-----

## 6. Hard Constraints

- ❌ Do not modify any file except `expense_object.py`
- ❌ Do not install new packages
- ❌ Do not refactor anything
- ❌ Do not touch commission or retention logic
- ❌ Do not make changes before I approve your proposed values

-----

## 7. Behavioral Expectations

- Follow existing code style — clean, no unnecessary comments
- Minimal change — only update the config values that changed
- Explain your reasoning before acting
- If unsure about any param value, ask rather than guess

-----

## 8. Implementation Checklist

Before finishing, verify:

- [ ] Only `expense_object.py` was modified
- [ ] Sheet name matches the actual tab in the 2026 Excel file
- [ ] Row skip and nrows values are correct for `LTV_SPL` tab
- [ ] Claims tab params updated if that tab also changed
- [ ] No other files were touched

-----

## 9. Additional Context

- The 2026 file is a **plan**; the old file was **actual** — do not
  conflate them in any comments or naming
- `use_current` should remain `False` unless explicitly told otherwise
- The `column_rename_dict` maps raw Excel column names to pipeline
  column names — only update if the 2026 file uses different column headers