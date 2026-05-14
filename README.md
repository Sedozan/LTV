# How to Run Retention Training / Scoring

This file provides instructions on how to retrain the Classic SPL retention models and how to run scoring. Key references:

1. [dodo_retention.py](https://github.allstate.com/D3-Lifetime-Value/classic-specialty-ltv/blob/main/classic_spl_ltv/dodo/dodo_retention.py) — doit task definitions for training and scoring
2. [Confluence: Retention Pipeline](https://confluence.allstate.com/display/EL/V7+Retention+-+Data) — preprocessing and featurization details
3. For questions about Classic SPL retention, reach out to <!-- TODO: update contacts -->

---

## Environment Setup

Before running any training or scoring jobs, set the following environment variables in your terminal **before** starting a kernel or submitting a Domino job:

```bash
export DYNACONF_GIT_BRANCH=<your-branch>
export DYNACONF_GIT_CHECKOUT=<your-branch>   # v9.1+ uses this for base/base_data paths
export ENV_FOR_DYNACONF=prod                  # uses system service account for Scality auth
```

**Important notes:**

- `DYNACONF_GIT_BRANCH` controls where outputs land in Scality S3. `DYNACONF_GIT_CHECKOUT` controls `base`/`base_data` path resolution. Both must be set.
- Domino's **Git Reference** controls which code runs — it does not communicate with the Dynaconf variables.
- Vault token expires every 24 hours and requires daily renewal.
- Set env vars in the terminal before starting the kernel. Using `%env` in notebook cells initializes too late for Dynaconf.

### Package Installation

```bash
pip install /mnt/imported/code/ltv-helpers/
pip install -e .   # installs classic_spl_ltv into base conda (/opt/conda/bin/python, Python 3.9.18)
```

Select the **base conda** environment as your kernel.

---

## Retention Training Process

Create a new branch off the current main branch before retraining to avoid overwriting models or encoders used in weekly scoring:

```bash
git checkout -b <new-branch> <existing-branch>
```

### Training Commands

All training runs go through `doit` tasks defined in `classic_spl_ltv/dodo/dodo_retention.py`. The invocation pattern:

```bash
./doit.sh prod <END_DATE> classic_spl_ltv/dodo/dodo_retention.py <task_name>
```

Where `<END_DATE>` is in `yyyymmdd` format (e.g., `20260327`, **not** `yyyy-mm-dd`).

The training tasks are:

1. `./doit.sh prod <END_DATE> classic_spl_ltv/dodo/dodo_retention.py train_auto_short_term`
2. `./doit.sh prod <END_DATE> classic_spl_ltv/dodo/dodo_retention.py train_auto_long_term`
3. `./doit.sh prod <END_DATE> classic_spl_ltv/dodo/dodo_retention.py train_hoc_short_term`
4. `./doit.sh prod <END_DATE> classic_spl_ltv/dodo/dodo_retention.py train_hoc_long_term`
5. `./doit.sh prod <END_DATE> classic_spl_ltv/dodo/dodo_retention.py train_other_short_term`
6. `./doit.sh prod <END_DATE> classic_spl_ltv/dodo/dodo_retention.py train_other_long_term`

Each `train_*` task runs the following pipeline steps in order:

- **Featurize** — `classic_spl_ltv/jobs/retention/spl_pipeline_featurize.py` builds features (autoregressive, RUFF, moving average, etc.). Note: this is a standalone Click script and cannot be invoked directly via `dodo.py`.
- **Train** — `classic_spl_ltv/jobs/retention/common_train.py` <!-- TODO: verify path --> splits data into train/test/validation and trains the LightGBM model.
- **Score** — `classic_spl_ltv/jobs/retention/common_score.py` reads the split data, scores each policy using the trained model, and writes output to the scored data path.
- **Evaluate** — <!-- TODO: verify path --> automates model review via evaluation notebook.

The `model_line` parameter (`auto`, `hoc`, `other`) and corresponding `where_clause` values are configured in `classic_spl_ltv/config/retention.toml`.

**Important:** Training jobs must be submitted as **Domino Jobs**, not run from the VSCode terminal (`domino-submit` is not available in workspace terminals).

---

## Retention Scoring Process

Scoring is typically handled by the weekly batch job. These commands are useful for ad-hoc scoring or testing changes to the scoring pipeline independent of other LTV tasks.

```bash
./doit.sh prod <END_DATE> classic_spl_ltv/dodo/dodo_retention.py score_auto_short_term
./doit.sh prod <END_DATE> classic_spl_ltv/dodo/dodo_retention.py score_auto_long_term
./doit.sh prod <END_DATE> classic_spl_ltv/dodo/dodo_retention.py score_hoc_short_term
./doit.sh prod <END_DATE> classic_spl_ltv/dodo/dodo_retention.py score_hoc_long_term
./doit.sh prod <END_DATE> classic_spl_ltv/dodo/dodo_retention.py score_other_short_term
./doit.sh prod <END_DATE> classic_spl_ltv/dodo/dodo_retention.py score_other_long_term
```

The scoring pipeline runs:

1. **Clean** — empties target scored output directories to prevent duplicate scores per policy.
2. **Score** — `classic_spl_ltv/jobs/retention/common_score.py` iterates over each model line / model type combination and writes results to the target output path.

### Post-Scoring Pipeline

After scoring, the downstream pipeline runs in this order:

```
dodo_finance.py → score_unbalanced → dodo_balance.py → score_balanced → check_validation → score_deliver
```

---

## Troubleshooting

### Environment

- **`ENV_FOR_DYNACONF=prod`** — use this to authenticate via system service account (`SYS_ID`/`SYS_ID_PASSWORD`) rather than personal AD credentials. Avoids Vault auth issues.
- **Vault token expiry** — tokens expire every 24 hours. Renew before running jobs.
- **`pip install -e .`** installs into base conda, not system Python. Make sure you select the correct kernel.

### Common Errors

- **`domino-submit: not found`** — you're trying to run a doit job from the VSCode terminal. Submit as a Domino Job instead.
- **`AttributeError: module 'domino.domino' has no attribute 'RunFailedException'`** — package version mismatch. Doesn't always mean the task actually failed. Check S3 for output artifacts and sub-job logs directly.
- **`Error 201: Invalid Upstream Credentials`** — owner-level permission issue on the Domino project. Escalate to the project owner.
- **Sub-jobs stuck in indefinite polling** — check sub-job status in Domino. If Queued/Pending → likely a Spark resource scheduling issue (reduce executor count/memory). If Failed → different root cause.
- **`export VAR=value doit ...`** — doesn't work. Use `&&` between export and doit commands.

### Data

- **`spark.read.parquet()` directly** — wrong approach in this codebase. Use `ph.read_parquet_s3()` from `ltv_helpers.pipeline_helpers`.
- **`len()` on Spark DataFrames** — doesn't work. Use `.count()`.
- **`eltv_pipeline` in `dae_prefix`** and Scality paths containing `eltv-policy/` are intentional naming — do not rename.
- **Git merge style** — `git pull origin main --no-rebase` is the team's preferred approach.
