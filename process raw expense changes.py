# ── CHANGE 1: read_raw_expenses (replace whole function) ──────────────────
def read_raw_expenses() -> pd.DataFrame:
    df_raw = nsh.read_excel_s3_to_pandas(
        p.expenses_raw_2026,
        sheet_name="LTV_SPL",
        usecols="C, AM:AT, AW:BD, BG:BN, BQ:BX, CA:CH, CK:CR, CU:DB, DY:EF, EI:EP, ES:EZ, FC:FJ",
        names=exp_cols,
        nrows=1,
        skiprows=13,
    )

    # GG15 = total claims expense (all channels)
    claims_raw = nsh.read_excel_s3_to_pandas(
        p.expenses_raw_2026,
        sheet_name="LTV_SPL",
        usecols="GG",
        names=["claims_total"],
        nrows=1,
        skiprows=14,
    )
    df_raw["claims_total"] = claims_raw["claims_total"].iloc[0]

    return df_raw


# ── CHANGE 2: add_claims_exp (replace whole function) ─────────────────────
def add_claims_exp(proc_exp_er, claims_total, earned_prem_total) -> pd.DataFrame:
    claims_ratio = claims_total / earned_prem_total
    proc_exp_er["claims_new_er"] = claims_ratio
    proc_exp_er["claims_renew_er"] = claims_ratio
    return proc_exp_er


# ── CHANGE 3: main (replace the body that builds the df) ──────────────────
@logger.catch(reraise=True)
def main() -> None:
    raw_exp_df = read_raw_expenses()
    create_agt_col = add_ea_ia(raw_exp_df)
    split_components = add_components(raw_exp_df, create_agt_col)
    piv_df = pivot_df(split_components)

    earned_prem_cols = [
        f"earned_prem_{ch}_{typ}" for ch in channels for typ in types
    ]
    earned_prem_total = raw_exp_df[earned_prem_cols].iloc[0].sum()
    claims_total = raw_exp_df["claims_total"].iloc[0]

    claims_exp = add_claims_exp(piv_df, claims_total, earned_prem_total)
    fill_nulls = fill_na(claims_exp)
    final_df = format_cols(fill_nulls)
    nsh.pandas_to_s3(final_df, p.processed_expense_all)
