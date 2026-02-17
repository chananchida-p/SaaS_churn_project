import os
import numpy as np
import pandas as pd

RAW_DIR = "data"
PROCESSED_DIR_1 = "data/processed"
PROCESSED_DIR_2 = "outputs/processed"   # in case you saved there
OUT_DIR = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)


def _pick_dir():
    # Prefer cleaned/processed data if it exists
    if os.path.isdir(PROCESSED_DIR_1) and len(os.listdir(PROCESSED_DIR_1)) > 0:
        return PROCESSED_DIR_1
    if os.path.isdir(PROCESSED_DIR_2) and len(os.listdir(PROCESSED_DIR_2)) > 0:
        return PROCESSED_DIR_2
    return RAW_DIR


def _read_csv(folder, filename):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _to_dt(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _safe_bool_to_int(s):
    # handles True/False, "true"/"false", 1/0, NaN
    return s.astype("boolean").fillna(False).astype(int)


def load_tables():
    folder = _pick_dir()
    print(f"Loading data from: {folder}")

    accounts = _read_csv(folder, "accounts_clean.csv") if "processed" in folder else _read_csv(folder, "ravenstack_accounts.csv")
    subs = _read_csv(folder, "subscriptions_clean.csv") if "processed" in folder else _read_csv(folder, "ravenstack_subscriptions.csv")
    usage = _read_csv(folder, "usage_clean.csv") if "processed" in folder else _read_csv(folder, "ravenstack_feature_usage.csv")
    support = _read_csv(folder, "support_clean.csv") if "processed" in folder else _read_csv(folder, "ravenstack_support_tickets.csv")
    churn = _read_csv(folder, "churn_clean.csv") if "processed" in folder else _read_csv(folder, "ravenstack_churn_events.csv")

    # Ensure datetime columns are parsed
    accounts = _to_dt(accounts, "signup_date")
    subs = _to_dt(subs, "start_date")
    subs = _to_dt(subs, "end_date")
    usage = _to_dt(usage, "usage_date")
    support = _to_dt(support, "submitted_at")
    support = _to_dt(support, "closed_at")
    churn = _to_dt(churn, "churn_date")

    return accounts, subs, usage, support, churn


def make_reference_dates(accounts, subs, usage, support, churn):
    """
    Create a per-account ref_date:
    - if churned: churn_date
    - else: last observed activity date (usage/support/subscription)
    """
    # churn date
    churn_ref = (
        churn.dropna(subset=["account_id", "churn_date"])
        .sort_values("churn_date")
        .groupby("account_id", as_index=False)["churn_date"]
        .last()
        .rename(columns={"churn_date": "ref_date"})
    )

    # usage date max requires mapping subscription_id -> account_id
    subs_map = subs[["subscription_id", "account_id"]].dropna()
    usage_acc = usage.merge(subs_map, on="subscription_id", how="left")
    usage_max = usage_acc.groupby("account_id", as_index=False)["usage_date"].max().rename(columns={"usage_date": "usage_max"})

    support_max = support.groupby("account_id", as_index=False)["submitted_at"].max().rename(columns={"submitted_at": "support_max"})

    subs_end_max = subs.groupby("account_id", as_index=False)["end_date"].max().rename(columns={"end_date": "subs_end_max"})
    subs_start_max = subs.groupby("account_id", as_index=False)["start_date"].max().rename(columns={"start_date": "subs_start_max"})

    fallback = accounts[["account_id", "signup_date"]].copy()
    fallback = fallback.merge(usage_max, on="account_id", how="left")
    fallback = fallback.merge(support_max, on="account_id", how="left")
    fallback = fallback.merge(subs_end_max, on="account_id", how="left")
    fallback = fallback.merge(subs_start_max, on="account_id", how="left")

    fallback["ref_date_fallback"] = fallback[["usage_max", "support_max", "subs_end_max", "subs_start_max", "signup_date"]].max(axis=1)

    ref = accounts[["account_id"]].drop_duplicates().merge(churn_ref, on="account_id", how="left")
    ref = ref.merge(fallback[["account_id", "ref_date_fallback"]], on="account_id", how="left")
    ref["ref_date"] = ref["ref_date"].fillna(ref["ref_date_fallback"])

    # final safety
    ref["ref_date"] = ref["ref_date"].fillna(pd.Timestamp.today().normalize())
    return ref[["account_id", "ref_date"]]


def build_feature_table(accounts, subs, usage, support, churn):
    ref = make_reference_dates(accounts, subs, usage, support, churn)

    # ---- label ----
    churned_accounts = churn[["account_id"]].dropna().drop_duplicates().assign(churn_label=1)

    base = accounts.copy()
    if "churn_flag" in base.columns:
        base["churn_flag"] = _safe_bool_to_int(base["churn_flag"])
    else:
        base["churn_flag"] = 0

    base = base.merge(churned_accounts, on="account_id", how="left")
    base["churn_label"] = base["churn_label"].fillna(base["churn_flag"]).astype(int)

    base = base.merge(ref, on="account_id", how="left")
    base["tenure_days"] = (base["ref_date"] - base["signup_date"]).dt.days
    base["tenure_days"] = base["tenure_days"].clip(lower=0)

    # =====================
    # Subscription features
    # =====================
    subs2 = subs.copy()
    for c in ["upgrade_flag", "downgrade_flag", "churn_flag", "auto_renew_flag", "is_trial"]:
        if c in subs2.columns:
            subs2[c] = _safe_bool_to_int(subs2[c])

    subs2 = subs2.merge(ref, on="account_id", how="left")
    subs2 = subs2.dropna(subset=["start_date"])
    subs_hist = subs2[subs2["start_date"] <= subs2["ref_date"]].copy()

    # numeric aggregates (only if columns exist)
    def col_or_nan(name):
        return name if name in subs_hist.columns else None

    agg_dict = {
        "subs_count": ("subscription_id", "nunique"),
    }
    if col_or_nan("mrr_amount"):
        agg_dict["mrr_mean"] = ("mrr_amount", "mean")
        agg_dict["mrr_max"] = ("mrr_amount", "max")
    if col_or_nan("arr_amount"):
        agg_dict["arr_mean"] = ("arr_amount", "mean")
    if col_or_nan("seats"):
        agg_dict["seats_mean"] = ("seats", "mean")
    if col_or_nan("upgrade_flag"):
        agg_dict["upgrades"] = ("upgrade_flag", "sum")
    if col_or_nan("downgrade_flag"):
        agg_dict["downgrades"] = ("downgrade_flag", "sum")
    if col_or_nan("auto_renew_flag"):
        agg_dict["auto_renew_rate"] = ("auto_renew_flag", "mean")
    if col_or_nan("is_trial"):
        agg_dict["trial_rate"] = ("is_trial", "mean")

    subs_agg = subs_hist.groupby("account_id").agg(**agg_dict).reset_index()

    # latest plan/billing
    subs_last = None
    keep_cols = ["account_id"]
    if "plan_tier" in subs_hist.columns:
        keep_cols.append("plan_tier")
    if "billing_frequency" in subs_hist.columns:
        keep_cols.append("billing_frequency")

    if len(keep_cols) > 1:
        subs_last = (
            subs_hist.sort_values(["account_id", "start_date"])
            .groupby("account_id")
            .tail(1)[keep_cols]
            .rename(columns={"plan_tier": "latest_plan_tier", "billing_frequency": "latest_billing_frequency"})
        )

    # =====================
    # Usage features (30/60/90d)
    # =====================
    subs_map = subs[["subscription_id", "account_id"]].dropna()
    usage2 = usage.merge(subs_map, on="subscription_id", how="left")
    usage2 = usage2.dropna(subset=["account_id", "usage_date"]).merge(ref, on="account_id", how="left")
    usage2 = usage2[usage2["usage_date"] <= usage2["ref_date"]].copy()

    # Some columns may not exist depending on version — guard them
    has_usage_id = "usage_id" in usage2.columns
    has_count = "usage_count" in usage2.columns
    has_dur = "usage_duration_secs" in usage2.columns
    has_err = "error_count" in usage2.columns
    has_feat = "feature_name" in usage2.columns
    has_beta = "is_beta_feature" in usage2.columns

    def usage_window(df, days):
        dfw = df[df["usage_date"] >= (df["ref_date"] - pd.to_timedelta(days, unit="D"))].copy()

        agg = {}
        agg[f"usage_events_{days}d"] = ("usage_id" if has_usage_id else "usage_date", "count")
        if has_count:
            agg[f"usage_count_sum_{days}d"] = ("usage_count", "sum")
        if has_dur:
            agg[f"usage_duration_sum_{days}d"] = ("usage_duration_secs", "sum")
        if has_err:
            agg[f"errors_sum_{days}d"] = ("error_count", "sum")
        if has_feat:
            agg[f"distinct_features_{days}d"] = ("feature_name", "nunique")
        if has_beta:
            agg[f"beta_share_{days}d"] = ("is_beta_feature", lambda x: np.mean(_safe_bool_to_int(x)) if len(x) else np.nan)

        return dfw.groupby("account_id").agg(**agg).reset_index()

    usage_30 = usage_window(usage2, 30)
    usage_60 = usage_window(usage2, 60)
    usage_90 = usage_window(usage2, 90)

    # =====================
    # Support features
    # =====================
    support2 = support.dropna(subset=["account_id", "submitted_at"]).merge(ref, on="account_id", how="left")
    support2 = support2[support2["submitted_at"] <= support2["ref_date"]].copy()

    if "escalation_flag" in support2.columns:
        support2["escalation_flag"] = _safe_bool_to_int(support2["escalation_flag"])

    sup_agg_dict = {
        "tickets": ("ticket_id", "nunique") if "ticket_id" in support2.columns else ("submitted_at", "count"),
    }
    if "priority" in support2.columns:
        sup_agg_dict["urgent_rate"] = ("priority", lambda x: np.mean((x == "urgent").astype(float)) if len(x) else np.nan)
        sup_agg_dict["high_rate"] = ("priority", lambda x: np.mean((x == "high").astype(float)) if len(x) else np.nan)
    if "resolution_time_hours" in support2.columns:
        sup_agg_dict["res_time_mean"] = ("resolution_time_hours", "mean")
        sup_agg_dict["res_time_max"] = ("resolution_time_hours", "max")
    if "first_response_time_minutes" in support2.columns:
        sup_agg_dict["first_resp_mean"] = ("first_response_time_minutes", "mean")
    if "satisfaction_score" in support2.columns:
        sup_agg_dict["satisfaction_mean"] = ("satisfaction_score", "mean")
        sup_agg_dict["satisfaction_missing"] = ("satisfaction_score", lambda x: np.mean(pd.isna(x).astype(float)) if len(x) else np.nan)
    if "escalation_flag" in support2.columns:
        sup_agg_dict["escalation_rate"] = ("escalation_flag", "mean")

    support_agg = support2.groupby("account_id").agg(**sup_agg_dict).reset_index()

    # =====================
    # Combine
    # =====================
    feat = base.merge(subs_agg, on="account_id", how="left")
    if subs_last is not None:
        feat = feat.merge(subs_last, on="account_id", how="left")

    feat = feat.merge(usage_30, on="account_id", how="left")
    feat = feat.merge(usage_60, on="account_id", how="left")
    feat = feat.merge(usage_90, on="account_id", how="left")
    feat = feat.merge(support_agg, on="account_id", how="left")

    return feat


if __name__ == "__main__":
    accounts, subs, usage, support, churn = load_tables()
    feat = build_feature_table(accounts, subs, usage, support, churn)

    out_path = os.path.join(OUT_DIR, "feature_table.csv")
    feat.to_csv(out_path, index=False)

    # small preview for quick check
    feat.head(50).to_csv(os.path.join(OUT_DIR, "feature_table_preview.csv"), index=False)

    print("Feature engineering complete.")
    print("Saved:", out_path)
    print("Rows (accounts):", feat.shape[0], " | Columns:", feat.shape[1])
    print("Churn rate:", round(feat["churn_label"].mean(), 4))
