import pandas as pd
import os

DATA_PATH = "data/"
OUTPUT_PATH = "data/processed/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_data():
    accounts = pd.read_csv(DATA_PATH + "ravenstack_accounts.csv")
    subs = pd.read_csv(DATA_PATH + "ravenstack_subscriptions.csv")
    usage = pd.read_csv(DATA_PATH + "ravenstack_feature_usage.csv")
    support = pd.read_csv(DATA_PATH + "ravenstack_support_tickets.csv")
    churn = pd.read_csv(DATA_PATH + "ravenstack_churn_events.csv")

    return accounts, subs, usage, support, churn


def clean_data(accounts, subs, usage, support, churn):

    # Parse dates safely
    accounts["signup_date"] = pd.to_datetime(accounts["signup_date"], errors="coerce")

    subs["start_date"] = pd.to_datetime(subs["start_date"], errors="coerce")
    subs["end_date"] = pd.to_datetime(subs["end_date"], errors="coerce")

    usage["usage_date"] = pd.to_datetime(usage["usage_date"], errors="coerce")

    support["submitted_at"] = pd.to_datetime(support["submitted_at"], errors="coerce")

    churn["churn_date"] = pd.to_datetime(churn["churn_date"], errors="coerce")

    return accounts, subs, usage, support, churn


if __name__ == "__main__":

    accounts, subs, usage, support, churn = load_data()
    accounts, subs, usage, support, churn = clean_data(accounts, subs, usage, support, churn)

    accounts.to_csv(OUTPUT_PATH + "accounts_clean.csv", index=False)
    subs.to_csv(OUTPUT_PATH + "subscriptions_clean.csv", index=False)
    usage.to_csv(OUTPUT_PATH + "usage_clean.csv", index=False)
    support.to_csv(OUTPUT_PATH + "support_clean.csv", index=False)
    churn.to_csv(OUTPUT_PATH + "churn_clean.csv", index=False)

    print("Data preparation complete.")
