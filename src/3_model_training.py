import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

import joblib

DATA_PATH = "outputs/feature_table.csv"
OUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def sanity_checks(df: pd.DataFrame):
    # 1) label distribution
    if "churn_label" not in df.columns:
        raise ValueError("Missing churn_label column in feature_table.csv")

    rate = df["churn_label"].mean()
    print(f"[Sanity] churn_label mean = {rate:.4f} (churn rate)")

    # 2) check if label is only 0/1
    uniq = sorted(df["churn_label"].dropna().unique().tolist())
    print(f"[Sanity] churn_label unique values = {uniq}")

    # 3) missingness quick scan
    missing_top = df.isna().mean().sort_values(ascending=False).head(8)
    print("[Sanity] top missing columns:")
    print(missing_top)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def evaluate_model(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }
    print(f"\n[{name}] AUC={metrics['roc_auc']:.4f} | Acc={metrics['accuracy']:.4f}")
    return metrics, proba, pred


def main():
    df = pd.read_csv(DATA_PATH)
    sanity_checks(df)

    # Target
    y = df["churn_label"].astype(int)

    # Features: drop IDs + raw dates + target
    drop_cols = [
        "account_id",
        "account_name",
        "signup_date",
        "ref_date",
        "ref_date_fallback",
        "churn_flag",
        "churn_label",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    # Model 1: Logistic Regression
    logreg = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ))
    ])

    # Model 2: Random Forest
    rf = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])

    # Train + eval
    results = {}

    logreg.fit(X_train, y_train)
    results["logistic_regression"], p_lr, pred_lr = evaluate_model("LogReg", logreg, X_test, y_test)

    rf.fit(X_train, y_train)
    results["random_forest"], p_rf, pred_rf = evaluate_model("RandomForest", rf, X_test, y_test)

    # Save metrics
    with open(os.path.join(OUT_DIR, "model_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions table (for report / debugging)
    pred_out = pd.DataFrame({
        "y_true": y_test.values,
        "p_churn_logreg": p_lr,
        "y_pred_logreg_0_5": pred_lr,
        "p_churn_rf": p_rf,
        "y_pred_rf_0_5": pred_rf,
    })
    pred_out.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    # Save models
    joblib.dump(logreg, os.path.join(MODEL_DIR, "logreg.joblib"))
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf.joblib"))

    # Optional: RF feature importances (works only if we rebuild feature names from onehot)
    # We'll approximate by saving importances with transformed feature names.
    try:
        prep = rf.named_steps["prep"]
        clf = rf.named_steps["clf"]

        # get feature names
        num_cols = prep.transformers_[0][2]
        cat_pipe = prep.transformers_[1][1]
        cat_cols = prep.transformers_[1][2]
        ohe = cat_pipe.named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(cat_cols)

        feature_names = np.concatenate([np.array(num_cols, dtype=str), np.array(cat_names, dtype=str)])
        importances = clf.feature_importances_

        fi = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi = fi.sort_values("importance", ascending=False)
        fi.to_csv(os.path.join(OUT_DIR, "feature_importance_rf.csv"), index=False)
        print("\nSaved RF feature importances to outputs/feature_importance_rf.csv")
    except Exception as e:
        print("\n[Note] Could not export RF feature importances:", str(e))

    print("\nSaved outputs:")
    print("- outputs/model_metrics.json")
    print("- outputs/test_predictions.csv")
    print("- models/logreg.joblib")
    print("- models/rf.joblib")


if __name__ == "__main__":
    main()