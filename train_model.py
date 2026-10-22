"""
Train a churn model and save it as customer_churn_model.pkl.

Use your own dataset:  python train_model.py --csv data.csv
(CSV needs columns: Age, Tenure, MonthlyCharges, SupportCalls, Churn [0/1])

Without --csv, a synthetic dataset is generated so the project runs end-to-end.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["Age", "Tenure", "MonthlyCharges", "SupportCalls"]
OUT = "customer_churn_model.pkl"


def synthetic(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Age": rng.integers(18, 80, n),
            "Tenure": rng.integers(0, 72, n),
            "MonthlyCharges": rng.uniform(20, 150, n).round(2),
            "SupportCalls": rng.poisson(2, n),
        }
    )
    logit = (
        -1.0
        + 0.45 * df["SupportCalls"]
        - 0.04 * df["Tenure"]
        + 0.015 * (df["MonthlyCharges"] - 70)
        - 0.01 * (df["Age"] - 40)
    )
    p = 1 / (1 + np.exp(-logit))
    df["Churn"] = (rng.random(n) < p).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Path to training CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.csv) if args.csv else synthetic()
    X, y = df[FEATURES], df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced",
                random_state=42, n_jobs=-1)),
        ]
    )
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    print(classification_report(y_te, pred))
    print("ROC-AUC:", round(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]), 4))

    joblib.dump(model, OUT)
    print(f"Saved model -> {OUT}")


if __name__ == "__main__":
    main()
