"""Trains a supervised model and an unsupervised one, then compares them.

- XGBoost sees the labels, so it learns what attacks look like.
- Isolation Forest only sees normal traffic and flags anything unusual.
"""
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from evaluate import score


def main():
    df = pd.read_csv("data/traffic.csv")
    X = df.drop(columns="label")
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    results = []

    # supervised: attacks are rare, so tell XGBoost to take them seriously
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                        scale_pos_weight=ratio, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, "models/xgboost.joblib")
    results.append(score("xgboost", y_test, xgb.predict(X_test)))

    # the 0.5 cutoff isn't sacred. Raising it cuts false alarms, lowers recall
    proba = xgb.predict_proba(X_test)[:, 1]
    results.append(score("xgboost (cutoff 0.8)", y_test, (proba > 0.8).astype(int)))

    # unsupervised: train on normal traffic only
    normal_only = X_train[y_train == 0]
    iso = IsolationForest(n_estimators=200, contamination=0.04, random_state=42)
    iso.fit(normal_only)
    joblib.dump(iso, "models/isolation_forest.joblib")
    iso_pred = (iso.predict(X_test) == -1).astype(int)  # -1 means anomaly
    results.append(score("isolation_forest", y_test, iso_pred))

    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    table.to_csv("results/comparison.csv", index=False)

    importances = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nWhat XGBoost relies on most:")
    print(importances.round(3))


if __name__ == "__main__":
    main()
