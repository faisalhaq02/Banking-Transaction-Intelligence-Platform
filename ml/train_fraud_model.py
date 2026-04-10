import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42

DATA_PATH = "/opt/project/data/gold/customer_features"
MODEL_DIR = "/opt/project/models"
METRICS_DIR = "/opt/project/metrics"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def load_data():
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def create_target(df: pd.DataFrame) -> pd.Series:
    """
    Synthetic fraud target.
    If fraud_flag already exists, use it.
    Otherwise create a rule-based target for demo.
    """
    if "fraud_flag" in df.columns:
        y = df["fraud_flag"].astype(int)
    else:
        y = (
            (
                (df["txn_count"] > df["txn_count"].quantile(0.95)) &
                (df["total_spend"] > df["total_spend"].quantile(0.95))
            ) |
            (
                (df["max_amount"] > df["max_amount"].quantile(0.99)) &
                (df["avg_amount"] > df["avg_amount"].quantile(0.95))
            )
        ).astype(int)

    print("Target distribution:")
    print(y.value_counts(dropna=False))
    return y


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["txn_count", "total_spend", "avg_amount", "max_amount"]].copy()

    X["spend_per_txn_ratio"] = df["total_spend"] / (df["txn_count"] + 1)
    X["max_to_avg_ratio"] = df["max_amount"] / (df["avg_amount"] + 1)
    X["amount_range"] = df["max_amount"] - df["avg_amount"]
    X["txn_intensity"] = df["txn_count"] * df["avg_amount"]

    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    print(f"Prepared feature shape: {X.shape}")
    print(f"Feature columns: {list(X.columns)}")
    return X


def evaluate_classifier(name: str, y_true, y_pred, y_prob=None):
    print(f"\n===== {name} Results =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            metrics["roc_auc"] = float(roc_auc)
            metrics["pr_auc"] = float(pr_auc)
            print("ROC-AUC:", roc_auc)
            print("PR-AUC:", pr_auc)
        except Exception as e:
            print(f"Could not compute ROC-AUC / PR-AUC for {name}: {e}")

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(cm)
    print(classification_report(y_true, y_pred, labels=[0, 1], zero_division=0))

    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def main():
    df = load_data()
    y = create_target(df)
    X = prepare_features(df)

    unique_classes = sorted(y.unique().tolist())
    print("Unique target classes:", unique_classes)

    if len(unique_classes) < 2:
        raise ValueError(
            f"Target has only one class: {unique_classes}. "
            "Adjust fraud label generation so both 0 and 1 exist."
        )

    print("Fraud ratio:", float(y.mean()))

    inspection_df = X.copy()
    inspection_df["fraud_flag"] = y
    print("\nFeature correlations with fraud_flag:")
    print(inspection_df.corr(numeric_only=True)["fraud_flag"].sort_values(ascending=False))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("\nTrain fraud ratio:", float(y_train.mean()))
    print("Test fraud ratio:", float(y_test.mean()))

    all_metrics = {}

    # ----------------------------
    # Logistic Regression
    # ----------------------------
    lr = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_probs = lr.predict_proba(X_test)[:, 1]

    lr_metrics = evaluate_classifier(
        "Logistic Regression",
        y_test,
        lr_preds,
        lr_probs
    )
    all_metrics["logistic_regression"] = lr_metrics

    # ----------------------------
    # Random Forest
    # ----------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    rf_metrics = evaluate_classifier(
        "Random Forest",
        y_test,
        rf_preds,
        rf_probs
    )

    feature_importance = dict(zip(X.columns, rf.feature_importances_))
    print("\nRandom Forest Feature Importances:")
    for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.4f}")

    rf_metrics["feature_importance"] = feature_importance
    all_metrics["random_forest"] = rf_metrics

    rf_model_path = os.path.join(MODEL_DIR, "random_forest_fraud.pkl")
    joblib.dump(rf, rf_model_path)
    print(f"Saved Random Forest model to {rf_model_path}")

    # ----------------------------
    # Gradient Boosting
    # ----------------------------
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    gb_probs = gb.predict_proba(X_test)[:, 1]

    gb_metrics = evaluate_classifier(
        "Gradient Boosting",
        y_test,
        gb_preds,
        gb_probs
    )
    all_metrics["gradient_boosting"] = gb_metrics

    gb_model_path = os.path.join(MODEL_DIR, "gradient_boosting_fraud.pkl")
    joblib.dump(gb, gb_model_path)
    print(f"Saved Gradient Boosting model to {gb_model_path}")

    # ----------------------------
    # XGBoost
    # ----------------------------
    if XGBOOST_AVAILABLE:
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE
        )
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        xgb_probs = xgb.predict_proba(X_test)[:, 1]

        xgb_metrics = evaluate_classifier(
            "XGBoost",
            y_test,
            xgb_preds,
            xgb_probs
        )
        all_metrics["xgboost"] = xgb_metrics

        xgb_model_path = os.path.join(MODEL_DIR, "xgboost_fraud.pkl")
        joblib.dump(xgb, xgb_model_path)
        print(f"Saved XGBoost model to {xgb_model_path}")
    else:
        print("\nXGBoost not installed. Skipping XGBoost model.")

    # ----------------------------
    # Isolation Forest
    # ----------------------------
    contamination = min(0.2, max(0.01, float(y.mean())))
    print(f"\nUsing Isolation Forest contamination: {contamination}")

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE
    )
    iso.fit(X)

    iso_raw_preds = iso.predict(X)
    iso_preds = np.where(iso_raw_preds == -1, 1, 0)

    # Use anomaly score converted so higher means more fraud-like
    iso_scores = -iso.score_samples(X)

    iso_metrics = evaluate_classifier(
        "Isolation Forest",
        y,
        iso_preds,
        iso_scores
    )
    all_metrics["isolation_forest"] = iso_metrics

    iso_model_path = os.path.join(MODEL_DIR, "isolation_forest_fraud.pkl")
    joblib.dump(iso, iso_model_path)
    print(f"Saved Isolation Forest model to {iso_model_path}")

    # ----------------------------
    # Save metrics
    # ----------------------------
    metrics = {
        "fraud_ratio": float(y.mean()),
        "feature_columns": list(X.columns),
        "models_trained": list(all_metrics.keys()),
        **all_metrics
    }

    metrics_path = os.path.join(METRICS_DIR, "fraud_model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics to {metrics_path}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()