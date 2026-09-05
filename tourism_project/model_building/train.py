from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import BaggingClassifier

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
DEPLOY_DIR = ROOT / "deployment"
TARGET = "ProdTaken"
RANDOM_STATE = 42

def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.round(np.arange(0.10, 0.91, 0.01), 2)
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in candidates]
    best_idx = int(np.argmax(scores))
    return float(candidates[best_idx]), float(scores[best_idx])

def main() -> None:
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(ARTIFACT_DIR / "train.csv")
    test = pd.read_csv(ARTIFACT_DIR / "test.csv")
    X_train, y_train = train.drop(columns=TARGET), train[TARGET].astype(int)
    X_test, y_test = test.drop(columns=TARGET), test[TARGET].astype(int)

    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

    # Mirrors the interactive preprocessor defined in Section 9.1 exactly
    # (same step names and sparse_output setting) to avoid silent drift
    # between the notebook's exploratory pipeline and this production script.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_features),
        ],
        remainder="drop",
    )

    # NOTE: BaggingClassifier is used here because it was the top-ranked model
    # in the Section 9.2-9.5 benchmarking matrix (highest Test PR-AUC and F1
    # among all 7 candidates evaluated in the notebook). This keeps the
    # production script consistent with the notebook's own model-selection
    # evidence rather than defaulting to a different algorithm.
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", BaggingClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    parameter_grid = {
        "model__n_estimators": [50, 100],
        "model__max_samples": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline, parameter_grid, scoring="average_precision", cv=cv,
        n_jobs=-1, refit=True
    )

    mlflow.set_tracking_uri((ROOT / "mlruns").resolve().as_uri())
    mlflow.set_experiment("tourism_package_prediction")

    with mlflow.start_run(run_name="production_bagging_pipeline"):
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        oof_probabilities = cross_val_predict(
            best_model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        threshold, oof_f1 = select_threshold(y_train, oof_probabilities)

        probabilities = best_model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

        metrics = {
            "test_roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "test_pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
            "test_precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "test_recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "test_f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "test_specificity": round(float(tn / (tn + fp)), 4),
            "operating_threshold": round(float(threshold), 2),
            "oof_f1_at_threshold": round(float(oof_f1), 4),
            "test_contacts_flagged": int(predictions.sum()),
            "test_true_positives": int(tp),
        }

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        RocCurveDisplay.from_predictions(y_test, probabilities, ax=axes[0], name="Bagging Classifier")
        axes[0].plot([0, 1], [0, 1], "--", color="grey")
        PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=axes[1], name="Bagging Classifier")
        axes[1].axhline(y_test.mean(), linestyle="--", color="grey")
        fig.tight_layout()
        figure_path = ARTIFACT_DIR / "model_evaluation.png"
        fig.savefig(figure_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        importance = permutation_importance(
            best_model, X_test, y_test, scoring="average_precision", n_repeats=10,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        importance_frame = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }).sort_values("importance_mean", ascending=False)
        importance_path = ARTIFACT_DIR / "permutation_importance.csv"
        importance_frame.to_csv(importance_path, index=False)

        bundle = {
            "model": best_model,
            "threshold": threshold,
            "feature_columns": X_train.columns.tolist(),
            "model_type": "BaggingClassifier",
        }
        model_path = DEPLOY_DIR / "best_model.joblib"
        joblib.dump(bundle, model_path)

        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        model_card = {
            "model_type": "BaggingClassifier",
            "target": TARGET,
            "data_split": "80/20 stratified, random_state=42",
            "best_parameters": search.best_params_,
            "metrics": metrics,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "classification_report": report,
            "limitations": [
                "Model reflects historical campaign patterns; retrain quarterly.",
                "Designed as decision support to prioritize sales outreach, not autonomous exclusion.",
                "Review demographic fairness metrics across city tiers and gender."
            ],
        }
        card_path = DEPLOY_DIR / "model_card.json"
        card_path.write_text(json.dumps(model_card, indent=2, default=float), encoding="utf-8")

        mlflow.log_artifact(str(figure_path), artifact_path="evaluation")
        mlflow.log_artifact(str(importance_path), artifact_path="interpretability")
        mlflow.log_artifact(str(card_path), artifact_path="model_card")
        mlflow.sklearn.log_model(best_model, artifact_path="model")

    print("=" * 60)
    print("           PRODUCTION MODEL TRAINING & REGISTRATION SUCCESS ")
    print("=" * 60)
    print(f"  Best Parameters      : {search.best_params_}")
    print(f"  Operating Threshold  : {threshold:.2f}")
    print(f"  Test F1-Score        : {metrics['test_f1']:.4f}")
    print(f"  Test PR-AUC          : {metrics['test_pr_auc']:.4f}")
    print(f"  Test ROC-AUC         : {metrics['test_roc_auc']:.4f}")
    print(f"  Model Serialized to  : {model_path}")
    print(f"  Model Card Written   : {card_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()