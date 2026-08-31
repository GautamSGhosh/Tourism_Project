from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tourism.csv"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "ProdTaken"
RANDOM_STATE = 42

def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load directly from repository data directory
    df = pd.read_csv(DATA_PATH).copy()

    # 2. Data cleaning & typo normalization
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # 3. Remove unnecessary columns & identifiers
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if "CustomerID" in df.columns:
        drop_cols.append("CustomerID")

    df = df.drop(columns=drop_cols, errors="ignore").drop_duplicates().reset_index(drop=True)

    # 4. Target validation
    if df[TARGET].isna().any():
        raise ValueError("Target ProdTaken contains missing values!")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # 5. Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # 6. Save split files locally for pipeline artifact passing
    X_train.to_csv(ARTIFACT_DIR / "Xtrain.csv", index=False)
    X_test.to_csv(ARTIFACT_DIR / "Xtest.csv", index=False)
    y_train.to_csv(ARTIFACT_DIR / "ytrain.csv", index=False)
    y_test.to_csv(ARTIFACT_DIR / "ytest.csv", index=False)

    train_df = X_train.copy(); train_df[TARGET] = y_train
    test_df = X_test.copy(); test_df[TARGET] = y_test
    train_df.to_csv(ARTIFACT_DIR / "train.csv", index=False)
    test_df.to_csv(ARTIFACT_DIR / "test.csv", index=False)

    # 7. Metadata audit
    metadata = {
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "dropped_columns": drop_cols,
        "train_records": len(train_df),
        "test_records": len(test_df),
        "train_positive_prevalence": round(float(y_train.mean()), 4),
        "test_positive_prevalence": round(float(y_test.mean()), 4),
        "feature_count": X_train.shape[1],
        "feature_names": X_train.columns.tolist()
    }

    meta_path = ARTIFACT_DIR / "split_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 60)
    print("           DATA PREPARATION & SPLIT REPORT                  ")
    print("=" * 60)
    print(f"  Training Set Size    : {len(X_train)} samples ({y_train.mean():.2%} positive)")
    print(f"  Testing Set Size     : {len(X_test)} samples ({y_test.mean():.2%} positive)")
    print(f"  Features Retained    : {X_train.shape[1]} features")
    print(f"  Artifact Directory   : {ARTIFACT_DIR.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    main()