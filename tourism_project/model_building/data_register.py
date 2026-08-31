from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tourism.csv"

REQUIRED_COLUMNS = {
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
}

def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Registered dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # 1. Check schema
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Schema validation failed! Missing expected columns: {sorted(missing_cols)}")

    # 2. Check target validity
    target_vals = set(df["ProdTaken"].dropna().unique())
    if not target_vals.issubset({0, 1}):
        raise ValueError(f"Target ProdTaken must be binary {0, 1}. Found: {target_vals}")

    # 3. Print registration summary
    summary = {
        "Status": "VALIDATED & REGISTERED",
        "Total Rows": len(df),
        "Total Columns": len(df.columns),
        "Target Prevalence (ProdTaken=1)": f"{df['ProdTaken'].mean():.2%}",
        "Missing Cell Count": int(df.isna().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
    }

    print("=" * 60)
    print("           DATA REGISTRATION & SCHEMA AUDIT REPORT          ")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:35s} : {v}")
    print("=" * 60)
    print("\nVerified Columns:\n" + ", ".join(sorted(df.columns)))
    print("=" * 60)

if __name__ == "__main__":
    main()