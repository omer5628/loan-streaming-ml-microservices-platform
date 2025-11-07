# models/train.py
# ------------------------------------------------------------
# Purpose:
#   Train a full scikit-learn Pipeline for tabular classification
#   (imputation + one-hot encoding + RandomForest model).
#   Normalize target labels to "Approved"/"Rejected".
#   Save both the trained pipeline and a simple schema for reference.
#
# How to run:
#   python models/train.py --csv ./data/loan_data.csv --target loan_status
#   (add --sep ";" if your CSV uses semicolon)
# ------------------------------------------------------------

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Automatically detect which columns are numeric and which are categorical.
# It removes the target column, checks data types:
# - Columns with dtype 'object' → categorical
# - All others → numeric

def infer_types(df: pd.DataFrame, target: str):
    """Infer numeric and categorical feature lists from dataframe dtypes."""
    features = [c for c in df.columns if c != target]
    cat_cols = [c for c in features if df[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]
    return num_cols, cat_cols

#creates a data preprocessing pipeline — a system that automatically
# cleans and prepares the dataset before sending it to a machine learning model.
def build_pipeline(num_cols, cat_cols):
    """Build a preprocessing + model pipeline."""
    # Numeric: median imputation
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
    ])
    # Categorical: mode imputation + one-hot
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # Column-wise preprocessing
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
   
    # Creates a Random Forest model with 300 trees and balanced class weights,
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    # builds a pipeline that first preprocesses the data (pre) and then trains the classifier (clf).
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe


def normalize_target_labels(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Map various target encodings to standardized string labels:
      Approved / Rejected
    Works for 0/1 or textual variants.
    """
    mapping = {
        0: "Approved",
        1: "Rejected",
        "0": "Approved",
        "1": "Rejected",
        "approved": "Approved",
        "rejected": "Rejected",
        "yes": "Approved",
        "no": "Rejected",
        "Approved": "Approved",
        "Rejected": "Rejected",
    }
    if target in df.columns:
        df[target] = df[target].map(mapping).fillna(df[target])
    return df


def main():
    # --------- CLI arguments ---------
    ap = argparse.ArgumentParser(description="Train a tabular classification pipeline.")
    ap.add_argument("--csv", required=True, help="Path to input CSV file")
    ap.add_argument("--target", required=True, help="Target (label) column name")
    ap.add_argument("--sep", default=",", help="CSV delimiter (default: ,)")
    ap.add_argument("--model_out", default="models/model.pkl", help="Output path for trained pipeline")
    ap.add_argument("--schema_out", default="models/schema.json", help="Output path for schema JSON")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # --------- Load & normalize data ---------
    df = pd.read_csv(csv_path, sep=args.sep).dropna(how="all").reset_index(drop=True)
    if args.target not in df.columns:
        print(f"[ERR] Target '{args.target}' not in columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Normalize labels to Approved/Rejected
    df = normalize_target_labels(df, args.target)

    X = df.drop(columns=[args.target])
    y = df[args.target]

    # --------- Build pipeline ---------
    num_cols, cat_cols = infer_types(df, args.target)
    pipe = build_pipeline(num_cols, cat_cols)

    # --------- Train/test split ---------
    stratify = y if y.nunique() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # --------- Fit model ---------
    pipe.fit(Xtr, ytr)

    # --------- Basic evaluation ---------
    ypred = pipe.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(yte, ypred))

    # --------- Save artifacts ---------
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.schema_out).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, args.model_out)
    
    # Creates a JSON schema describing the dataset structure:
    # - numeric and categorical feature names
    # - target column
    # - unique label values
    # - path of the CSV used for training
    # Saves this info to a file for consistent use during prediction.
    schema = {
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "target": args.target,
        "label_values": sorted(y.unique().tolist()),
        "csv_used": str(csv_path),
    }
    with open(args.schema_out, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"Saved model to: {args.model_out}")
    print(f"Saved schema to: {args.schema_out}")


if __name__ == "__main__":
    main()
