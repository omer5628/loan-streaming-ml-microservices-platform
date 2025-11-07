# producer.py
"""
Producer service for the loan prediction microservices system.

This script is responsible for:
  1. Loading the loan dataset from a CSV file.
  2. Iterating over the rows one by one.
  3. Building a feature dictionary for each row according to the model schema.
  4. Sending each row as an asynchronous Celery task to the prediction worker.
  5. Waiting a configurable delay between rows to simulate a streaming data source.

Flow:
    CSV row  --->  Celery task (predict_loan)  --->  RabbitMQ  --->  Celery worker
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd

# Import the Celery task from the worker module.
# The worker module must define:
#   app = Celery(...)
#   @app.task(name="predict_loan")
#   def predict_loan(record: dict, true_label=None): ...
from worker import predict_loan

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the CSV dataset. This should be mounted inside the container.
DATA_CSV_PATH = Path(os.getenv("DATA_CSV_PATH", "data/loan_data.csv"))

# Path to the JSON schema describing feature columns and target column.
SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", "models/schema.json"))

# Delay (in seconds) between sending two consecutive records.
PRODUCER_SLEEP_SECONDS = float(os.getenv("PRODUCER_SLEEP_SECONDS", "5.0"))

# These will be populated after reading the schema.
FEATURE_COLUMNS = []
TARGET_COLUMN = "loan_status"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_schema() -> None:
    """
    Load the schema JSON file and populate FEATURE_COLUMNS and TARGET_COLUMN.

    The schema is expected to contain:
      - "feature_columns": list of feature names used by the model.
      - "target_column" or "target": name of the label column (optional).
    """
    global FEATURE_COLUMNS, TARGET_COLUMN

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Schema file not found at: {SCHEMA_PATH}. "
            f"Make sure models/schema.json is available inside the container."
        )

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    feature_cols = schema.get("feature_columns")
    if not feature_cols:
        raise ValueError(
            "Schema JSON does not contain 'feature_columns'. "
            "Cannot build feature dictionaries for the model."
        )

    FEATURE_COLUMNS = feature_cols
    TARGET_COLUMN = schema.get("target_column") or schema.get("target") or "loan_status"

    print(f"[PRODUCER] Loaded schema from {SCHEMA_PATH}")
    print(f"[PRODUCER] Feature columns: {FEATURE_COLUMNS}")
    print(f"[PRODUCER] Target column: {TARGET_COLUMN}")


def load_dataset() -> pd.DataFrame:
    """
    Load the loan dataset from the configured CSV path.

    Returns:
        A pandas DataFrame containing the full dataset.

    Raises:
        FileNotFoundError: if the CSV file does not exist.
        ValueError: if required feature columns are missing.
    """
    if not DATA_CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found at: {DATA_CSV_PATH}. "
            f"Make sure the dataset is mounted to this path in docker-compose."
        )

    print(f"[PRODUCER] Loading dataset from: {DATA_CSV_PATH}")
    df = pd.read_csv(DATA_CSV_PATH)

    # Verify that all feature columns from the schema are present in the CSV
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following feature columns from schema are missing in the CSV: {missing_cols}"
        )

    print(f"[PRODUCER] Loaded {len(df)} rows from dataset.")
    return df


def build_record_from_row(row: pd.Series) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Build a feature dictionary and optional true label from a DataFrame row.

    Args:
        row: A pandas Series representing one row of the dataset.

    Returns:
        A tuple (record_dict, true_label_str):
          - record_dict: dict mapping feature names to values (ready for the model).
          - true_label_str: the raw label value converted to string (if present),
            or None if the target column does not exist or is NaN.

    Note:
        We intentionally store the true label as a string without trying
        to guess whether 0/1 means "Approved"/"Rejected". The model itself
        is trained on normalized string labels, but for evaluation we keep
        the original raw label value.
    """
    # Build the features dict according to schema
    record = {col: row[col] for col in FEATURE_COLUMNS}

    true_label = None
    if TARGET_COLUMN in row.index:
        value = row[TARGET_COLUMN]
        # Check for NaN (pandas uses float('nan') for missing values)
        if pd.notna(value):
            true_label = str(value)

    return record, true_label


# ---------------------------------------------------------------------------
# Main producer loop
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point for the producer service.

    Flow:
      1. Load schema to know which columns to send to the model.
      2. Load the dataset from CSV into a DataFrame.
      3. In an infinite loop:
         - Iterate over all rows in the DataFrame.
         - For each row:
             * Build feature dict + true label.
             * Send Celery task: predict_loan.delay(record, true_label).
             * Sleep for PRODUCER_SLEEP_SECONDS.
         - When the end of the dataset is reached, start again from the first row.
    """
    print("[PRODUCER] Starting producer service...")
    print(f"[PRODUCER] DATA_CSV_PATH={DATA_CSV_PATH}")
    print(f"[PRODUCER] SCHEMA_PATH={SCHEMA_PATH}")
    print(f"[PRODUCER] PRODUCER_SLEEP_SECONDS={PRODUCER_SLEEP_SECONDS}")

    # Step 1: load schema (feature columns + target column)
    load_schema()

    # Step 2: load dataset into memory
    df = load_dataset()

    iteration = 0

    # Step 3: infinite loop over the dataset
    while True:
        iteration += 1
        print(f"[PRODUCER] Starting iteration #{iteration} over dataset ({len(df)} rows).")

        for index, row in df.iterrows():
            record, true_label = build_record_from_row(row)

            # Send asynchronous Celery task to the prediction worker.
            # Celery will publish a message to RabbitMQ (using the broker URL
            # configured in the worker module).
            async_result = predict_loan.delay(record, true_label=true_label)

            print(
                f"[PRODUCER] Enqueued row index={index} as Celery task id={async_result.id} "
                f"true_label={true_label}"
            )

            # Sleep between records to simulate a streaming source
            time.sleep(PRODUCER_SLEEP_SECONDS)

        # Once we reach the end of the dataset, we simply start over.
        print(
            f"[PRODUCER] Finished one full pass over the dataset. "
            f"Restarting from the first row after a short delay..."
        )
        time.sleep(PRODUCER_SLEEP_SECONDS)


if __name__ == "__main__":
    main()
