import os
import json
from pathlib import Path

import joblib
import pandas as pd
from celery import Celery

from db import SessionLocal, LoanPredictionORM

# --------- Celery / RabbitMQ configuration ---------
BROKER = os.getenv("CELERY_BROKER_URL", "amqp://customuser:custompass@rabbitmq:5672//")
BACKEND = os.getenv("CELERY_RESULT_BACKEND", "rpc://")

# Celery application instance that will listen for tasks from RabbitMQ.
app = Celery("worker", broker=BROKER, backend=BACKEND)

# --------- Model and schema loading ---------
MODEL_PATH = Path("models/model.pkl")
SCHEMA_PATH = Path("models/schema.json")

# The worker cannot run without a trained model file.
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at: {MODEL_PATH}")

# Load the trained scikit-learn Pipeline (preprocessing + classifier).
pipe = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = None
TARGET_COLUMN = "loan_status"

# Optionally load the schema to know the expected feature columns and target.
if SCHEMA_PATH.exists():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    FEATURE_COLUMNS = schema.get("feature_columns")
    TARGET_COLUMN = schema.get("target_column") or schema.get("target") or "loan_status"


def _run_prediction(record: dict):
    """
    Run a prediction on a single record.

    Args:
        record: A dictionary of raw feature values for one loan request.

    Returns:
        A tuple (label, confidence):
          - label: normalized string label ("Approved" / "Rejected").
          - confidence: maximum predicted probability (float) if available,
            otherwise None.
    """
    # Build a one-row DataFrame from the input feature dictionary.
    df = pd.DataFrame([record])

    # If we have a feature list from the schema, ensure:
    #   - all expected columns exist in the DataFrame
    #   - columns are ordered exactly as the model expects.
    if FEATURE_COLUMNS:
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[FEATURE_COLUMNS]

    # Run the scikit-learn pipeline to get the raw prediction.
    pred = pipe.predict(df)[0]

    # Normalize prediction to human-readable label "Approved" / "Rejected".
    # This mirrors the logic used in the training/evaluation scripts.
    if isinstance(pred, str):
        label = "Approved" if pred.lower().startswith("a") else "Rejected"
    else:
        label = "Approved" if int(pred) == 0 else "Rejected"

    # Try to compute a confidence score using predict_proba, if available.
    confidence = None
    clf = pipe.named_steps.get("clf")
    if clf is not None and hasattr(clf, "predict_proba"):
        proba = pipe.predict_proba(df)[0]
        confidence = float(proba.max())

    return label, confidence


@app.task(bind=True, name="predict_loan")
def predict_loan(self, record: dict, true_label=None):
    """
    Celery task that:
      1. Receives a loan record as a feature dictionary.
      2. Runs the ML model to obtain a predicted label and confidence.
      3. Persists the result into PostgreSQL (loan_predictions table).

    Args:
        self: Celery task instance (unused, but available for retries/logging).
        record: Dict of feature_name -> value for a single loan.
        true_label: Optional ground-truth label from the dataset (raw form).

    Returns:
        A dictionary containing the stored record:
          {
            "id": <database primary key>,
            "predicted_label": <"Approved" / "Rejected">,
            "confidence": <float or None>,
            "true_label": <original label or None>,
          }
    """
    # 1. Run prediction using the loaded pipeline.
    predicted_label, confidence = _run_prediction(record)

    # 2. Open a new database session for this task.
    session = SessionLocal()
    try:
        # 3. Create a new ORM row representing this prediction.
        row = LoanPredictionORM(
            features_json=json.dumps(record, ensure_ascii=False),
            predicted_label=predicted_label,
            confidence=confidence,
            true_label=true_label,
        )
        # 4. Add and commit the new row to the database.
        session.add(row)
        session.commit()
        session.refresh(row)  # refresh to get auto-generated fields (e.g., id)

        # 5. Return a lightweight representation of the stored record.
        return {
            "id": row.id,
            "predicted_label": row.predicted_label,
            "confidence": row.confidence,
            "true_label": row.true_label,
        }
    except Exception as e:
        # In case of any error, roll back the transaction so the DB stays consistent.
        session.rollback()
        raise e
    finally:
        # Always close the session to release the connection back to the pool.
        session.close()
