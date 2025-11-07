# models/predict_one.py
# ------------------------------------------------------------
# Purpose:
#   Load the trained Pipeline (preprocessing + model) and
#   run inference on a single record (Python dict).
#   Returns human-friendly labels: "Approved"/"Rejected".
#
# How to run:
#   python models/predict_one.py
#   (Edit the 'sample' dict below for quick sanity checks.)
# ------------------------------------------------------------

import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/model.pkl")

# Load the full pipeline once (fast inference afterwards)
pipe = joblib.load(MODEL_PATH)


def predict_one(record: dict):
    """
    Predict on a single record (dict of column_name -> value).
    The pipeline handles imputation/encoding internally.
    Returns:
      {"prediction": "Approved"/"Rejected", "confidence": float or None}
    """
    df = pd.DataFrame([record])
    pred = pipe.predict(df)[0]

    # Ensure human-readable label (in case model returns non-string)
    if isinstance(pred, str):
        pred_label = "Approved" if pred.lower().startswith("a") else "Rejected"
    else:
        pred_label = "Approved" if int(pred) == 0 else "Rejected"

    proba = None
    clf = pipe.named_steps.get("clf")
    if clf is not None and hasattr(clf, "predict_proba"):
        # Confidence = highest class probability
        proba = float(pipe.predict_proba(df)[0].max())

    return {"prediction": pred_label, "confidence": proba}


if __name__ == "__main__":
    # Example record (adjust fields to your CSV columns)
    sample = {
        "person_age": 35,
        "person_gender": "male",
        "person_education": "Bachelor",
        "person_income": 55000,
        "person_emp_exp": 5,
        "person_home_ownership": "RENT",
        "loan_amnt": 12000,
        "loan_intent": "EDUCATION",
        "loan_int_rate": 7.5,
        "loan_percent_income": 0.22,
        "cb_person_cred_hist_length": 8,
        "credit_score": 690,
        "previous_loan_defaults_on_file": "N",
    }

    result = predict_one(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))
