# aggregator.py
"""
Aggregator microservice for the loan prediction system.

This service is responsible for:
  - Reading loan prediction records from PostgreSQL.
  - Computing aggregated statistics such as:
      * total number of records
      * count of records per predicted_label
      * count of records per true_label
      * confusion-like matrix of predicted vs true labels
  - Computing dynamic segment aggregations by a chosen feature
    (e.g., person_gender, loan_intent, person_home_ownership).
  - Exposing these aggregates via a simple HTTP JSON API.

Typical flow:
  Producer  ->  Celery worker  ->  PostgreSQL  ->  Aggregator  ->  Flask UI
"""

import json
from flask import Flask, jsonify, request
from sqlalchemy import func

from db import SessionLocal, LoanPredictionORM

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # allow UTF-8 in JSON responses


# -----------------------------------------------------------------------------
# Global aggregates
# -----------------------------------------------------------------------------
@app.get("/aggregates")
def get_aggregates():
    """
    Return aggregated statistics about loan predictions.

    Response JSON structure:
      {
        "total_count": int,
        "by_predicted_label": { "<label>": count, ... },
        "by_true_label": { "<label>": count, ... },
        "predicted_vs_true": {
          "<predicted_label>": { "<true_label>": count, ... },
          ...
        }
      }
    """
    session = SessionLocal()
    try:
        # Total number of records
        total_count = session.query(func.count(LoanPredictionORM.id)).scalar() or 0

        # Count per predicted_label
        by_predicted = {}
        rows = (
            session.query(
                LoanPredictionORM.predicted_label,
                func.count(LoanPredictionORM.id),
            )
            .group_by(LoanPredictionORM.predicted_label)
            .all()
        )
        for label, cnt in rows:
            key = label if label is not None else "None"
            by_predicted[key] = int(cnt)

        # Count per true_label (ignoring NULLs)
        by_true = {}
        rows = (
            session.query(
                LoanPredictionORM.true_label,
                func.count(LoanPredictionORM.id),
            )
            .filter(LoanPredictionORM.true_label.isnot(None))
            .group_by(LoanPredictionORM.true_label)
            .all()
        )
        for label, cnt in rows:
            key = label if label is not None else "None"
            by_true[key] = int(cnt)

        # Confusion-like matrix: predicted vs true
        predicted_vs_true = {}
        rows = (
            session.query(
                LoanPredictionORM.predicted_label,
                LoanPredictionORM.true_label,
                func.count(LoanPredictionORM.id),
            )
            .group_by(
                LoanPredictionORM.predicted_label,
                LoanPredictionORM.true_label,
            )
            .all()
        )
        for predicted, true_label, cnt in rows:
            p_key = predicted if predicted is not None else "None"
            t_key = true_label if true_label is not None else "None"
            predicted_vs_true.setdefault(p_key, {})
            predicted_vs_true[p_key][t_key] = int(cnt)

        result = {
            "total_count": int(total_count),
            "by_predicted_label": by_predicted,
            "by_true_label": by_true,
            "predicted_vs_true": predicted_vs_true,
        }

        print("[AGGREGATOR] /aggregates ->", result)
        return jsonify(result), 200

    except Exception as e:
        print("[AGGREGATOR] Error in /aggregates:", repr(e))
        return jsonify({"error": str(e)}), 500

    finally:
        session.close()


# -----------------------------------------------------------------------------
# Dynamic segmentation endpoint
# -----------------------------------------------------------------------------
@app.get("/segments")
def get_segments():
    """
    Return dynamic segmentation statistics for a chosen feature.

    Query parameters:
      - feature: name of the feature inside features_json
                 (e.g., person_gender, loan_intent, person_home_ownership).

    For each distinct value of the feature, the API computes:
      - total number of records with that value
      - number of Approved predictions
      - number of Rejected predictions
      - approval_rate (approved / total)

    Example response:
      {
        "feature": "person_gender",
        "segments": [
          {
            "value": "male",
            "total": 40,
            "approved": 10,
            "rejected": 30,
            "approval_rate": 0.25
          },
          ...
        ]
      }
    """
    feature = request.args.get("feature")
    if not feature:
        return jsonify({"error": "Missing required query parameter: feature"}), 400

    session = SessionLocal()
    try:
        rows = session.query(LoanPredictionORM).all()

        segments = {}  # value -> stats dict

        for row in rows:
            try:
                features = json.loads(row.features_json)
            except Exception:
                # Skip rows with malformed JSON
                continue

            raw_value = features.get(feature, "MISSING")
            if raw_value is None:
                raw_value = "MISSING"

            key = str(raw_value)
            if key not in segments:
                segments[key] = {
                    "value": key,
                    "total": 0,
                    "approved": 0,
                    "rejected": 0,
                }

            segments[key]["total"] += 1
            if row.predicted_label == "Approved":
                segments[key]["approved"] += 1
            elif row.predicted_label == "Rejected":
                segments[key]["rejected"] += 1

        segment_list = []
        for seg in segments.values():
            total = seg["total"]
            seg["approval_rate"] = (seg["approved"] / total) if total > 0 else 0.0
            segment_list.append(seg)

        # Sort segments by total descending for a nicer UI
        segment_list.sort(key=lambda s: s["total"], reverse=True)

        result = {
            "feature": feature,
            "segments": segment_list,
        }

        print(f"[AGGREGATOR] /segments?feature={feature} -> {result}")
        return jsonify(result), 200

    except Exception as e:
        print("[AGGREGATOR] Error in /segments:", repr(e))
        return jsonify({"error": str(e)}), 500

    finally:
        session.close()


# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    """
    Health check endpoint for the aggregator service.
    """
    return {"status": "ok"}, 200


if __name__ == "__main__":
    # Aggregator runs on port 8001 inside the Docker network.
    app.run(host="0.0.0.0", port=8001)
