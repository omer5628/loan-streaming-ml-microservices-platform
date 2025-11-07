# db.py
"""
Database models and session management for the loan prediction system.

This module defines:
  - SQLAlchemy engine and session factory.
  - The LoanPredictionORM model used to store loan prediction records.

"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Text, Float
from sqlalchemy.orm import sessionmaker, declarative_base

# Use DATABASE_URL from environment or fall back to a sensible default
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://app:app@postgres:5432/appdb"
)

# SQLAlchemy engine and session factory
engine = create_engine(DB_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Base class for ORM models
Base = declarative_base()


class LoanPredictionORM(Base):
    """
    ORM model for the 'loan_predictions' table.

    Columns:
      id             - Auto-increment primary key.
      features_json  - Raw features of the loan request as JSON string.
      predicted_label- Model prediction (e.g. "Approved" / "Rejected").
      confidence     - Prediction confidence (probability), if available.
      true_label     - Ground-truth label from the dataset, if known.
    """
    __tablename__ = "loan_predictions"

    id = Column(Integer, primary_key=True, index=True)
    features_json = Column(Text, nullable=False)
    predicted_label = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=True)
    true_label = Column(String(32), nullable=True)
