
"""
One-time database initialization script.

This script is responsible for:
  - Connecting to the PostgreSQL database using SQLAlchemy.
  - Creating all tables defined in the ORM models (currently LoanPredictionORM).

It should be executed once on deployment (e.g., via a dedicated docker
service) before other services that rely on the database start.
"""

from db import Base, engine


def init_db() -> None:
    """
    Create all database tables defined in the ORM metadata.

    This is safe to run if the tables do not exist yet.
    It should NOT be run concurrently by multiple services.
    """
    print("[DB_INIT] Creating database tables (if not existing)...")
    Base.metadata.create_all(bind=engine)
    print("[DB_INIT] Database initialization completed.")


if __name__ == "__main__":
    init_db()
