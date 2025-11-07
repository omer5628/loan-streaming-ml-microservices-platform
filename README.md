# Loan Streaming ML Microservices Platform

End-to-end **loan prediction streaming platform** using **microservices**, **RabbitMQ**, **Celery**, **PostgreSQL**, **scikit-learn**, **Flask**, and **Nginx**.

The system simulates a real-time data stream of loan applications, runs them through a trained ML model, stores predictions in a database, and exposes a small dashboard + APIs for monitoring.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Services](#services)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Running with Docker Compose](#running-with-docker-compose)
- [Exposed Endpoints](#exposed-endpoints)
- [Data Flow](#data-flow)
- [Future Improvements](#future-improvements)

---

## Architecture Overview

The platform is built as a small microservices system:

1. **Producer** – streams rows from a loan CSV file and sends them as async tasks.
2. **RabbitMQ** – message broker used by Celery.
3. **Celery Worker** – loads a trained scikit-learn model and runs predictions.
4. **PostgreSQL** – stores loan prediction records.
5. **Aggregator Service** – computes aggregates/segments over the prediction table.
6. **Flask UI (x2 instances)** – reads from PostgreSQL and shows a simple dashboard.
7. **Nginx** – reverse proxy + load balancer in front of the Flask + Aggregator services.

High-level flow:

> CSV → Producer → RabbitMQ → Celery Worker (ML model) → PostgreSQL → Aggregator + UI (via Nginx)

---

## Tech Stack

- **Language:** Python 3.11
- **ML:** scikit-learn (trained `Pipeline` saved with `joblib`)
- **Message Broker:** RabbitMQ (with management UI)
- **Task Queue:** Celery
- **Database:** PostgreSQL + SQLAlchemy ORM
- **Web:** Flask (UI + Aggregator APIs)
- **Reverse Proxy / Load Balancer:** Nginx
- **Containerization:** Docker, Docker Compose

---

## Services

Defined in `docker-compose.yml`:

- **rabbitmq**
  - Image: `rabbitmq:3-management`
  - Ports:
    - `5672` – AMQP (used by Celery)
    - `15672` – Management UI

- **postgres**
  - Image: `postgres:16`
  - Stores the `loan_predictions` table.

- **db_init**
  - One-time init container.
  - Runs `db_init.py` to create database tables.

- **celery_worker**
  - Runs `celery -A worker worker --loglevel=info`.
  - Loads `models/model.pkl` and `models/schema.json`.
  - Consumes tasks from RabbitMQ and writes predictions to PostgreSQL.

- **producer**
  - Runs `producer.py`.
  - Streams rows from `data/loan_data.csv` according to `models/schema.json`.
  - Sends async Celery tasks (`predict_loan`) with features + true label.
  - Sleep between rows is configurable (`PRODUCER_SLEEP_SECONDS`).

- **web1**, **web2**
  - Run `app.py` (Flask UI).
  - Expose a simple dashboard and REST API for all loan predictions.
  - Nginx load-balances between them.

- **aggregator**
  - Runs `aggregator.py` (Flask).
  - Exposes `/aggregates` and `/segments` JSON endpoints.

- **nginx**
  - Image: `nginx:1.27-alpine`
  - Uses custom `nginx.conf`.
  - Listens on **host port `8000`**.
  - Proxies traffic to:
    - `web1` / `web2` (UI)
    - `aggregator` (aggregated statistics)

---

## Project Structure

```text
.
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── .env
├── requirements.txt
├── app.py              # Flask UI (dashboard + /loans + health)
├── aggregator.py       # Aggregator microservice (/aggregates, /segments, /health)
├── producer.py         # CSV → Celery tasks (streaming producer)
├── worker.py           # Celery worker (loads model + writes to DB)
├── db.py               # SQLAlchemy engine + LoanPredictionORM
├── db_init.py          # One-time DB initialization
└── models/
    └── schema.json     # Model schema (features, target, label values, CSV used)
    # model.pkl        # (Not in repo) Trained scikit-learn Pipeline

You are expected to provide:

models/model.pkl – trained scikit-learn Pipeline compatible with schema.json.

data/loan_data.csv – loan dataset used by the producer (path configurable).

Prerequisites

Docker and Docker Compose installed.

A trained scikit-learn model (model.pkl) that matches the schema.

A CSV dataset with the same columns as defined in models/schema.json.

Configuration
1. Environment variables (.env)

Example (already included in the repo):

# RabbitMQ
RABBITMQ_USER=customuser
RABBITMQ_PASS=custompass
RABBITMQ_URL=amqp://customuser:custompass@rabbitmq:5672//

# Celery
CELERY_BROKER_URL=amqp://customuser:custompass@rabbitmq:5672//
CELERY_RESULT_BACKEND=rpc://

# Postgres
POSTGRES_USER=app
POSTGRES_PASSWORD=app
POSTGRES_DB=appdb
DATABASE_URL=postgresql+psycopg2://app:app@postgres:5432/appdb


You can adjust usernames/passwords/DB name as needed.

2. Model + Schema

models/schema.json describes:

numeric_features

categorical_features

target / target_column

feature_columns

label_values

models/model.pkl must be a scikit-learn Pipeline that expects exactly these feature_columns.

Note: model.pkl is not committed to GitHub due to size limits. You should train it separately and copy it into models/model.pkl.

3. Dataset

Default path: data/loan_data.csv

Configurable via DATA_CSV_PATH in docker-compose.yml (environment for producer).

The CSV must contain all columns from feature_columns and the target column.

Running with Docker Compose

Clone the repository

git clone https://github.com/omer5628/loan-streaming-ml-microservices-platform.git
cd loan-streaming-ml-microservices-platform


Add the trained model

mkdir -p models
# Copy your trained model here:
# models/model.pkl


Add the dataset

mkdir -p data
# Copy your CSV here:
# data/loan_data.csv


Check/update .env (optional, or keep defaults)

Build and start all services

docker compose up --build


Docker Compose will:

Start RabbitMQ + PostgreSQL

Run db_init once to create tables

Start:

celery_worker

producer

web1, web2

aggregator

nginx

Open the UI

Go to: http://localhost:8000/

As the producer loops over the dataset, new predictions will appear in the DB and in the dashboard.

Exposed Endpoints

Through Nginx (http://localhost:8000):

Flask UI (app.py)

GET /
HTML dashboard for loan predictions (simple template).

GET /loans
Returns all predictions as JSON:

[
  {
    "id": 1,
    "predicted_label": "Approved",
    "confidence": 0.92,
    "true_label": "Approved",
    "features_json": "{...}"
  },
  ...
]


GET /health
Simple health check for the UI.

GET /debug/loans/count
Returns only the number of rows in loan_predictions (for quick checks).

Aggregator (aggregator.py)

GET /aggregates
Global aggregates:

{
  "total_count": 123,
  "by_predicted_label": { "Approved": 80, "Rejected": 43 },
  "by_true_label": { "Approved": 75, "Rejected": 48 },
  "predicted_vs_true": {
    "Approved": { "Approved": 70, "Rejected": 10 },
    "Rejected": { "Approved": 5, "Rejected": 38 }
  }
}


GET /segments?feature=person_gender
Dynamic segmentation for any feature inside features_json:

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


GET /health
Health check for the aggregator service.

Data Flow

Detailed step-by-step:

Producer

Loads models/schema.json.

Loads data/loan_data.csv into a pandas DataFrame.

Iterates over rows:

Builds a feature dict (feature_columns).

Extracts optional true label (target_column).

Sends predict_loan.delay(record, true_label) via Celery.

Sleeps PRODUCER_SLEEP_SECONDS.

Celery Worker

Receives predict_loan tasks from RabbitMQ.

Uses models/model.pkl (scikit-learn Pipeline) to predict:

Normalized label: "Approved" or "Rejected".

Optional confidence score using predict_proba.

Writes a new row into loan_predictions (PostgreSQL).

Aggregator + UI

Read from loan_predictions.

Aggregator computes:

Total count

Distribution by predicted/true labels

Confusion matrix

Dynamic segments

UI:

Calls /loans, /aggregates, /segments to render data on the page.

Future Improvements

Some ideas to extend this project:

Add training pipeline scripts directly into this repo.

Add authentication in front of the dashboard.

Add Grafana/Prometheus for metrics and monitoring.

Add Kafka instead of RabbitMQ for higher throughput use cases.

Deploy to Kubernetes with proper Helm charts.
