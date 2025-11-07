# Dockerfile
# -----------------------------------------------------------------------------
# Base image: lightweight Python
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# -----------------------------------------------------------------------------
# System-level dependencies (optional; keep minimal for this assignment)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Workdir inside the container
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# Install Python dependencies
#   - We copy only requirements.txt first to leverage Docker layer caching.
# -----------------------------------------------------------------------------
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Copy the full project into the image
#   This includes:
#     - app.py, db.py, worker.py, producer.py
#     - models/*.py, eval/train scripts
#     - loan_data.csv, model.pkl, schema.json (if present in project root)
# -----------------------------------------------------------------------------
COPY . .

# -----------------------------------------------------------------------------
# Ensure expected folders exist and move model/data files into them.
#   - data/loan_data.csv     : streaming dataset for the producer
#   - models/model.pkl       : trained sklearn Pipeline
#   - models/schema.json     : schema with feature/target columns
# -----------------------------------------------------------------------------
RUN mkdir -p data models && \
    if [ -f "loan_data.csv" ]; then cp loan_data.csv data/loan_data.csv; fi && \
    if [ -f "model.pkl" ]; then cp model.pkl models/model.pkl; fi && \
    if [ -f "schema.json" ]; then cp schema.json models/schema.json; fi

# -----------------------------------------------------------------------------
# Default environment configuration (can be overridden by docker-compose/.env)
# -----------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    DATA_CSV_PATH=data/loan_data.csv \
    SCHEMA_PATH=models/schema.json

# -----------------------------------------------------------------------------
# Default command:
#   For containers that use this image directly (not overridden by compose),
#   we start the Flask app. In docker-compose.yaml we override the command
#   for celery_worker and producer services.
# -----------------------------------------------------------------------------
CMD ["python", "app.py"]
