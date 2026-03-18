FROM python:3.12-slim
# Cache bust: 2026-03-12-v2

# System deps for psycopg2 and ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
# requirements-api.txt contains only what the API needs (no ML training libs)
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY . .

EXPOSE 8000

# Start the FastAPI app — Railway injects PORT at runtime
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
