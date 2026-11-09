# ─────────────────────────────────────────────────────────────────────────────
#  Dockerfile — AEPM Production Container
#  Aircraft Engine Predictive Maintenance using Machine Learning
#
#  Build:
#      docker build -t aepm:latest .
#
#  Run:
#      docker run -p 8501:8501 aepm:latest
#
#  Then open:  http://localhost:8501
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="AEPM Project Team"
LABEL description="Aircraft Engine Predictive Maintenance — RUL Prediction Dashboard"
LABEL version="1.0.0"

# ── Environment Configuration ─────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── System Dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python Dependencies ───────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy Project Source ───────────────────────────────────────────────────────
COPY . .

# ── Expose Dashboard Port ─────────────────────────────────────────────────────
EXPOSE 8501

# ── Health Check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
    || exit 1

# ── Launch Streamlit Dashboard ────────────────────────────────────────────────
CMD ["python", "-m", "streamlit", "run", "app/dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false"]
