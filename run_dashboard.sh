#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  run_dashboard.sh — AEPM Streamlit Dashboard Launcher (Unix / macOS)
#  Aircraft Engine Predictive Maintenance using Machine Learning
#
#  Usage:
#      chmod +x run_dashboard.sh   # (first time only)
#      ./run_dashboard.sh
#
#  Then open your browser at:  http://localhost:8501
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================"
echo "  ✈️  AEPM Streamlit Dashboard"
echo "  Aircraft Engine Predictive Maintenance"
echo "  Dashboard  |  http://localhost:8501"
echo "======================================================"

# ── Activate virtual environment (if present) ─────────────────────────────────
VENV_ACTIVATE=".venv/bin/activate"
if [ -f "$VENV_ACTIVATE" ]; then
    echo "[INFO] Activating virtual environment: .venv"
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
else
    echo "[WARN] No .venv found. Using system Python."
fi

# ── Verify Streamlit is available ─────────────────────────────────────────────
if ! python -c "import streamlit" &>/dev/null; then
    echo "[ERROR] Streamlit is not installed. Run:"
    echo "        pip install -r requirements.txt"
    exit 1
fi

# ── Launch Dashboard ──────────────────────────────────────────────────────────
echo "[INFO] Starting Streamlit dashboard..."
python -m streamlit run app/dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
