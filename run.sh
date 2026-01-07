#!/bin/bash

# Resolve project directory automatically
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# Run Streamlit in background
nohup "$PROJECT_DIR/.venv/bin/streamlit" run "$PROJECT_DIR/app/main.py" \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    > "$PROJECT_DIR/streamlit.log" 2>&1 &

echo "🚀 Crypto Quant Dashboard running in background on port 8501"
