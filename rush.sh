#!/bin/bash

# Activate venv
source venv/bin/activate

# Launch Streamlit in background with nohup
nohup streamlit run app/main.py --server.address=0.0.0.0 --server.port=8501 > streamlit.log 2>&1 &

echo "🚀 Streamlit dashboard is now running in the background!"
