@echo off
echo Starting AEPM Streamlit Dashboard...
echo ======================================
echo  Aircraft Engine Predictive Maintenance
echo  Dashboard  ^|  http://localhost:8501
echo ======================================
call .venv\Scripts\activate
.venv\Scripts\python.exe -m streamlit run app/dashboard.py --server.port 8501
