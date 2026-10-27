@echo off
python -m src.modeling
python -m src.monitoring
python predict_batch.py
python -m uvicorn api:app --reload
