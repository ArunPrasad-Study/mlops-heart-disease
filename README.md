# Heart Disease MLOps Pipeline

## Overview
An end-to-end MLOps pipeline for predicting heart disease using the UCI Heart Disease dataset. Includes data preprocessing, model training with MLflow, CI/CD via GitHub Actions, Docker containerization, and Kubernetes deployment.

## Setup & Installation
1. Clone the repository: `git clone https://github.com/ArunPrasad-Study/mlops-heart-disease`
2. Install dependencies: `pip install -r requirements.txt`
3. Download and clean data: `python download_data.py`
4. Train model: `python src/train.py`

## API Usage
Run locally: `uvicorn src.app:app --reload`
Test endpoint: `curl -X POST "http://127.0.0.1:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"age":50,"sex":1,"cp":2,"trestbps":130,"chol":200,"fbs":0,"restecg":1,"thalach":140,"exang":0,"oldpeak":1.5,"slope":2,"ca":0,"thal":2}'`

## Docker & K8s
- Build: `docker build -t heart-disease-api .`
- Deploy: `kubectl apply -f k8s/`
