from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Heart Disease Prediction API")

# Initialize Prometheus Metrics (Step 8 requirement)
Instrumentator().instrument(app).expose(app)

# Load Model
model = joblib.load("models/best_model.pkl")

class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert input to array
        input_data = np.array([[
            data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs, 
            data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, 
            data.ca, data.thal
        ]])
        
        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        logger.info(f"Prediction made: {prediction} with probability {probability}")
        
        return {
            "prediction": int(prediction),
            "disease_present": bool(prediction),
            "confidence": float(probability)
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))