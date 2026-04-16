from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import date
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Load model on startup
def load_model(path=os.path.join(DATA_DIR, "model.pkl")):
    data = joblib.load(path)
    return data["model"], data["features"]

model, features = load_model()

# Initialize FastAPI app
app = FastAPI(
    title="PlasmaCast API",
    description="Donor demand forecasting API for plasma donation centers",
    version="1.0.0"
)

# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "features": features
    }

class PredictRequest(BaseModel):
    center_id: str
    date: date
    temp_max: float
    precipitation: float
    donor_lag_7: float
    donor_lag_14: float
    rolling_7day_avg: float
    rolling_14day_avg: float

@app.post("/predict")
def predict(request: PredictRequest):
    input_date = pd.Timestamp(request.date)

    input_data = {
        "temp_max": request.temp_max,
        "precipitation": request.precipitation,
        "day_of_week": input_date.dayofweek,
        "is_holiday": int(input_date in __import__("holidays").US()),
        "month": input_date.month,
        "year": input_date.year,
        "day_of_year": input_date.dayofyear,
        "is_weekend": int(input_date.dayofweek >= 5),
        "season": {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[input_date.month],
        "donor_lag_7": request.donor_lag_7,
        "donor_lag_14": request.donor_lag_14,
        "rolling_7day_avg": request.rolling_7day_avg,
        "rolling_14day_avg": request.rolling_14day_avg,
    }

    input_df = pd.DataFrame([input_data])[features]
    prediction = model.predict(input_df)[0]

    return {
        "center_id": request.center_id,
        "date": str(request.date),
        "predicted_donors": int(prediction),
    }