# ================================================================================
# Project:      PlasmaCast — Plasma Donor Demand Forecasting
# Contributor:  Puranjay Wadhera (GitHub: @PW13052003)
# File:         api.py
# Purpose:      Serves the trained XGBoost model as a REST API using FastAPI.
#               Exposes two endpoints: a health check and a prediction endpoint
#               that accepts donor center details and weather inputs and returns
#               a predicted daily donor count. This allows any external
#               application (dashboard, mobile app, etc.) to query the model
#               without needing Python or ML knowledge.
# Language:     Python 3.12
# Libraries:    joblib, pandas, numpy, os
# Frameworks:   FastAPI, Pydantic, Uvicorn
# APIs:         None
# GitHub:       https://github.com/PW13052003/plasmacast/blob/main/src/api.py
# ================================================================================


# Import the necessary modules
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import date
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BASE = pathlib.Path(__file__).parent.parent

def load_model(path=os.path.join(DATA_DIR, "model.pkl")):
    """
        Loads the trained XGBoost model and its feature list from disk.
        Called once at API startup — loading is slow, but prediction is fast.
        Loading once at startup rather than per request keeps the API responsive.

        Args:
            path (str): Path to the saved .pkl model file.

        Returns:
            model: The trained XGBoost model.
            features (list): The ordered list of features the model expects.
    """
    data = joblib.load(path)
    return data["model"], data["features"]

model, features = load_model()

# Initialize FastAPI app
app = FastAPI(
    title="PlasmaCast API",
    description="Donor demand forecasting API for plasma donation centers",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=BASE / "dashboard"), name="static")

@app.get("/")
def serve_dashboard():
    return FileResponse(BASE / "dashboard" / "dashboard.html")

# Health check endpoint
@app.get("/health")
def health():
    """
        Health check endpoint. Returns API status and confirms the model
        is loaded and ready to serve predictions.

        Args:
            NONE

        Returns:
            dict: API status, model load status, and list of active features.
    """
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
    """
        Main prediction endpoint. Accepts center and weather information,
        engineers all required calendar features from the date, assembles
        the feature vector, and returns a predicted donor count.

        Args:
            request (PredictRequest): Validated request body.

        Returns:
            dict: center_id, date, and predicted_donors (integer).
    """
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