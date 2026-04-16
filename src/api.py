from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import date

# Load model on startup
def load_model(path="src/data/model.pkl"):
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