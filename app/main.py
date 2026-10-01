import os
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.pkl"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

app = FastAPI(title="MLOps House Price Prediction API", version="1.0.0")


@lru_cache
def get_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model artifact not available")


class HouseFeatures(BaseModel):
    area_sqft: float
    bedrooms: int
    age_years: float
    distance_km: float


@app.get("/")
def home():
    return {"message": "MLOps House Price Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(features: HouseFeatures, model=Depends(get_model)):
    values = [[features.area_sqft, features.bedrooms, features.age_years, features.distance_km]]
    prediction = float(model.predict(values)[0])
    return {"predicted_price_lakh": round(prediction, 2)}