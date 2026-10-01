from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load(Path("model/model.pkl"))
app = FastAPI(title="MLOps House Price Prediction API", version="1.0.0")

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
def predict(features: HouseFeatures):
    values = [[features.area_sqft, features.bedrooms, features.age_years, features.distance_km]]
    prediction = float(model.predict(values)[0])
    return {"predicted_price_lakh": round(prediction, 2)}
