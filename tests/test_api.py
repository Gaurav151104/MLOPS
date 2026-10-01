from fastapi.testclient import TestClient

from app.main import app, get_model


class DummyModel:
    def predict(self, X):
        return [75.456]


app.dependency_overrides[get_model] = lambda: DummyModel()
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_prediction():
    r = client.post("/predict", json={
        "area_sqft": 1500,
        "bedrooms": 3,
        "age_years": 5,
        "distance_km": 4,
    })
    assert r.status_code == 200
    assert r.json() == {"predicted_price_lakh": 75.46}