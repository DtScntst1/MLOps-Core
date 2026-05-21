from fastapi.testclient import TestClient
import sys
import os

# Ensure the root directory is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint_invalid_features():
    # Sending fewer than 30 features should raise a 400 error
    response = client.post("/predict", json={"features": [0.1, 0.2, 0.3]})
    assert response.status_code in [400, 422, 503] # 503 if model not loaded, 422 for pydantic, 400 for logic

def test_predict_endpoint_valid_features():
    # Simulate 30 dummy features
    dummy_features = [0.5] * 30
    response = client.post("/predict", json={"features": dummy_features})
    
    # If model is loaded, it should return 200
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
    else:
        # If model is not loaded (e.g. before training), it returns 503
        assert response.status_code == 503
