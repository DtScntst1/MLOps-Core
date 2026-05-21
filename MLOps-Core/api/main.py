from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(
    title="MLOps-Core Predictor API",
    description="A production-ready REST API for Breast Cancer Prediction.",
    version="1.0.0"
)

# Global variable to hold the model
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Warning: Model not found. The API will start, but predictions will fail until a model is trained.")

class PredictionRequest(BaseModel):
    # Breast cancer dataset has 30 features. For simplicity in the API, we expect a list of 30 floats.
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
    
    if len(request.features) != 30:
        raise HTTPException(status_code=400, detail="Expected 30 features for prediction.")
    
    # Convert list to numpy array and reshape for single prediction
    input_data = np.array(request.features).reshape(1, -1)
    
    prediction = model.predict(input_data)[0]
    # Get probability for the predicted class
    prob = model.predict_proba(input_data)[0][prediction]
    
    return PredictionResponse(
        prediction=int(prediction),
        probability=float(prob)
    )
