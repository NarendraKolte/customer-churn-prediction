from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

print("FastAPI app loaded")

# =============================
# Load model & encoder safely
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "churn_model.pkl"))
segment_encoder = joblib.load(os.path.join(BASE_DIR, "models", "segment_encoder.pkl"))

# =============================
# Initialize FastAPI
# =============================
app = FastAPI(title="Customer Churn Prediction API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json")

# =============================
# Business Action Engine
# =============================
def retention_action(churn_prob: float):
    if churn_prob >= 0.75:
        return "Offer 25% discount + retention call"
    elif churn_prob >= 0.5:
        return "Send personalized email offer"
    else:
        return "No action needed"

# =============================
# Input Schema (Single Prediction)
# =============================
class CustomerInput(BaseModel):
    frequency: int
    monetary: float
    segment: str

# =============================
# Single Prediction Endpoint
# =============================
@app.post("/predict_churn")
def predict_churn(data: CustomerInput):
    # Encode segment
    segment_encoded = segment_encoder.transform([data.segment])[0]

    # Prepare input
    X = np.array([[data.frequency, data.monetary, segment_encoded]])

    # Predict
    churn_prob = model.predict_proba(X)[0][1]

    return {
        "churn_probability": round(float(churn_prob), 3),
        "churn_risk": "High" if churn_prob > 0.6 else "Low",
        "recommended_action": retention_action(churn_prob)
    }

# =============================
# Batch Prediction Endpoint
# =============================
@app.post("/predict_churn_batch")
async def predict_churn_batch(file: UploadFile = File(...)):
    # Read CSV
    df = pd.read_csv(file.file)

    # Encode segment
    df["segment_encoded"] = segment_encoder.transform(df["segment"])

    # Prepare features
    X = df[["frequency", "monetary", "segment_encoded"]]

    # Predict
    churn_probs = model.predict_proba(X)[:, 1]

    # Add outputs
    df["churn_probability"] = churn_probs.round(3)
    df["churn_risk"] = df["churn_probability"].apply(
        lambda x: "High" if x > 0.6 else "Low"
    )
    df["recommended_action"] = df["churn_probability"].apply(retention_action)

    return df.to_dict(orient="records")

# =============================
# Root Endpoint (Optional)
# =============================
@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}
