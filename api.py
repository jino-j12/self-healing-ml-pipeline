from fastapi import FastAPI
import joblib
import os
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API")


# -------------------------
# Load Latest Model
# -------------------------

models = sorted(os.listdir("models"))
latest_model = f"models/{models[-1]}"

model = joblib.load(latest_model)


# -------------------------
# Input Schema
# -------------------------

class Transaction(BaseModel):
    features: list


# -------------------------
# Routes
# -------------------------

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}


@app.post("/predict")
def predict(transaction: Transaction):

    data = np.array(transaction.features).reshape(1, -1)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        result = "Fraud"
    else:
        result = "Normal"

    return {
        "prediction": result,
        "fraud_probability": float(probability)
    }