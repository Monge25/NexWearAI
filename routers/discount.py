from fastapi import APIRouter
import joblib
import pandas as pd
import os

router = APIRouter()

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "discount_model.pkl")
model = joblib.load(MODEL_PATH)

@router.get("/{user_id}")
def predict_discount(user_id: str, recency_days: int, frequency: int, avg_order_value: float, cart_abandon_rate: float = 0.0):
    X = pd.DataFrame(
        [[recency_days, frequency, avg_order_value, cart_abandon_rate]],
        columns=["recency_days", "frequency", "avg_order_value", "cart_abandon_rate"]
    )
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(max(probabilities))

    return {
        "userId": user_id,
        "recommendedPromotion": prediction,
        "confidence": round(confidence, 4)
    }