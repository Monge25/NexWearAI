"""
Endpoint de Smart Personalized Discounts.

Predice qué tipo de promoción (Discount, FreeShipping, NoPromotion) conviene
ofrecer a un usuario según sus métricas RFM (recencia, frecuencia, valor
promedio de orden), usando el modelo RandomForestClassifier entrenado en
training/train_discount.py.
"""

from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
import os

router = APIRouter()

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "discount_model.pkl")
model = joblib.load(MODEL_PATH)

@router.get("/{user_id}")
def predict_discount(user_id: str, recency_days: int, frequency: int, avg_order_value: float, cart_abandon_rate: float = 0.0):
    """Devuelve la promoción recomendada para un usuario y la confianza del modelo."""
    if recency_days < 0 or frequency < 0 or avg_order_value < 0:
        raise HTTPException(status_code=400, detail="recency_days, frequency y avg_order_value no pueden ser negativos")
    if not 0.0 <= cart_abandon_rate <= 1.0:
        raise HTTPException(status_code=400, detail="cart_abandon_rate debe estar entre 0.0 y 1.0")

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