from fastapi import APIRouter
import joblib
import pandas as pd
import os

router = APIRouter()

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "demand_model.pkl")
model = joblib.load(MODEL_PATH)

@router.get("/{product_id}")
def predict_demand(product_id: str, week: int, month: int, on_sale: bool = False):
    X = pd.DataFrame([[week, month, int(on_sale)]], columns=["week", "month", "on_sale"])
    pred = model.predict(X)[0]
    return {
        "productId": product_id,
        "week": week,
        "month": month,
        "onSale": on_sale,
        "predictedUnits": round(float(pred), 1)
    }