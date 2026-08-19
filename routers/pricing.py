"""
Endpoint de Fashion Pricing Intelligence.

Sugiere un ajuste de precio (%) para una variante de producto según su
stock, ventas recientes y agregados al carrito, usando el modelo
GradientBoostingRegressor entrenado en training/train_pricing.py.
"""

from fastapi import APIRouter
import joblib
import pandas as pd
import os

router = APIRouter()

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "pricing_model.pkl")
model = joblib.load(MODEL_PATH)

UMBRAL_NOTIFICACION = 8.0  # % de cambio a partir del cual se avisa al admin

@router.get("/{variant_id}")
def predict_pricing(variant_id: str, stock: int, units_sold_30d: int, cart_adds: int, current_price: float):
    """Devuelve el precio sugerido y si el cambio amerita notificar al admin."""
    X = pd.DataFrame(
        [[stock, units_sold_30d, cart_adds, current_price]],
        columns=["stock", "units_sold_30d", "cart_adds", "current_price"]
    )
    delta_pct = float(model.predict(X)[0])
    suggested_price = round(current_price * (1 + delta_pct / 100), 2)

    return {
        "variantId": variant_id,
        "currentPrice": current_price,
        "suggestedDeltaPct": round(delta_pct, 2),
        "suggestedPrice": suggested_price,
        "notify": abs(delta_pct) >= UMBRAL_NOTIFICACION
    }