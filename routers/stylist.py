"""
Endpoint de Fashion Stylist.

Asigna a cada usuario un "estilo" (Casual/Clasico/Urbano/Elegante) usando
K-Means sobre categoría + precio de los productos (entrenado en
training/train_stylist.py), y arma un outfit sugerido (top + bottom +
accesorio) repartiendo el presupuesto real del usuario (45%/45%/20%).
El nombre del estilo es solo descriptivo; el outfit se arma con reglas
de negocio sobre el catálogo real, no directamente del cluster.
"""

from fastapi import APIRouter
import joblib
import pandas as pd
import numpy as np
import os

router = APIRouter()

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "stylist_model.pkl")
data = joblib.load(MODEL_PATH)

mlp_model = data["mlp_model"]
scaler = data["scaler"]
category_columns = data["category_columns"]
style_names = data["style_names"]
products = pd.DataFrame(data["products"])

TOP_KEYWORDS = ["camisa", "blusa", "playera", "camiseta", "polo", "top", "sueter", "suéter", "sudadera", "cardigan"]
BOTTOM_KEYWORDS = ["pantalón", "pantalon", "short", "falda", "jean", "palazzo"]


def infer_slot(name: str) -> str:
    """Clasifica un producto como prenda superior, inferior u otro, por palabras clave del nombre."""
    n = name.lower()
    for kw in TOP_KEYWORDS:
        if kw in n:
            return "top"
    for kw in BOTTOM_KEYWORDS:
        if kw in n:
            return "bottom"
    return "other"


def pick_best_within_budget(candidates: pd.DataFrame, slot_budget: float):
    """Elige el producto más caro que quepa en el presupuesto del slot; si nada cabe, el más barato disponible."""
    if candidates.empty:
        return None
    affordable = candidates[candidates["Price"] <= slot_budget]
    if not affordable.empty:
        return affordable.sort_values("Price", ascending=False).iloc[0]
    return candidates.sort_values("Price", ascending=True).iloc[0]


@router.get("/{user_id}")
def get_outfit_recommendation(user_id: str, avg_budget: float, preferred_category: str = "mujer"):
    """Devuelve un outfit (top + bottom + accesorio) ajustado al presupuesto y categoría preferida del usuario."""
    cat = preferred_category.strip().lower()
    if cat not in ("mujer", "hombre"):
        cat = "mujer"

    cat_col = f"cat_{cat}"
    style_name = "Personalizado"
    if cat_col in category_columns:
        cat_vector = [1 if col == cat_col else 0 for col in category_columns]
        budget_scaled = scaler.transform(pd.DataFrame([[avg_budget]], columns=["Price"]))[0][0]
        features = np.array(cat_vector + [budget_scaled]).reshape(1, -1)
        predicted_cluster = int(mlp_model.predict(features)[0])
        style_name = style_names.get(str(predicted_cluster), style_names.get(predicted_cluster, "Personalizado"))

    gender_products = products[products["Category"].str.lower() == cat].copy()
    gender_products["slot"] = gender_products["Name"].apply(infer_slot)

    tops = gender_products[gender_products["slot"] == "top"]
    bottoms = gender_products[gender_products["slot"] == "bottom"]
    accessories = products[products["Category"].str.lower() == "accesorios"]

    top_item = pick_best_within_budget(tops, avg_budget * 0.45)
    bottom_item = pick_best_within_budget(bottoms, avg_budget * 0.45)
    accessory_item = pick_best_within_budget(accessories, avg_budget * 0.20)

    outfit = []
    for item in [top_item, bottom_item, accessory_item]:
        if item is not None:
            image_url = item["ImageUrl"]
            outfit.append({
                "productId": str(item["Id"]),
                "name": item["Name"],
                "category": item["Category"],
                "price": float(item["Price"]),
                "imageUrl": None if pd.isna(image_url) else image_url,
            })

    return {
        "userId": user_id,
        "styleName": style_name,
        "outfit": outfit
    }