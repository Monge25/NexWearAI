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
    n = name.lower()
    for kw in TOP_KEYWORDS:
        if kw in n:
            return "top"
    for kw in BOTTOM_KEYWORDS:
        if kw in n:
            return "bottom"
    return "other"


def pick_best_within_budget(candidates: pd.DataFrame, slot_budget: float):
    if candidates.empty:
        return None
    affordable = candidates[candidates["Price"] <= slot_budget]
    if not affordable.empty:
        # el más caro que SÍ quepa en el presupuesto de ese slot -> aprovecha el presupuesto
        return affordable.sort_values("Price", ascending=False).iloc[0]
    # si nada cabe, regresa el más barato disponible de todos modos
    return candidates.sort_values("Price", ascending=True).iloc[0]


@router.get("/{user_id}")
def get_outfit_recommendation(user_id: str, avg_budget: float, preferred_category: str = "mujer"):
    cat = preferred_category.strip().lower()
    if cat not in ("mujer", "hombre"):
        cat = "mujer"

    # 1. Predecir el "estilo" solo como etiqueta descriptiva (usa el modelo ya entrenado)
    cat_col = f"cat_{cat}"
    style_name = "Personalizado"
    if cat_col in category_columns:
        cat_vector = [1 if col == cat_col else 0 for col in category_columns]
        budget_scaled = scaler.transform(pd.DataFrame([[avg_budget]], columns=["Price"]))[0][0]
        features = np.array(cat_vector + [budget_scaled]).reshape(1, -1)
        predicted_cluster = int(mlp_model.predict(features)[0])
        style_name = style_names.get(str(predicted_cluster), style_names.get(predicted_cluster, "Personalizado"))

# 2. Clasificar productos del género elegido por tipo de prenda
    gender_products = products[products["Category"].str.lower() == cat].copy()
    gender_products["slot"] = gender_products["Name"].apply(infer_slot)

    tops = gender_products[gender_products["slot"] == "top"]
    bottoms = gender_products[gender_products["slot"] == "bottom"]
    accessories = products[products["Category"].str.lower() == "accesorios"]

    # 3. Repartir el presupuesto: 45% prenda superior, 45% inferior, 20% accesorio
# 3. Repartir el presupuesto: 45% prenda superior, 45% inferior, 20% accesorio
    print(f"\n\n>>> DEBUG VERSION-2 <<< avg_budget={avg_budget}, cat={cat}")
    print(f">>> tops disponibles: {len(tops)}, bottoms disponibles: {len(bottoms)}")
    top_item = pick_best_within_budget(tops, avg_budget * 0.45)
    bottom_item = pick_best_within_budget(bottoms, avg_budget * 0.45)
    accessory_item = pick_best_within_budget(accessories, avg_budget * 0.20)
    print(f">>> top elegido: {top_item['Name'] if top_item is not None else None} (${top_item['Price'] if top_item is not None else 0})")

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