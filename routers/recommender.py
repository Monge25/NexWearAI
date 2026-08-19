from fastapi import APIRouter, HTTPException
import joblib
import os

router = APIRouter()

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "recommender_model.pkl")
data = joblib.load(MODEL_PATH)

product_ids = data["product_ids"]
product_names = data["product_names"]
similarity_matrix = data["similarity_matrix"]

@router.get("/{product_id}")
def get_recommendations(product_id: str, top_k: int = 4):
    if product_id not in product_ids:
        raise HTTPException(status_code=404, detail="Producto no encontrado en el modelo de recomendaciones")

    idx = product_ids.index(product_id)
    sims = similarity_matrix[idx]

    # ordenar por similitud descendente, excluyendo el mismo producto
    ranked = sorted(
        [(i, s) for i, s in enumerate(sims) if i != idx],
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    results = [
        {"productId": product_ids[i], "name": product_names[i], "similarity": round(float(s), 4)}
        for i, s in ranked
    ]
    return {"results": results}