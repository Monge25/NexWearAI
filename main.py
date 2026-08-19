"""
NexWearAI — Microservicio de Inteligencia Artificial
=======================================================
API REST (FastAPI) que expone los 5 modelos de machine learning entrenados
para NexWear. Cada modelo vive en su propio router y se sirve como un
endpoint independiente, consumido por el backend principal (NexWearAPI, .NET)
vía HTTP.

Modelos servidos:
- /predict/demand      → Demand Forecast AI (regresión)
- /predict/pricing     → Fashion Pricing Intelligence (regresión)
- /search/image        → Búsqueda de productos por imagen (CLIP, zero-shot)
- /recommend           → Recomendador de productos (similitud por contenido)
- /predict/discount    → Smart Personalized Discounts (clasificación)
- /detect/fake-review  → Detección de reseñas falsas (clasificación de texto)
- /stylist             → Fashion Stylist (clustering + red neuronal)
"""

from fastapi import FastAPI
from routers import demand, pricing, image_search, recommender, discount, review_detector, stylist

app = FastAPI(title="NexWearAI")

app.include_router(demand.router, prefix="/predict/demand")
app.include_router(pricing.router, prefix="/predict/pricing")
app.include_router(image_search.router, prefix="/search/image")
app.include_router(recommender.router, prefix="/recommend")
app.include_router(discount.router, prefix="/predict/discount")
app.include_router(review_detector.router, prefix="/detect/fake-review")
app.include_router(stylist.router, prefix="/stylist")


@app.get("/health")
def health():
    """Endpoint de salud, usado para confirmar que el servicio está arriba."""
    return {"status": "ok"}