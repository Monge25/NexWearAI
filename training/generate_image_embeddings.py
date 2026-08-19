"""
Genera embeddings CLIP para todos los productos activos con imagen,
usados por routers/image_search.py para calcular similitud visual.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import open_clip
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import json
from db import engine
from sqlalchemy import text

print("Cargando modelo CLIP...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model.eval()


def fetch_products_with_image(engine) -> list:
    """Trae de la base los productos activos que sí tienen ImageUrl."""
    with engine.connect() as conn:
        return list(conn.execute(text("""
            SELECT "Id", "Name", "ImageUrl"
            FROM "Products"
            WHERE "IsActive" = true AND "ImageUrl" IS NOT NULL AND "ImageUrl" != ''
        """)))


products: list = fetch_products_with_image(engine)
print(f"Productos con imagen encontrados: {len(products)}")

embeddings: dict = {}
for row in products:
    product_id: str = str(row[0])
    name: str = row[1]
    image_url: str = row[2]
    try:
        response = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_input = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            embedding = model.encode_image(img_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        embeddings[product_id] = {
            "name": name,
            "vector": embedding[0].tolist()
        }
        print(f"  OK: {name}")
    except Exception as e:
        print(f"  ERROR con {name}: {e}")

with open("models/product_embeddings.json", "w") as f:
    json.dump(embeddings, f)

print(f"\nEmbeddings guardados: {len(embeddings)} productos en models/product_embeddings.json")