from fastapi import APIRouter, UploadFile, File
import torch
import open_clip
from PIL import Image
from io import BytesIO
import numpy as np
import json
import os
import re

router = APIRouter()

print("Cargando modelo CLIP para búsqueda por imagen...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "product_embeddings.json"
)

# Tipos de prenda con prompt en inglés (para CLIP) + palabras clave en español (para detectar en el nombre del producto)
GARMENT_TYPES = {
    "pants":    {"prompt": "a photo of pants or trousers", "keywords": ["pantalón", "pantalon", "short", "palazzo"]},
    "jacket":   {"prompt": "a photo of a jacket or coat or blazer or cardigan", "keywords": ["chaqueta", "chamarra", "abrigo", "blazer", "gabardina", "trench", "parka", "cardigan", "chaleco"]},
    "shirt":    {"prompt": "a photo of a shirt or blouse or polo or t-shirt", "keywords": ["camisa", "playera", "camiseta", "polo", "top", "blusa", "sudadera", "suéter", "sueter"]},
    "dress":    {"prompt": "a photo of a dress", "keywords": ["vestido"]},
    "skirt":    {"prompt": "a photo of a skirt", "keywords": ["falda"]},
    "accessory":{"prompt": "a photo of a bag, hat, watch, belt or backpack", "keywords": ["bolso", "mochila", "sombrero", "reloj", "cinturón", "cinturon"]},
}

with torch.no_grad():
    garment_names = list(GARMENT_TYPES.keys())
    garment_tokens = tokenizer([GARMENT_TYPES[g]["prompt"] for g in garment_names])
    garment_text_features = model.encode_text(garment_tokens)
    garment_text_features = garment_text_features / garment_text_features.norm(dim=-1, keepdim=True)

def load_embeddings():
    with open(EMBEDDINGS_PATH, "r") as f:
        return json.load(f)

def infer_garment_type(product_name: str) -> str:
    name_lower = product_name.lower()
    for garment, info in GARMENT_TYPES.items():
        for kw in info["keywords"]:
            if kw in name_lower:
                return garment
    return "other"

@router.post("/")
async def search_by_image(file: UploadFile = File(...), top_k: int = 5):
    embeddings = load_embeddings()

    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    img_input = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        query_embedding = model.encode_image(img_input)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

        garment_similarities = (query_embedding @ garment_text_features.T)[0]
        predicted_garment = garment_names[int(garment_similarities.argmax())]

    query_vector = query_embedding[0].numpy()

    results = []
    for product_id, data in embeddings.items():
        product_vector = np.array(data["vector"])
        visual_similarity = float(np.dot(query_vector, product_vector))
        product_garment = infer_garment_type(data["name"])

        results.append({
            "productId": product_id,
            "name": data["name"],
            "similarity": round(visual_similarity, 4),
            "_same_garment": product_garment == predicted_garment,
        })

    # Priorizar SIEMPRE mismo tipo de prenda; dentro de cada grupo, ordenar por similitud visual
    results.sort(key=lambda x: (x["_same_garment"], x["similarity"]), reverse=True)
    for r in results:
        del r["_same_garment"]

    return {"results": results[:top_k], "detectedGarment": predicted_garment}