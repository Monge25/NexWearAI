"""
Script de verificación — CLIP.
Confirma que el modelo CLIP (ViT-B-32) se descarga y carga correctamente,
y si hay GPU disponible para acelerar inferencia. Se corre una sola vez
antes de generar embeddings o servir búsqueda por imagen.
"""

import open_clip
import torch

print("Descargando modelo CLIP (puede tardar unos minutos la primera vez)...")

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

print("Modelo cargado correctamente.")
print("Dispositivo disponible:", "GPU" if torch.cuda.is_available() else "CPU")