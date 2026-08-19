import open_clip
import torch

print("Descargando modelo CLIP (puede tardar unos minutos la primera vez)...")

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

print("Modelo cargado correctamente.")
print("Dispositivo disponible:", "GPU" if torch.cuda.is_available() else "CPU")