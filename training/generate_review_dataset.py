import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from db import engine
from sqlalchemy import text

np.random.seed(42)

# 1. Traer reseñas reales (para probarlas después, no para entrenar con ellas como fake)
with engine.connect() as conn:
    real_reviews = pd.read_sql(text('''
        SELECT "Id", "Rating", "Comment"
        FROM "Reviews"
    '''), conn)

print(f"Reseñas reales encontradas: {len(real_reviews)}")

# 2. Plantillas de reseñas GENUINAS (detalladas, específicas, variadas)
genuine_templates = [
    "La tela es de muy buena calidad, me quedó justo en talla {size}. El color es igual al de las fotos.",
    "Lo compré para un evento y quedé encantada, aunque la manga es un poco larga para mi gusto.",
    "Buen producto pero tardó más de lo esperado en llegar. La calidad sí es buena.",
    "Es mi segunda compra de esta marca, siempre cumplen con la calidad prometida.",
    "El material se siente premium, pero el precio me parece un poco elevado para lo que es.",
    "Me encantó el corte, se ajusta bien al cuerpo sin quedar apretado.",
    "La talla me quedó un poco grande, tuve que hacer un ajuste con la costurera.",
    "Excelente para el trabajo, versátil y cómodo durante todo el día.",
    "El color en persona es un poco más oscuro que en la foto, pero aun así me gustó.",
    "Se lava fácil y no ha perdido color después de varios lavados.",
]

# 3. Plantillas de reseñas FALSAS/SPAM (genéricas, exageradas, repetitivas, promocionales)
fake_templates = [
    "Excelente!!! Lo mejor que he comprado en mi vida!!! 100% recomendado!!!",
    "Buen producto buen producto buen producto",
    "Compra ya este producto es increible no te vas a arrepentir super recomendado",
    "5 estrellas al mejor precio visita mi perfil para mas descuentos",
    "wow amazing product best quality ever 10/10 recommend",
    "Todo perfecto perfecto perfecto gracias gracias gracias",
    "El mejor vendedor de todos compren aqui rapido antes que se agote",
    "Buenisimo buenisimo buenisimo lo super recomiendo a todos",
    "Como siempre excelente calidad,rapido,recomendado,super,excelente",
    "Genial genial genial 100 estrellas de 5",
]

rows = []

# Genuinas: variaciones con ruido
sizes = ["S", "M", "L", "XL"]
for i in range(150):
    template = np.random.choice(genuine_templates)
    text_val = template.format(size=np.random.choice(sizes)) if "{size}" in template else template
    rows.append({
        "rating": np.random.randint(3, 6),
        "comment": text_val,
        "comment_length": len(text_val),
        "exclamation_count": text_val.count("!"),
        "is_fake": 0
    })

# Falsas: repetición de plantillas cortas con ruido de puntuación
for i in range(150):
    template = np.random.choice(fake_templates)
    rows.append({
        "rating": np.random.choice([1, 5], p=[0.2, 0.8]),  # extremos, típico de spam
        "comment": template,
        "comment_length": len(template),
        "exclamation_count": template.count("!"),
        "is_fake": 1
    })

df = pd.DataFrame(rows)
df.to_csv("training/review_dataset.csv", index=False)
print(f"Dataset final: {len(df)} filas")
print(df["is_fake"].value_counts())