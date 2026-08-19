import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from db import engine
from sqlalchemy import text

np.random.seed(42)

# 1. Traer variantes reales
with engine.connect() as conn:
    variants = pd.read_sql(text("""
        SELECT pv."Id" as variant_id, pv."Stock", pv."PriceModifier",
               p."Price" as base_price
        FROM "ProductVariants" pv
        JOIN "Products" p ON p."Id" = pv."ProductId"
        WHERE pv."IsActive" = true
    """), conn)

print(f"Variantes reales encontradas: {len(variants)}")

# 2. Simular métricas de ventas/interés por variante (30 días) + generar el target con reglas de negocio
rows = []
for _, v in variants.iterrows():
    stock = int(v["Stock"]) if v["Stock"] else np.random.randint(0, 50)
    units_sold_30d = np.random.randint(0, 20)
    cart_adds = np.random.randint(0, 15)
    current_price = float(v["base_price"]) + float(v["PriceModifier"] or 0)

    # Regla de negocio para generar el "target" (delta de precio sugerido en %)
    demand_score = units_sold_30d + cart_adds * 0.5
    if stock < 5 and demand_score > 10:
        delta_pct = np.random.uniform(8, 15)      # poco stock + alta demanda -> subir precio
    elif stock > 30 and demand_score < 5:
        delta_pct = np.random.uniform(-20, -8)    # mucho stock + baja demanda -> bajar precio
    else:
        delta_pct = np.random.uniform(-3, 3)       # sin cambios relevantes

    rows.append({
        "variant_id": v["variant_id"],
        "stock": stock,
        "units_sold_30d": units_sold_30d,
        "cart_adds": cart_adds,
        "current_price": current_price,
        "delta_pct": round(delta_pct, 2),
    })

# 3. Generar variantes sintéticas adicionales (para tener volumen de entrenamiento)
for i in range(300):
    stock = np.random.randint(0, 60)
    units_sold_30d = np.random.randint(0, 25)
    cart_adds = np.random.randint(0, 20)
    current_price = np.random.uniform(200, 2000)

    demand_score = units_sold_30d + cart_adds * 0.5
    if stock < 5 and demand_score > 10:
        delta_pct = np.random.uniform(8, 15)
    elif stock > 30 and demand_score < 5:
        delta_pct = np.random.uniform(-20, -8)
    else:
        delta_pct = np.random.uniform(-3, 3)

    rows.append({
        "variant_id": f"synthetic-{i}",
        "stock": stock,
        "units_sold_30d": units_sold_30d,
        "cart_adds": cart_adds,
        "current_price": round(current_price, 2),
        "delta_pct": round(delta_pct, 2),
    })

df = pd.DataFrame(rows)
df.to_csv("training/pricing_dataset.csv", index=False)
print(f"Dataset final: {len(df)} filas")
print(df.head())