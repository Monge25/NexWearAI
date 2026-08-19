import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from db import engine
from sqlalchemy import text

np.random.seed(42)

with engine.connect() as conn:
    users = pd.read_sql(text('''
        SELECT u."Id" as user_id,
               COUNT(o."Id") as frequency,
               COALESCE(AVG(o."Total"), 0) as avg_order_value,
               COALESCE(EXTRACT(DAY FROM (NOW() - MAX(o."CreatedAt"))), 999) as recency_days
        FROM "Users" u
        LEFT JOIN "Orders" o ON o."UserId" = u."Id"
        GROUP BY u."Id"
    '''), conn)

print(f"Usuarios reales encontrados: {len(users)}")

def assign_promo(recency, frequency, avg_value):
    """Regla de negocio para generar el target de entrenamiento."""
    if recency > 60 and frequency <= 1:
        return "Discount"
    elif frequency >= 3 and avg_value > 800:
        return "NoPromotion"
    elif avg_value < 500:
        return "FreeShipping"
    else:
        return np.random.choice(["Discount", "FreeShipping", "NoPromotion"], p=[0.3, 0.3, 0.4])

rows = []
for _, u in users.iterrows():
    recency = float(u["recency_days"]) if u["recency_days"] is not None else 999
    frequency = int(u["frequency"])
    avg_value = float(u["avg_order_value"])
    promo = assign_promo(recency, frequency, avg_value)
    rows.append({
        "recency_days": min(recency, 365),
        "frequency": frequency,
        "avg_order_value": avg_value,
        "cart_abandon_rate": np.random.uniform(0, 1),
        "promotion_type": promo,
        "source": "real"
    })

for i in range(400):
    recency = np.random.randint(0, 365)
    frequency = np.random.randint(0, 15)
    avg_value = np.random.uniform(100, 2500)
    cart_abandon_rate = np.random.uniform(0, 1)
    promo = assign_promo(recency, frequency, avg_value)
    rows.append({
        "recency_days": recency,
        "frequency": frequency,
        "avg_order_value": round(avg_value, 2),
        "cart_abandon_rate": round(cart_abandon_rate, 2),
        "promotion_type": promo,
        "source": "synthetic"
    })

df = pd.DataFrame(rows)
print("Valores nulos por columna:\n", df.isnull().sum())
df.to_csv("training/discount_dataset.csv", index=False)
print(f"Dataset final: {len(df)} filas")
print(df["promotion_type"].value_counts())