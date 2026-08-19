import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from db import engine
from sqlalchemy import text

np.random.seed(42)

# 1. Traer los productos reales de tu catálogo
with engine.connect() as conn:
    products = pd.read_sql(text('SELECT "Id", "Category" FROM "Products" WHERE "IsActive" = true'), conn)

print(f"Productos encontrados: {len(products)}")

# 2. Generar 52 semanas de historial sintético por producto
rows = []
for _, prod in products.iterrows():
    base_demand = np.random.randint(3, 15)  # unidades base por semana
    for week in range(1, 53):
        month = ((week - 1) // 4) % 12 + 1
        seasonality = 1.5 if month in [11, 12] else (0.7 if month in [1, 2] else 1.0)  # boost en nov-dic
        on_sale = np.random.choice([0, 1], p=[0.85, 0.15])
        sale_boost = 1.4 if on_sale else 1.0
        noise = np.random.normal(0, 1.5)
        units = max(0, round(base_demand * seasonality * sale_boost + noise))
        rows.append({
            "ProductId": prod["Id"],
            "week": week,
            "month": month,
            "on_sale": on_sale,
            "units_sold": units,
            "source": "synthetic"
        })

synthetic_df = pd.DataFrame(rows)

# 3. Traer los datos reales (aunque sean pocos) y agregarlos
query_real = """
SELECT oi."ProductId", o."CreatedAt", oi."Quantity", pv."IsOnSale"
FROM "OrderItems" oi
JOIN "Orders" o ON o."Id" = oi."OrderId"
JOIN "ProductVariants" pv ON pv."Id" = oi."VariantId"
WHERE o."Status" != 'Cancelled'
"""
with engine.connect() as conn:
    real_df = pd.read_sql(text(query_real), conn)

if len(real_df) > 0:
    real_df["CreatedAt"] = pd.to_datetime(real_df["CreatedAt"])
    real_df["week"] = real_df["CreatedAt"].dt.isocalendar().week.astype(int)
    real_df["month"] = real_df["CreatedAt"].dt.month
    real_grouped = (
        real_df.groupby(["ProductId", "week", "month"])
        .agg(units_sold=("Quantity", "sum"), on_sale=("IsOnSale", "max"))
        .reset_index()
    )
    real_grouped["source"] = "real"
else:
    real_grouped = pd.DataFrame(columns=synthetic_df.columns)

# 4. Combinar y guardar
final_df = pd.concat([synthetic_df, real_grouped], ignore_index=True)
final_df.to_csv("training/demand_dataset.csv", index=False)

print(f"Dataset final: {len(final_df)} filas ({len(synthetic_df)} sintéticas, {len(real_grouped)} reales)")
print(final_df.head())