"""
Script de verificación — Pricing Intelligence.
Cuenta variantes de producto e items históricos en carritos, usados como
proxy de popularidad para el dataset de entrenamiento.
"""

from db import engine
from sqlalchemy import text

with engine.connect() as conn:
    variants = conn.execute(text('SELECT COUNT(*) FROM "ProductVariants"')).scalar()
    cart_items = conn.execute(text('SELECT COUNT(*) FROM "CartItems"')).scalar()

    if variants == 0:
        print("Advertencia: no hay variantes de producto registradas.")
    else:
        print("Variantes de producto:", variants)

    print("Items en carritos (histórico):", cart_items)