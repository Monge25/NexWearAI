from db import engine
from sqlalchemy import text

with engine.connect() as conn:
    variants = conn.execute(text('SELECT COUNT(*) FROM "ProductVariants"')).scalar()
    cart_items = conn.execute(text('SELECT COUNT(*) FROM "CartItems"')).scalar()
    print("Variantes de producto:", variants)
    print("Items en carritos (histórico):", cart_items)