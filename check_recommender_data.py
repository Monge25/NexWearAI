"""
Script de verificación — Recomendador.
Revisa cuántas órdenes tienen más de un producto distinto, dato clave para
decidir si es viable collaborative filtering o si conviene content-based
(fue este último caso: casi ninguna orden tenía múltiples productos).
"""

from db import engine
from sqlalchemy import text

with engine.connect() as conn:
    orders = conn.execute(text('SELECT COUNT(DISTINCT "OrderId") FROM "OrderItems"')).scalar()
    products = conn.execute(text('SELECT COUNT(*) FROM "Products" WHERE "IsActive" = true')).scalar()
    multi_item_orders = conn.execute(text('''
        SELECT COUNT(*) FROM (
            SELECT "OrderId" FROM "OrderItems"
            GROUP BY "OrderId"
            HAVING COUNT(*) > 1
        ) sub
    ''')).scalar()
    print("Órdenes con al menos 1 item:", orders)
    print("Productos activos:", products)
    print("Órdenes con más de 1 producto distinto:", multi_item_orders)