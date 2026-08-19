"""
Script de verificación — Smart Discounts.
Cuenta usuarios totales vs usuarios con al menos una orden, para confirmar
que hay suficiente historial real para calcular RFM (recencia, frecuencia,
valor promedio) antes de entrenar el clasificador de promociones.
"""

from db import engine
from sqlalchemy import text

with engine.connect() as conn:
    users = conn.execute(text('SELECT COUNT(*) FROM "Users"')).scalar()
    orders_per_user = conn.execute(text('''
        SELECT COUNT(*) FROM (
            SELECT "UserId" FROM "Orders" GROUP BY "UserId"
        ) sub
    ''')).scalar()
    print("Usuarios totales:", users)
    print("Usuarios con al menos 1 orden:", orders_per_user)