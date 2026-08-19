from db import engine
from sqlalchemy import text

with engine.connect() as conn:
    orders = conn.execute(text('SELECT COUNT(*) FROM "Orders" WHERE "Status" != \'Cancelled\'')).scalar()
    items = conn.execute(text('SELECT COUNT(*) FROM "OrderItems"')).scalar()
    print("Órdenes válidas:", orders)
    print("OrderItems totales:", items)