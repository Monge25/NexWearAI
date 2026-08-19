from db import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text('''
        SELECT o."Id" as order_id, o."UserId", oi."ProductId", p."Name"
        FROM "Orders" o
        JOIN "OrderItems" oi ON oi."OrderId" = o."Id"
        JOIN "Products" p ON p."Id" = oi."ProductId"
        WHERE o."Status" = 'Paid'
        LIMIT 5
    '''))
    for row in result:
        print(f"OrderId: {row[0]} | UserId: {row[1]} | ProductId: {row[2]} | Producto: {row[3]}")