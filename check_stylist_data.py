from db import engine
from sqlalchemy import text

with engine.connect() as conn:
    products_by_cat = conn.execute(text('''
        SELECT "Category", COUNT(*) FROM "Products"
        WHERE "IsActive" = true
        GROUP BY "Category"
    ''')).fetchall()

    users_with_orders = conn.execute(text('''
        SELECT COUNT(DISTINCT "UserId") FROM "Orders" WHERE "Status" = 'Paid'
    ''')).scalar()

    print("Productos por categoría:")
    for cat, count in products_by_cat:
        print(f"  {cat}: {count}")
    print("Usuarios con al menos 1 orden pagada:", users_with_orders)