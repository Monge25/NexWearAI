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