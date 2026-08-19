from db import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text('DELETE FROM "PromotionLogs"'))
    conn.commit()
    print(f"Registros eliminados: {result.rowcount}")