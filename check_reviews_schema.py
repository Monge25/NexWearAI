from db import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'Reviews'
        ORDER BY ordinal_position
    """))
    for row in result:
        print(row[0], "-", row[1])