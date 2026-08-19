from db import engine
from sqlalchemy import text

with engine.connect() as conn:
    total = conn.execute(text('SELECT COUNT(*) FROM "Reviews"')).scalar()
    with_text = conn.execute(text('''
        SELECT COUNT(*) FROM "Reviews"
        WHERE "Comment" IS NOT NULL AND LENGTH(TRIM("Comment")) > 0
    ''')).scalar()
    print("Reseñas totales:", total)
    print("Reseñas con comentario de texto:", with_text)