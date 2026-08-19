from db import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text('''
        SELECT "Id", "IsSuspectedFake", "FakeReviewConfidence"
        FROM "Reviews"
        WHERE "Id" = :id
    '''), {"id": "fd323437-4a05-415c-8ec0-a7c6d3e91e2f"})
    row = result.fetchone()
    print(row)