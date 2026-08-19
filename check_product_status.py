from db import engine
from sqlalchemy import text

ids = [
    "05fc7cb4-9468-4086-96df-e11c2cdb75b3",
    "8e92f087-ccd5-4615-8216-1b294921b883",
    "b7dc8385-1e5b-4f8a-9293-ddb0c0e8ff98",
]

with engine.connect() as conn:
    for pid in ids:
        result = conn.execute(text('SELECT "Name", "IsActive" FROM "Products" WHERE "Id" = :id'), {"id": pid}).fetchone()
        print(pid, "->", result)