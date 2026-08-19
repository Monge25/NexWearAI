from db import engine
from sqlalchemy import text
import pandas as pd

with engine.connect() as conn:
    df = pd.read_sql(text('''
        SELECT "Id", "Name", "Category", "Price", "Material", "Tags"
        FROM "Products"
        WHERE "IsActive" = true
    '''), conn)

print(df)