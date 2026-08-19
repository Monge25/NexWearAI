from db import engine
from sqlalchemy import text
import pandas as pd

TOP_KEYWORDS = ["camisa", "blusa", "playera", "camiseta", "polo", "top", "sueter", "suéter", "sudadera", "cardigan"]
BOTTOM_KEYWORDS = ["pantalón", "pantalon", "short", "falda", "jean", "palazzo"]

def infer_slot(name):
    n = name.lower()
    for kw in TOP_KEYWORDS:
        if kw in n:
            return "top"
    for kw in BOTTOM_KEYWORDS:
        if kw in n:
            return "bottom"
    return "other"

with engine.connect() as conn:
    df = pd.read_sql(text('SELECT "Name", "Category", "Price" FROM "Products" WHERE "IsActive" = true'), conn)

df["slot"] = df["Name"].apply(infer_slot)

for gender in ["mujer", "hombre"]:
    sub = df[df["Category"].str.lower() == gender]
    print(f"\n=== {gender.upper()} ===")
    for slot in ["top", "bottom"]:
        items = sub[sub["slot"] == slot][["Name", "Price"]]
        print(f"  {slot}: {len(items)} productos")
        print(items.to_string(index=False))