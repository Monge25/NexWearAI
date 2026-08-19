import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
from db import engine
from sqlalchemy import text

with engine.connect() as conn:
    df = pd.read_sql(text('''
        SELECT "Id", "Name", "Category", "Price", "Description"
        FROM "Products"
        WHERE "IsActive" = true
    '''), conn)

print(f"Productos encontrados: {len(df)}")

df["Description"] = df["Description"].fillna("")

# 1. Similitud de texto (Name + Description) con TF-IDF
text_data = df["Name"] + " " + df["Description"]
tfidf = TfidfVectorizer(stop_words=None, max_features=500)
tfidf_matrix = tfidf.fit_transform(text_data)
text_similarity = cosine_similarity(tfidf_matrix)

# 2. Similitud de categoría (1 si es igual, 0 si no)
category_similarity = np.equal.outer(df["Category"].values, df["Category"].values).astype(float)

# 3. Similitud de precio (más cercano = más similar, normalizado 0-1)
prices = df["Price"].values.reshape(-1, 1)
price_diff = np.abs(prices - prices.T)
max_diff = price_diff.max() if price_diff.max() > 0 else 1
price_similarity = 1 - (price_diff / max_diff)

# 4. Combinar con pesos: categoría pesa más, luego texto, luego precio
final_similarity = (
    0.5 * category_similarity +
    0.35 * text_similarity +
    0.15 * price_similarity
)

# Guardar todo lo necesario para servir recomendaciones después
joblib.dump({
    "product_ids": df["Id"].astype(str).tolist(),
    "product_names": df["Name"].tolist(),
    "similarity_matrix": final_similarity,
}, "models/recommender_model.pkl")

print("Modelo de recomendación guardado en models/recommender_model.pkl")
print("\nEjemplo — productos más parecidos al primero:")
idx = 0
sims = final_similarity[idx]
top_indices = sims.argsort()[::-1][1:4]  # top 3 excluyendo el mismo producto
for i in top_indices:
    print(f"  {df['Name'].iloc[i]} (similitud: {sims[i]:.3f})")