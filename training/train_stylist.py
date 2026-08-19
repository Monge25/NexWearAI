"""
Entrenamiento — Fashion Stylist
=================================
Algoritmo: K-Means (clustering de productos en 4 estilos) + MLPClassifier
(red neuronal que asigna usuarios nuevos a un cluster).
Justificación: K-Means agrupa productos por categoría+precio sin necesitar
etiquetas manuales de "estilo"; el MLP luego aprende a mapear presupuesto+
categoría preferida de un usuario nuevo al cluster más cercano.

Optimización: GridSearchCV sobre arquitectura de capas ocultas y alpha,
con validación cruzada, para elegir la red más simple que generalice bien.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from db import engine
from sqlalchemy import text
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

np.random.seed(42)


def normalize_category(cat: str) -> str:
    """Normaliza variantes de nombre de categoría a un valor único."""
    cat = (cat or "").strip().lower()
    if cat in ("exterior", "exteriores"):
        return "exteriores"
    if cat in ("tops",):
        return "mujer"
    return cat


def load_active_products(engine) -> pd.DataFrame:
    """Carga los productos activos con su categoría normalizada."""
    with engine.connect() as conn:
        df = pd.read_sql(text('''
            SELECT "Id", "Name", "Category", "Price", "ImageUrl"
            FROM "Products"
            WHERE "IsActive" = true
        '''), conn)
    df["Category"] = df["Category"].apply(normalize_category)
    return df


products: pd.DataFrame = load_active_products(engine)
print(f"Productos encontrados: {len(products)}")
print(products["Category"].value_counts())

cat_dummies = pd.get_dummies(products["Category"], prefix="cat")
scaler = StandardScaler()
price_scaled = scaler.fit_transform(products[["Price"]])

X_products = np.hstack([cat_dummies.values, price_scaled])

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
product_clusters = kmeans.fit_predict(X_products)
products["cluster"] = product_clusters

print("\nProductos por cluster:")
print(products["cluster"].value_counts())

STYLE_NAMES = {0: "Casual", 1: "Clasico", 2: "Urbano", 3: "Elegante"}

rows = []
for i in range(300):
    avg_budget = np.random.uniform(300, 2500)
    pref_category_encoded = np.random.randint(0, len(cat_dummies.columns))
    cat_vector = [0] * len(cat_dummies.columns)
    cat_vector[pref_category_encoded] = 1

    budget_scaled = scaler.transform(pd.DataFrame([[avg_budget]], columns=["Price"]))[0][0]
    synthetic_features = np.array(cat_vector + [budget_scaled]).reshape(1, -1)
    distances = kmeans.transform(synthetic_features)[0]
    target_cluster = int(np.argmin(distances))

    rows.append({
        "avg_budget": avg_budget,
        **{f"cat_{i}": cat_vector[i] for i in range(len(cat_vector))},
        "cluster": target_cluster
    })

df_users = pd.DataFrame(rows)
X_users = df_users.drop(columns=["cluster"]).values
y_users = df_users["cluster"].values

X_train, X_test, y_train, y_test = train_test_split(X_users, y_users, test_size=0.2, random_state=42, stratify=y_users)

param_grid = {
    "hidden_layer_sizes": [(16,), (32, 16), (32, 16, 8)],
    "alpha": [0.0001, 0.001, 0.01],
}

grid = GridSearchCV(
    MLPClassifier(max_iter=2000, random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

print(f"Mejores hiperparámetros encontrados: {grid.best_params_}")
print(f"Accuracy promedio en validación cruzada: {grid.best_score_:.2%}")

mlp = grid.best_estimator_
test_acc = mlp.score(X_test, y_test)
print(f"Accuracy en conjunto de prueba (hold-out): {test_acc:.2%}")

joblib.dump({
    "mlp_model": mlp,
    "scaler": scaler,
    "category_columns": list(cat_dummies.columns),
    "style_names": STYLE_NAMES,
    "products": products.to_dict("records"),
}, "models/stylist_model.pkl")

print("\nModelo guardado en models/stylist_model.pkl")