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
import joblib

np.random.seed(42)

def normalize_category(cat: str) -> str:
    cat = (cat or "").strip().lower()
    if cat in ("exterior", "exteriores"):
        return "exteriores"
    if cat in ("tops",):
        return "mujer"  # se integra al catálogo general de mujer
    return cat

# 1. Traer productos reales
with engine.connect() as conn:
    products = pd.read_sql(text('''
        SELECT "Id", "Name", "Category", "Price", "ImageUrl"
        FROM "Products"
        WHERE "IsActive" = true
    '''), conn)

products["Category"] = products["Category"].apply(normalize_category)
print(f"Productos encontrados: {len(products)}")
print(products["Category"].value_counts())

# 2. Features para clustering: categoría (one-hot) + precio normalizado
cat_dummies = pd.get_dummies(products["Category"], prefix="cat")
scaler = StandardScaler()
price_scaled = scaler.fit_transform(products[["Price"]])

X_products = np.hstack([cat_dummies.values, price_scaled])

# 3. Clustering: 4 estilos (clásico, casual, urbano, elegante) - ajustable
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
product_clusters = kmeans.fit_predict(X_products)
products["cluster"] = product_clusters

print("\nProductos por cluster:")
print(products["cluster"].value_counts())

# 4. Generar dataset sintético de usuarios (presupuesto + categoría preferida) -> cluster
STYLE_NAMES = {0: "Casual", 1: "Clasico", 2: "Urbano", 3: "Elegante"}

rows = []
for i in range(300):
    avg_budget = np.random.uniform(300, 2500)
    pref_category_encoded = np.random.randint(0, len(cat_dummies.columns))
    cat_vector = [0] * len(cat_dummies.columns)
    cat_vector[pref_category_encoded] = 1

    # Simular a qué cluster tiende ese perfil (basado en presupuesto + categoría)
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

# 5. Entrenar red neuronal (MLP) para asignar usuarios nuevos a un cluster de estilo
# Optimización: GridSearchCV sobre arquitectura de capas ocultas y tasa de
# regularización (alpha), con validación cruzada, para elegir la red más simple
# que generalice bien (evitar sobreajuste de una red demasiado grande para
# un problema de solo 2 features de entrada).
from sklearn.model_selection import train_test_split, GridSearchCV

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
# 6. Guardar todo lo necesario para servir el modelo
joblib.dump({
    "mlp_model": mlp,
    "scaler": scaler,
    "category_columns": list(cat_dummies.columns),
    "style_names": STYLE_NAMES,
    "products": products.to_dict("records"),
}, "models/stylist_model.pkl")

print("\nModelo guardado en models/stylist_model.pkl")