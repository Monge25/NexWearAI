"""
Entrenamiento — Fashion Pricing Intelligence
==============================================
Algoritmo: GradientBoostingRegressor
Justificación: a diferencia de Random Forest (promedio de árboles independientes),
Gradient Boosting construye árboles secuenciales que corrigen el error del anterior,
lo cual captura mejor relaciones de umbral (ej. "si stock < 5 Y demanda alta -> subir
precio") típicas de una regla de negocio como la que generó el dataset de entrenamiento.

Optimización: GridSearchCV sobre n_estimators, learning_rate y max_depth, con
validación cruzada de 5 folds, para evitar sobreajuste (learning_rate alto +
muchos estimadores memoriza el dataset sintético) sin perder precisión.
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("training/pricing_dataset.csv")

X = df[["stock", "units_sold_30d", "cart_adds", "current_price"]]
y = df["delta_pct"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4],
}

grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

print(f"Mejores hiperparámetros encontrados: {grid.best_params_}")
print(f"MAE promedio en validación cruzada: {-grid.best_score_:.2f}")

model = grid.best_estimator_
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"MAE en conjunto de prueba (hold-out): {mae:.2f}")

joblib.dump(model, "models/pricing_model.pkl")
print("Modelo optimizado guardado en models/pricing_model.pkl")