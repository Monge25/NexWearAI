"""
Entrenamiento — Demand Forecast AI
====================================
Algoritmo: RandomForestRegressor
Justificación: los árboles de decisión en ensamble manejan bien relaciones
no lineales entre estacionalidad (semana/mes) y promociones sin necesitar
escalado de features, y son robustos a outliers en el historial de ventas.

Optimización: se aplica GridSearchCV con validación cruzada (5 folds) sobre
n_estimators y max_depth para evitar tanto subajuste como sobreajuste.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import joblib


def load_dataset(path: str) -> pd.DataFrame:
    """Carga el dataset sintético de demanda generado por generate_demand_dataset.py."""
    return pd.read_csv(path)


df: pd.DataFrame = load_dataset("training/demand_dataset.csv")

X: pd.DataFrame = df[["week", "month", "on_sale"]]
y: pd.Series = df["units_sold"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
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

joblib.dump(model, "models/demand_model.pkl")
print("Modelo optimizado guardado en models/demand_model.pkl")