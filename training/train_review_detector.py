"""
Entrenamiento — Detección de Reseñas Falsas
==============================================
Algoritmo: RandomForestClassifier sobre features TF-IDF + numéricas
Justificación: TF-IDF captura patrones léxicos de spam sin necesitar
embeddings pesados; Random Forest sobre esas features dispersas es rápido
de entrenar y menos propenso a sobreajuste que un solo árbol.

Optimización: GridSearchCV sobre n_estimators y max_depth con validación
cruzada estratificada, para confirmar que el 100% de accuracy original no
era producto de un único split afortunado sino un resultado estable.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
import numpy as np
import joblib

df = pd.read_csv("training/review_dataset.csv")

# Descarta filas con comentario vacío o solo espacios: TF-IDF no puede
# vectorizar texto vacío de forma útil y distorsionaría el entrenamiento.
df = df[df["comment"].fillna("").str.strip() != ""]
print(f"Reseñas usadas para entrenamiento (tras filtrar vacías): {len(df)}")

tfidf = TfidfVectorizer(max_features=300, min_df=2)
X_text = tfidf.fit_transform(df["comment"])

X_numeric = df[["rating", "comment_length", "exclamation_count"]].values
X = hstack([X_text, X_numeric])
y = df["is_fake"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

print(f"Mejores hiperparámetros encontrados: {grid.best_params_}")
print(f"Accuracy promedio en validación cruzada: {grid.best_score_:.2%}")

model = grid.best_estimator_
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Accuracy en conjunto de prueba (hold-out): {acc:.2%}")
print(classification_report(y_test, preds))

joblib.dump({"model": model, "vectorizer": tfidf}, "models/review_detector_model.pkl")
print("Modelo optimizado guardado en models/review_detector_model.pkl")