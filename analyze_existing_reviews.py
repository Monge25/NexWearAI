"""
Script de análisis retroactivo — Detección de Reseñas Falsas.
Corre el modelo ya entrenado sobre todas las reseñas existentes en la base
(que se crearon antes de que el modelo estuviera conectado a
ReviewsController.Create) y actualiza IsSuspectedFake/FakeReviewConfidence
para cada una, sin auto-aprobar ni auto-rechazar.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
from scipy.sparse import hstack
from db import engine
from sqlalchemy import text

data = joblib.load("models/review_detector_model.pkl")
model = data["model"]
vectorizer = data["vectorizer"]

def predict(comment: str, rating: int) -> tuple:
    """Devuelve (is_fake, confidence) para un comentario y rating dados."""
    X_text = vectorizer.transform([comment])
    comment_length = len(comment)
    exclamation_count = comment.count("!")
    X_numeric = np.array([[rating, comment_length, exclamation_count]])
    X = hstack([X_text, X_numeric])

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(max(probabilities))
    return bool(prediction), round(confidence, 4)

with engine.connect() as conn:
    reviews = conn.execute(text('''
        SELECT "Id", "Comment", "Rating"
        FROM "Reviews"
        WHERE "Comment" IS NOT NULL AND LENGTH(TRIM("Comment")) > 0
    ''')).fetchall()

    print(f"Reseñas a analizar: {len(reviews)}")

    for review_id, comment, rating in reviews:
        is_fake, confidence = predict(comment, rating)
        conn.execute(
            text('''
                UPDATE "Reviews"
                SET "IsSuspectedFake" = :is_fake, "FakeReviewConfidence" = :confidence
                WHERE "Id" = :id
            '''),
            {"is_fake": is_fake, "confidence": confidence, "id": review_id}
        )
        print(f"  {review_id} -> isFake={is_fake}, confidence={confidence}")

    conn.commit()
    print("Listo, todas las reseñas fueron analizadas.")