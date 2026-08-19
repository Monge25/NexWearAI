from fastapi import APIRouter
from pydantic import BaseModel
import joblib
from scipy.sparse import hstack
import numpy as np
import os

router = APIRouter()

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "review_detector_model.pkl")
data = joblib.load(MODEL_PATH)
model = data["model"]
vectorizer = data["vectorizer"]

class ReviewInput(BaseModel):
    comment: str
    rating: int

@router.post("/")
def detect_fake_review(review: ReviewInput):
    X_text = vectorizer.transform([review.comment])
    comment_length = len(review.comment)
    exclamation_count = review.comment.count("!")
    X_numeric = np.array([[review.rating, comment_length, exclamation_count]])
    X = hstack([X_text, X_numeric])

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(max(probabilities))

    return {
        "isFake": bool(prediction),
        "confidence": round(confidence, 4)
    }