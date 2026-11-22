
from pathlib import Path

import joblib

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "emergency_classifier.joblib"
)

# Provisional threshold; not a calibrated probability of correctness.
REVIEW_THRESHOLD = 0.60

_model = None


def load_model():
    """Load the trained classifier once and reuse it."""
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run train_model.py first."
            )

        _model = joblib.load(MODEL_PATH)

    return _model


def classify_request(message: str) -> dict:
    """Classify a disaster message and flag uncertain predictions."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please enter a non-empty message.")

    model = load_model()
    probabilities = model.predict_proba([message])[0]

    best_index = probabilities.argmax()
    category = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])

    return {
        "category": category,
        "confidence": round(confidence, 4),
        "requires_review": confidence < REVIEW_THRESHOLD,
        "review_threshold": REVIEW_THRESHOLD,
        "safety_note": (
            "This is an AI-generated classification, not a verified fact. "
            "Check the original message and confirm details before acting."
        ),
    }


if __name__ == "__main__":
    examples = [
        "We urgently need drinking water for families affected by flooding.",
        "Several buildings have collapsed after the earthquake.",
        "Volunteers are distributing food to affected residents.",
        "Donate money to help cyclone victims.",
    ]

    for message in examples:
        print(f"\nMessage: {message}")
        print("Result:", classify_request(message))

