from dataclasses import dataclass
import numpy as np

@dataclass
class PredictionResult:
    confidence: float
    probabilities: np.ndarray
    character : str


class PredictionService:
    EMNIST_CLASSES = (
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
        )
    def __init__(self, model):
        self.model = model

    def predict(self, image_tensor: np.ndarray) -> PredictionResult:
        # Ask the model for the probability of each possible digit.
        probabilities = self.model.predict_proba(image_tensor)[0].astype(float)
        # Select the digit with the highest probability.
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])

        return PredictionResult(
            confidence=confidence,
            probabilities=probabilities,
            character = self.EMNIST_CLASSES[predicted_idx]
        )
