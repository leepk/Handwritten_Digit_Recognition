from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionResult:
    digit: int
    confidence: float
    probabilities: np.ndarray


class PredictionService:
    def __init__(self, model):
        self.model = model

    def predict(self, image_tensor: np.ndarray) -> PredictionResult:
        # Ask the model for the probability of each possible digit.
        probabilities = self.model.predict_proba(image_tensor)[0].astype(float)
        # Select the digit with the highest probability.
        digit = int(np.argmax(probabilities))
        confidence = float(probabilities[digit])

        return PredictionResult(
            digit=digit,
            confidence=confidence,
            probabilities=probabilities,
        )
