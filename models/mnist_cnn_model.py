from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MnistCnnModel:
    name = "MNIST CNN"

    def __init__(self, model_path: str = "artifacts/mnist_cnn.keras"):
        self.model_path = Path(model_path)
        self.model = None

    def load_or_train(self):
        if self.model_path.exists():
            self.model = keras.models.load_model(self.model_path)
            return self.model

        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

        X_train = X_train.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0

        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)

        self.model = keras.Sequential(
            [
                layers.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, 3, activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, activation="relu"),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dropout(0.25),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.25),
                layers.Dense(10, activation="softmax"),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model.fit(
            X_train,
            y_train,
            epochs=3,
            batch_size=128,
            validation_split=0.10,
            verbose=1,
        )

        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"MNIST test accuracy: {test_accuracy:.4f}")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_path)
        return self.model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load_or_train()
        return self.model.predict(X, verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
