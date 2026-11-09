from pathlib import Path

import numpy as np
from PIL import Image

from preprocessing.image_preprocessor import ImagePreprocessor
from services.database_service import get_training_samples
from services.prediction_service import PredictionService
from tensorflow import keras
from models.emnist_model import EmnistCnnModel


def prepare_training_data():

    samples = get_training_samples()
    preprocessor = ImagePreprocessor()

    characters = PredictionService.EMNIST_CLASSES

    X = []
    y = []

    for sample in samples:

        image_path = Path(sample["image_path"])
        character = sample["actual_character"]

        if not image_path.is_file():
            continue

        if character not in characters:
            continue

        with Image.open(image_path) as image:
            processed = preprocessor.preprocess(image)

        if processed is None:
            continue

        X.append(processed.image_28x28)
        y.append(characters.index(character))

    if not X:
        return None, None

    X = np.array(X, dtype=np.float32)
    X = X.reshape(-1, 28, 28, 1)

    y = np.array(y, dtype=np.int64)

    return X, y

def prepare_combined_training_data(model):
    # Load reviewed handwriting from SQLite
    X_feedback, y_feedback = prepare_training_data()

    if X_feedback is None:
        raise ValueError("No reviewed handwriting samples found.")

    # Load original EMNIST samples
    X_emnist, y_emnist = model.load_retraining_samples(
        count=2000
    )

    # Combine both datasets
    X = np.concatenate(
        [X_emnist, X_feedback],
        axis=0
    )

    y = np.concatenate(
        [y_emnist, y_feedback],
        axis=0
    )

    # Shuffle training samples together
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(y))

    return X[indices], y[indices]

def retrain_model(epochs=2):
    model = EmnistCnnModel()
    model.load_or_train()

    # Prepare combined training data
    X, y = prepare_combined_training_data(model)

    # Use a low learning rate
    model.model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Fine-tune existing CNN
    history = model.model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=32,
        verbose=1
    )

    # Save to a NEW file, preserving original model
    new_path = model.model_path.with_name(
        "emnist_model_finetuned.keras"
    )

    model.model.save(new_path)

    return history, new_path