import struct
import zipfile
import gzip
from pathlib import Path

import numpy as np
import requests
from tensorflow import keras
from tensorflow.keras import layers


class EmnistCnnModel:

    def __init__(self, model_path: str = "artifacts/emnist_model.keras"):
        self.model_path = Path(model_path)
        self.model = None

        self.data_dir = Path("artifacts/emnist")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_images(self, filename):

        with gzip.open(filename, "rb") as f:
            magic, num_images, rows, cols = struct.unpack(
                ">IIII",
                f.read(16)
            )

            images = np.frombuffer(
                f.read(),
                dtype=np.uint8
            )

        return images.reshape(
            num_images,
            rows,
            cols
        )


    def load_labels(self, filename):

        with gzip.open(filename, "rb") as f:
            magic, num_labels = struct.unpack(
                ">II",
                f.read(8)
            )

            labels = np.frombuffer(
                f.read(),
                dtype=np.uint8
            )

        return labels


    def dataset_exists(self):
        gzip_dir = self.data_dir / "gzip"
        required_files = [
            "emnist-byclass-train-images-idx3-ubyte.gz",
            "emnist-byclass-train-labels-idx1-ubyte.gz",
            "emnist-byclass-test-images-idx3-ubyte.gz",
            "emnist-byclass-test-labels-idx1-ubyte.gz",
        ]

        for filename in required_files:

            if not (gzip_dir / filename).exists():
                return False

        return True


    def download_dataset(self):

        print("EMNIST dataset is missing.")
        print("Downloading EMNIST...")

        zip_file = self.data_dir / "emnist.zip"

        # Put the verified EMNIST download URL here.
        url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

        response = requests.get(
            url,
            timeout=120
        )

        response.raise_for_status()

        with open(zip_file, "wb") as f:
            f.write(response.content)


        # IMPORTANT:
        # Verify that what we downloaded really is a ZIP.
        if not zipfile.is_zipfile(zip_file):

            zip_file.unlink(missing_ok=True)

            raise RuntimeError(
                "EMNIST download did not return a valid ZIP file."
            )


        print("Extracting EMNIST...")

        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(self.data_dir)

        zip_file.unlink()

        print("EMNIST download completed.")


    def load_or_train(self):

        # -----------------------------------------
        # Load existing trained model
        # -----------------------------------------

        if self.model_path.exists():

            print("Loading existing EMNIST model...")

            self.model = keras.models.load_model(
                self.model_path
            )

            return self.model


        # -----------------------------------------
        # Make sure EMNIST data exists
        # -----------------------------------------

        if not self.dataset_exists():

            self.download_dataset()


        # -----------------------------------------
        # Load EMNIST
        # -----------------------------------------
        gzip_dir = self.data_dir / "gzip"

        X_train = self.load_images(
            gzip_dir /
            "emnist-byclass-train-images-idx3-ubyte.gz"
        )

        y_train = self.load_labels(
            gzip_dir /
            "emnist-byclass-train-labels-idx1-ubyte.gz"
        )

        X_test = self.load_images(
            gzip_dir /
            "emnist-byclass-test-images-idx3-ubyte.gz"
        )

        y_test = self.load_labels(
            gzip_dir /
            "emnist-byclass-test-labels-idx1-ubyte.gz"
        )


        # -----------------------------------------
        # Correct EMNIST image orientation
        # -----------------------------------------

        X_train = np.transpose(
            X_train,
            (0, 2, 1)
        )

        X_test = np.transpose(
            X_test,
            (0, 2, 1)
        )


        # -----------------------------------------
        # Normalize
        # -----------------------------------------

        X_train = (
            X_train.astype("float32") / 255.0
        )

        X_test = (
            X_test.astype("float32") / 255.0
        )


        # Add grayscale channel
        X_train = np.expand_dims(
            X_train,
            -1
        )

        X_test = np.expand_dims(
            X_test,
            -1
        )


        # -----------------------------------------
        # CNN
        # -----------------------------------------

        self.model = keras.Sequential(
            [
                layers.Input(
                    shape=(28, 28, 1)
                ),

                layers.Conv2D(
                    32,
                    3,
                    activation="relu"
                ),

                layers.MaxPooling2D(),

                layers.Conv2D(
                    64,
                    3,
                    activation="relu"
                ),

                layers.MaxPooling2D(),

                layers.Flatten(),

                layers.Dropout(0.25),

                layers.Dense(
                    128,
                    activation="relu"
                ),

                layers.Dropout(0.25),

                layers.Dense(
                    62,
                    activation="softmax"
                ),
            ]
        )


        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )


        print("Training EMNIST model...")


        self.model.fit(
            X_train,
            y_train,
            epochs=3,
            batch_size=128,
            validation_split=0.10,
            verbose=1,
        )


        test_loss, test_accuracy = (
            self.model.evaluate(
                X_test,
                y_test,
                verbose=0
            )
        )


        print(
            f"EMNIST test accuracy: "
            f"{test_accuracy:.4f}"
        )


        # -----------------------------------------
        # Save trained model
        # -----------------------------------------

        self.model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.model.save(
            self.model_path
        )


        return self.model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load_or_train()
        return self.model.predict(X, verbose=0)

    def load_retraining_samples(self, count=2000):
        gzip_dir = self.data_dir / "gzip"

        X = self.load_images(
            gzip_dir / "emnist-byclass-train-images-idx3-ubyte.gz"
        )

        y = self.load_labels(
            gzip_dir / "emnist-byclass-train-labels-idx1-ubyte.gz"
        )

        rng = np.random.default_rng(42)
        indices = rng.choice(
            len(y),
            size=min(count, len(y)),
            replace=False
        )

        X = np.transpose(X[indices], (0, 2, 1))
        X = X.astype(np.float32) / 255.0
        X = X.reshape(-1, 28, 28, 1)

        return X, y[indices]

    def load_test_samples(self, count=2000):
        gzip_dir = self.data_dir / "gzip"

        X = self.load_images(
            gzip_dir / "emnist-byclass-test-images-idx3-ubyte.gz"
        )

        y = self.load_labels(
            gzip_dir / "emnist-byclass-test-labels-idx1-ubyte.gz"
        )

        rng = np.random.default_rng(123)
        indices = rng.choice(
            len(y),
            size=min(count, len(y)),
            replace=False
        )

        X = np.transpose(X[indices], (0, 2, 1))
        X = X.astype(np.float32) / 255.0
        X = X.reshape(-1, 28, 28, 1)

        return X, y[indices]

    def fine_tune(self, X, y, epochs=2):
        if self.model is None:
            self.load_or_train()

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=16,
            verbose=1
        )

        return history
