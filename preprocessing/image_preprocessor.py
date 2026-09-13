from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class ProcessedImage:
    image_28x28: np.ndarray
    tensor: np.ndarray


class ImagePreprocessor:
    """
    Convert a canvas or uploaded image to MNIST format.
    """

    def preprocess(self, image: Image.Image) -> ProcessedImage | None:
        # Convert the Pillow image to a NumPy RGB image for OpenCV.
        rgb = np.array(image.convert("RGB"))

        # Convert the input image to grayscale.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Detect the background from the image border.
        border = np.concatenate(
            [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]
        )

        # MNIST uses a bright digit on a black background.
        if float(border.mean()) > 127:
            gray = cv2.bitwise_not(gray)

        # Remove weak background noise and create a binary mask.
        _, binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

        # Find the handwritten digit area.
        points = cv2.findNonZero(binary)
        if points is None:
            return None

        # Crop the image around the handwritten digit.
        x, y, width, height = cv2.boundingRect(points)
        digit = gray[y:y + height, x:x + width]

        # Resize the digit to fit inside a 20x20 MNIST-style area.
        scale = min(20.0 / max(height, 1), 20.0 / max(width, 1))
        new_width = max(1, min(20, int(round(width * scale))))
        new_height = max(1, min(20, int(round(height * scale))))

        interpolation = (
            cv2.INTER_AREA
            if new_width < width or new_height < height
            else cv2.INTER_CUBIC
        )
        resized = cv2.resize(
            digit,
            (new_width, new_height),
            interpolation=interpolation,
        )

        # Place the resized digit in the center of a 28x28 image.
        canvas = np.zeros((28, 28), dtype=np.uint8)
        start_x = (28 - new_width) // 2
        start_y = (28 - new_height) // 2
        canvas[
            start_y:start_y + new_height,
            start_x:start_x + new_width,
        ] = resized

        # Center the digit more accurately using image moments.
        canvas = self._center_by_moments(canvas)

        # Normalize pixel values to the 0..1 range expected by the model.
        normalized = canvas.astype(np.float32) / 255.0
        normalized[normalized < 0.03] = 0.0

        # Add batch and channel dimensions for the current MNIST model.
        tensor = normalized.reshape(1, 28, 28, 1)

        return ProcessedImage(
            image_28x28=normalized,
            tensor=tensor,
        )

    @staticmethod
    def _center_by_moments(image: np.ndarray) -> np.ndarray:
        # Calculate the center of the handwritten digit.
        moments = cv2.moments(image)
        if moments["m00"] == 0:
            return image

        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]

        # Move the digit toward the center of the 28x28 image.
        shift_x = 13.5 - center_x
        shift_y = 13.5 - center_y

        transform = np.float32(
            [
                [1, 0, shift_x],
                [0, 1, shift_y],
            ]
        )

        return cv2.warpAffine(
            image,
            transform,
            (28, 28),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
