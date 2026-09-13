from PIL import Image, ImageDraw

from preprocessing.image_preprocessor import ImagePreprocessor


def test_preprocessor_returns_mnist_shape():
    image = Image.new("RGB", (280, 280), "black")
    draw = ImageDraw.Draw(image)
    draw.ellipse((70, 35, 210, 245), outline="white", width=18)

    result = ImagePreprocessor().preprocess(image)

    assert result is not None
    assert result.image_28x28.shape == (28, 28)
    assert result.tensor.shape == (1, 28, 28, 1)
    assert result.image_28x28.min() >= 0.0
    assert result.image_28x28.max() <= 1.0
