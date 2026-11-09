from pathlib import Path
from datetime import datetime

IMAGE_DIR = Path("data/images")


def save_original_image(image):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png"
    image_path = IMAGE_DIR / filename

    image.save(image_path)

    return str(image_path)