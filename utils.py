import numpy as np
from PIL import Image


class Utils:
    @staticmethod
    def str_to_image(image_blob):
        """Convert a string blob to an image object. """
        image_string = image_blob.split(' ')
        image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
        return Image.fromarray(image_data)

    @staticmethod
    def pillow_loader(path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
