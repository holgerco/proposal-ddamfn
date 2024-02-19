import os

import numpy as np
from PIL import Image
import collections

Dataset_Info = collections.namedtuple("Dataset_Info", ["data_file", "categories", "csv_file"])


class Utils:
    @staticmethod
    def fer2013():
        return Dataset_Info(
            data_file='/home/holger/Desktop/AI/vision/datasets/Fer2013',
            categories=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
            csv_file="fer2013.csv"
        )

    @staticmethod
    def fer_plus():
        return Dataset_Info(
            data_file='/home/holger/Desktop/AI/vision/datasets/FerPlus',
            categories=['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt'],
            csv_file="ferplus.csv"
        )

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

    @staticmethod
    def folder_exists_or_create(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    @staticmethod
    def file_exists(file_path):
        return os.path.exists(file_path)
