import csv
import os
from itertools import islice
from typing import *

from utils import Utils, Dataset_Info
from .fer2013 import Fer2013


class FerPlus(Fer2013):
    def __init__(
            self,
            info: Dataset_Info = Utils.fer_plus(),
            fer_info: Dataset_Info = Utils.fer2013(),
            loader: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,

    ) -> None:
        self.fer_root = os.path.join(fer_info.data_file, fer_info.csv_file)
        super().__init__(info, loader, transform=transform, target_transform=target_transform)

    def make_dataset(self, info) -> tuple[list[tuple[str, list[float]]], list[list[float]]]:
        samples = []
        targets = []

        ferplus_entries = []
        with open(self.root, 'r') as fer_plus:
            ferplus_rows = csv.reader(fer_plus, delimiter=',')
            for row in islice(ferplus_rows, 1, None):
                ferplus_entries.append(row)

        index = 0
        with open(self.fer_root, 'r') as fer2013:
            fer_rows = csv.reader(fer2013, delimiter=',')
            for row in islice(fer_rows, 1, None):
                fer_plus_row = ferplus_entries[index]
                file_name = fer_plus_row[1].strip()
                if len(file_name) > 0:
                    file_name = f"fer{str(index).zfill(7)}.jpg"
                label = list(map(float, fer_plus_row[2:len(fer_plus_row)]))
                folder = row[2].lower().capitalize()
                Utils.folder_exists_or_create(os.path.join(info.data_file, 'usage_separated', folder))
                image_path = os.path.join(info.data_file, 'usage_separated', folder, file_name)
                if not Utils.file_exists(image_path):
                    image = Utils.str_to_image(row[1])
                    image.save(image_path, compress_level=0)
                item = (image_path, label)
                samples.append(item)
                targets.append(label)
                index += 1
        return samples, targets
