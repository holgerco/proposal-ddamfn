import os
from itertools import islice

from utils import Utils, Dataset_Info
from .base import BaseDataset
from typing import *
import csv


class Fer2013(BaseDataset):

    def __init__(
            self,
            info: Dataset_Info = Utils.fer2013(),
            loader: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        root = os.path.join(info.data_file, info.csv_file)
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.classes, self.class_to_idx = self.find_classes(info)
        self.samples, self.targets = self.make_dataset(info)

        if loader:
            self.loader = loader
        else:
            self.loader = self._default_loader

    def find_classes(self, info) -> tuple[list[str], dict[str, int]]:
        classes = info.categories

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    def make_dataset(self, info) -> tuple[list[tuple[str, int]], list[int]]:
        samples = []
        targets = []
        index = 0
        with open(self.root, 'r') as fer2013:
            fer_rows = csv.reader(fer2013, delimiter=',')
            for row in islice(fer_rows, 1, None):
                label = row[0]
                file_name = f"fer{str(index).zfill(7)}.jpg"
                folder = row[2].lower().capitalize()
                Utils.folder_exists_or_create(os.path.join(info.data_file, 'usage_separated', folder))
                image_path = os.path.join(info.data_file, 'usage_separated', folder, file_name)
                if not Utils.file_exists(image_path):
                    image = Utils.str_to_image(row[1])
                    image.save(image_path, compress_level=0)
                item = (image_path, int(label))
                samples.append(item)
                targets.append(int(label))
                index += 1
                if len(samples) >= 1000:
                    break
        return samples, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)
