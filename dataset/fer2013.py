import os
from itertools import islice

from utils import Utils
from .base import BaseDataset
from path.path import PATH
from typing import *
import csv

path = PATH()


class Fer2013(BaseDataset):

    def __init__(
            self,
            root: str,
            loader: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.classes, self.class_to_idx = self.find_classes()
        self.samples, self.targets = self.make_dataset()

        if loader:
            self.loader = loader
        else:
            self.loader = self._default_loader

    def find_classes(self) -> tuple[list[str], dict[str, int]]:
        classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    def make_dataset(
            self
    ) -> tuple[list[tuple[str, int]], list[int]]:
        samples = []
        targets = []
        index = 0
        with open(self.root, 'r') as fer2013:
            fer_rows = csv.reader(fer2013, delimiter=',')
            for row in islice(fer_rows, 1, None):
                label = row[0]
                # image = Utils.str_to_image(row[1]) #FIXME
                image_path = os.path.join('static/fer2013', row[2], f"{row[2]}{index}.jpg")
                # image.save(image_path, compress_level=0) #FIXME
                item = (image_path, label)
                samples.append(item)
                targets.append(label)
                index += 1
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
