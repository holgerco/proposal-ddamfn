from abc import ABC

from torchvision import datasets

from utils import Utils


class BaseDataset(datasets.VisionDataset, ABC):
    @property
    def _default_loader(self):
        return Utils.pillow_loader
