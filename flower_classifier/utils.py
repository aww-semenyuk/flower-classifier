"""Utils module"""

import torch
from datasets import Dataset
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms
from transformers import ViTImageProcessor


class FlowerDatasetWrapper:
    """Dataset wrapper with image transforming"""

    def __init__(self, ds: Dataset, cfg: DictConfig):
        self.cfg = cfg
        self.ds = ds

        # define image processor based on model type
        if self.cfg.model.name == "cnn":
            self.processor = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                ]
            )

        elif self.cfg.model.name == "vit":
            self.processor = ViTImageProcessor.from_pretrained(self.cfg.model.model_id)

    def _transform(self, image: Image.Image) -> torch.Tensor:
        if self.cfg.model.name == "cnn":
            return self.processor(image)
        if self.cfg.model.name == "vit":
            return self.processor(image, return_tensors="pt")["pixel_values"][0]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, label = self.ds[idx]["image"], self.ds[idx]["label"]
        image = self._transform(image)

        return image, label
