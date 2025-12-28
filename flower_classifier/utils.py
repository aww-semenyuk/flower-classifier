from omegaconf import DictConfig
from torchvision import transforms
from transformers import ViTImageProcessor


class FlowerDatasetWrapper:
    def __init__(self, ds, cfg: DictConfig):
        self.cfg = cfg
        self.ds = ds

        if self.cfg.model.name == "cnn":
            self.processor = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                ]
            )

        elif self.cfg.model.name == "vit":
            self.processor = ViTImageProcessor.from_pretrained(self.cfg.model.model_id)

    def _transform(self, image):
        if self.cfg.model.name == "cnn":
            return self.processor(image)
        if self.cfg.model.name == "vit":
            return self.processor(image, return_tensors="pt")["pixel_values"][0]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        label = item["label"]

        image = self._transform(image)

        return image, label
