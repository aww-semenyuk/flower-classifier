import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassAccuracy
from transformers import ViTForImageClassification

from flower_classifier.models import SimpleCNN


class FlowerClassifierModule(L.LightningModule):
    def __init__(self, num_classes: int, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.lr = cfg.model.learning_rate

        if cfg.model.name == "cnn":
            self.model = SimpleCNN(num_classes)

        elif cfg.model.name == "vit":
            self.model = ViTForImageClassification.from_pretrained(
                cfg.model.model_id, num_labels=num_classes, ignore_mismatched_sizes=True
            )
            if cfg.model.train_only_classifier_head:
                for param in self.model.vit.parameters():
                    param.requires_grad = False

                for param in self.model.classifier.parameters():
                    param.requires_grad = True
        else:
            raise ValueError(f"Unknown model_name: {cfg.model.name}")

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        if isinstance(self.model, ViTForImageClassification):
            return self.model(pixel_values=x).logits
        return self.model(x)

    def on_fit_start(self):
        self.model.train()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = logits.argmax(dim=1)
        acc = self.train_acc(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = logits.argmax(dim=1)
        acc = self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        acc = self.test_acc(preds, y)

        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
