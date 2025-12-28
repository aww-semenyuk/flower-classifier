import lightning as L
from datasets import load_from_disk
from lightning.pytorch import seed_everything
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from flower_classifier.config import DATA_DIR, get_hydra_cfg
from flower_classifier.lightning_modules import FlowerClassifierModule
from flower_classifier.utils import FlowerDatasetWrapper


def train(cfg: DictConfig) -> None:
    seed_everything(42, workers=True)

    dataset = load_from_disk(DATA_DIR / cfg.data.dataset_output_dir)

    train_loader = DataLoader(
        FlowerDatasetWrapper(dataset["train"], cfg),
        batch_size=cfg.model.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        FlowerDatasetWrapper(dataset["valid"], cfg),
        batch_size=cfg.model.val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        FlowerDatasetWrapper(dataset["test"], cfg),
        batch_size=cfg.model.val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    num_classes = dataset["train"].features["label"].num_classes

    model = FlowerClassifierModule(num_classes=num_classes, cfg=cfg)

    trainer = L.Trainer(
        deterministic=True,
        max_epochs=cfg.model.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


def main(overrides: list[str] | None = None) -> None:
    cfg = get_hydra_cfg(overrides)
    train(cfg)


if __name__ == "__main__":
    main()
