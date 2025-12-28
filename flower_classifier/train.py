import git
import lightning as L
from datasets import load_from_disk
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from flower_classifier.config import DATA_DIR, PROJECT_DIR, get_hydra_cfg
from flower_classifier.lightning_modules import FlowerClassifierModule
from flower_classifier.utils import FlowerDatasetWrapper


def train(cfg: DictConfig) -> None:
    seed_everything(cfg.train.seed, workers=True)

    dataset = load_from_disk(DATA_DIR / cfg.data.dataset_output_dir)

    train_loader = DataLoader(
        FlowerDatasetWrapper(dataset["train"], cfg),
        batch_size=cfg.model.train_batch_size,
        shuffle=True,
        num_workers=cfg.train.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        FlowerDatasetWrapper(dataset["valid"], cfg),
        batch_size=cfg.model.val_batch_size,
        shuffle=False,
        num_workers=cfg.train.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        FlowerDatasetWrapper(dataset["test"], cfg),
        batch_size=cfg.model.val_batch_size,
        shuffle=False,
        num_workers=cfg.train.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    num_classes = dataset["train"].features["label"].num_classes

    model = FlowerClassifierModule(num_classes=num_classes, cfg=cfg)

    commit_id = git.Repo(str(PROJECT_DIR)).head.commit.hexsha

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        run_name=cfg.model.name,
        tracking_uri=cfg.logging.tracking_uri,
    )

    mlf_logger.log_hyperparams({"commit_id": commit_id})

    train_cfg = cfg.train
    checkpoint_cb = ModelCheckpoint(
        dirpath=PROJECT_DIR / "checkpoints",
        monitor=train_cfg.callbacks.checkpoint.monitor,
        mode=train_cfg.callbacks.checkpoint.mode,
        save_top_k=train_cfg.callbacks.checkpoint.save_top_k,
        filename=train_cfg.callbacks.checkpoint.filename,
    )
    earlystopping_cb = EarlyStopping(
        monitor=train_cfg.callbacks.early_stopping.monitor,
        mode=train_cfg.callbacks.early_stopping.mode,
        patience=train_cfg.callbacks.early_stopping.patience,
        min_delta=train_cfg.callbacks.early_stopping.min_delta,
    )

    trainer = L.Trainer(
        deterministic=True,
        max_epochs=cfg.model.epochs,
        accelerator=cfg.train.trainer.accelerator,
        devices=cfg.train.trainer.devices,
        log_every_n_steps=len(train_loader) // cfg.train.log_steps_per_epoch,
        logger=mlf_logger,
        callbacks=[checkpoint_cb, earlystopping_cb],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt_path = checkpoint_cb.best_model_path

    if best_ckpt_path:
        mlf_logger.experiment.log_artifact(
            run_id=mlf_logger.run_id,
            local_path=best_ckpt_path,
            artifact_path="checkpoints",
        )

    trainer.test(model, dataloaders=test_loader, ckpt_path="best", weights_only=False)


def main(overrides: list[str] | None = None) -> None:
    cfg = get_hydra_cfg(overrides)
    train(cfg)


if __name__ == "__main__":
    main()
