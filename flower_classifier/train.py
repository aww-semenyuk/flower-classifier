"""Module for training models"""

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

    # dataloaders definition
    dataset = load_from_disk(DATA_DIR / cfg.data.dataset_output_dir)
    num_classes = dataset["train"].features["label"].num_classes

    train_loader = DataLoader(
        FlowerDatasetWrapper(dataset["train"], cfg),
        batch_size=cfg.model.train_batch_size,
        shuffle=True,
        num_workers=cfg.train.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        FlowerDatasetWrapper(dataset["val"], cfg),
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

    # logger definition
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        run_name=cfg.model.name,
        tracking_uri=cfg.logging.tracking_uri,
    )

    # callbacks definition
    callbacks_cfg = cfg.train.callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=PROJECT_DIR / "checkpoints",
        monitor=callbacks_cfg.checkpoint.monitor,
        mode=callbacks_cfg.checkpoint.mode,
        save_top_k=callbacks_cfg.checkpoint.save_top_k,
        filename=callbacks_cfg.checkpoint.filename,
    )
    earlystopping_cb = EarlyStopping(
        monitor=callbacks_cfg.early_stopping.monitor,
        mode=callbacks_cfg.early_stopping.mode,
        patience=callbacks_cfg.early_stopping.patience,
        min_delta=callbacks_cfg.early_stopping.min_delta,
    )

    # trainer
    trainer = L.Trainer(
        deterministic=True,
        max_epochs=cfg.model.epochs,
        accelerator=cfg.train.trainer.accelerator,
        devices=cfg.train.trainer.devices,
        log_every_n_steps=len(train_loader) // cfg.train.log_steps_per_epoch,
        logger=mlf_logger,
        callbacks=[checkpoint_cb, earlystopping_cb],
    )

    model = FlowerClassifierModule(num_classes=num_classes, cfg=cfg)
    mlf_logger.log_hyperparams({"commit_id": git.Repo(str(PROJECT_DIR)).head.commit.hexsha})

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # log best checkpoint as mlflow artifact
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
